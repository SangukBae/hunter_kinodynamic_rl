"""Factorized ego-warped scene, ego and plant belief encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import BeliefState, DecisionContext, HealthState, TractorConfig, TractorInputs
from .input_adapter import LegacyObservationAdapter, goal_vector_from_tail
from .ray_lift import RayLift
from .scene_forecast import ConvGRUCell, SceneForecast
from .se2_warp import SE2HistoryWarp


def _masked_recurrent_update(
    proposal: torch.Tensor,
    prior: torch.Tensor,
    prior_valid: torch.Tensor,
    usable: torch.Tensor,
    reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(prior)
    reset_view = reset.view(reset.shape[0], *([1] * (prior.ndim - 1))).bool()
    usable_view = usable.view(usable.shape[0], *([1] * (prior.ndim - 1))).bool()
    valid_view = prior_valid.view(prior_valid.shape[0], *([1] * (prior.ndim - 1))).bool()
    base = torch.where(reset_view | ~valid_view, zero, prior)
    next_state = torch.where(usable_view, proposal, base)
    next_valid = torch.where(reset.bool(), usable.bool(), prior_valid.bool() | usable.bool())
    return next_state, next_valid


class BeliefEncoder(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        self.cfg = config
        self.adapter = LegacyObservationAdapter(config)
        self.ray_lift = RayLift(config)
        self.warp = SE2HistoryWarp(config)
        c = config.scene_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(config.t_obs * 4, c, 5, padding=2), nn.GroupNorm(4, c), nn.ELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(4, c), nn.ELU(),
        )
        self.scene_gru = ConvGRUCell(c, c)
        self.occupancy_head = nn.Conv2d(c, 4, 1)
        self.flow_head = nn.Conv2d(c, 2, 1)
        self.ego_encoder = nn.Sequential(
            nn.Linear(config.tail_dim + 9 + 4, 128), nn.ELU(), nn.Linear(128, config.ego_dim), nn.ELU()
        )
        self.response_gru = nn.GRUCell(2 + 3 + 4, config.plant_dim)
        self.response_head = nn.Linear(config.plant_dim, 3)
        self.forecast = SceneForecast(config)

    def forward(self, inputs: TractorInputs) -> tuple[BeliefState, DecisionContext]:
        inputs.validate(self.cfg)
        scans, tail = self.adapter(inputs.observation)
        evidence = self.ray_lift(scans, inputs.scan_valid)
        aligned, aligned_valid = self.warp(evidence, inputs.motion_delta, inputs.motion_valid)
        feature_input = aligned.reshape(
            scans.shape[0], self.cfg.t_obs * 4, self.cfg.grid_height, self.cfg.grid_width
        )
        aligned_mask = aligned_valid[:, :, None, None, None].expand(-1, -1, 4, -1, -1)
        feature_input = feature_input * aligned_mask.to(feature_input.dtype).reshape(
            scans.shape[0], self.cfg.t_obs * 4, 1, 1
        )
        encoded = self.backbone(feature_input)

        b = scans.shape[0]
        scene_prior = inputs.previous_scene_hidden
        if scene_prior is None:
            scene_prior = encoded.new_zeros((b, self.cfg.scene_channels, self.cfg.grid_height, self.cfg.grid_width))
        scene_prior_valid = inputs.previous_scene_hidden_valid
        if scene_prior_valid is None:
            scene_prior_valid = torch.zeros((b, 1), dtype=torch.bool, device=encoded.device)
        current_scan_usable = inputs.scan_valid[:, 0].any(dim=-1, keepdim=True)
        sensor_usable = inputs.sensor_freshness_valid.bool() & torch.isfinite(inputs.sensor_freshness_sec)
        scene_usable = current_scan_usable & sensor_usable & inputs.localization_valid.bool()
        scene_base = torch.where(
            (inputs.scene_reset.bool() | ~scene_prior_valid.bool())[:, :, None, None],
            torch.zeros_like(scene_prior), scene_prior,
        )
        scene_proposal = self.scene_gru(encoded, scene_base)
        scene_hidden, scene_hidden_valid = _masked_recurrent_update(
            scene_proposal, scene_prior, scene_prior_valid, scene_usable, inputs.scene_reset
        )
        occupancy = F.softmax(self.occupancy_head(scene_hidden), dim=1)
        observation_support = aligned[:, :, 2].amax(dim=1, keepdim=True) > 0.0
        explicit_unknown = torch.zeros_like(occupancy)
        explicit_unknown[:, 3] = 1.0
        # Front-LiDAR-only observability is a frozen conservative contract:
        # cells never covered by any valid warped ray remain exactly U,
        # rather than a learned decoder hallucinating free rear space.
        occupancy = torch.where(observation_support, occupancy, explicit_unknown)
        dynamic_flow = self.flow_head(scene_hidden)

        response = tail[:, 5:8]
        response_valid = (
            torch.isfinite(response).all(dim=-1, keepdim=True)
            & inputs.vehicle_response_valid.bool().all(dim=-1, keepdim=True)
            & inputs.previous_intent_valid.bool()
            & inputs.previous_command_valid.bool()
            & inputs.motion_valid[:, 0:1].bool()
            & (inputs.motion_delta[:, 0, 3:4] > 0.0)
        )
        response_prior = inputs.previous_response_hidden
        if response_prior is None:
            response_prior = response.new_zeros((b, self.cfg.plant_dim))
        response_prior_valid = inputs.previous_response_hidden_valid
        if response_prior_valid is None:
            response_prior_valid = torch.zeros((b, 1), dtype=torch.bool, device=response.device)
        response_base = torch.where(
            (inputs.response_reset.bool() | ~response_prior_valid.bool()),
            torch.zeros_like(response_prior), response_prior,
        )
        response_input = torch.cat(
            (inputs.previous_command_published, response, inputs.motion_delta[:, 0]), dim=-1
        )
        response_proposal = self.response_gru(response_input, response_base)
        plant_hidden, plant_hidden_valid = _masked_recurrent_update(
            response_proposal, response_prior, response_prior_valid, response_valid, inputs.response_reset
        )
        response_prediction = self.response_head(plant_hidden)

        covariance_flat = inputs.localization_covariance.reshape(b, 9)
        ego_input = torch.cat((
            tail, covariance_flat, inputs.sensor_freshness_sec,
            inputs.localization_valid.float(), inputs.localization_confidence,
            inputs.localization_confidence_valid.float(),
        ), dim=-1)
        ego_latent = self.ego_encoder(ego_input)
        future_feature, future_occupancy, future_flow = self.forecast(scene_hidden)
        future_support = observation_support[:, None].expand(
            -1, self.cfg.horizon_steps, -1, -1, -1
        )
        future_unknown = torch.zeros_like(future_occupancy)
        future_unknown[:, :, 3] = 1.0
        future_occupancy = torch.where(future_support, future_occupancy, future_unknown)
        health = HealthState(
            scene_valid=scene_hidden_valid.bool(),
            response_valid=plant_hidden_valid.bool(),
            localization_valid=(
                inputs.localization_valid.bool() & inputs.localization_confidence_valid.bool()
            ),
            sensor_valid=sensor_usable.bool(),
        )
        context = DecisionContext(
            goal_xy_m=goal_vector_from_tail(tail),
            response=response,
            response_valid=inputs.vehicle_response_valid.bool(),
            previous_command_published=inputs.previous_command_published,
            localization_covariance=inputs.localization_covariance,
            timestamp_sec=inputs.decision_timestamp_sec,
        )
        return BeliefState(
            scene_feature=scene_hidden,
            occupancy=occupancy,
            dynamic_flow=dynamic_flow,
            future_feature=future_feature,
            future_occupancy=future_occupancy,
            future_dynamic_flow=future_flow,
            ego_latent=ego_latent,
            plant_latent=plant_hidden,
            response_prediction=response_prediction,
            next_scene_hidden=scene_hidden,
            next_scene_hidden_valid=scene_hidden_valid,
            next_response_hidden=plant_hidden,
            next_response_hidden_valid=plant_hidden_valid,
            health=health,
        ), context
