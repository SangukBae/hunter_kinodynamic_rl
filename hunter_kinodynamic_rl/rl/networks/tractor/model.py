"""Composed TRACTOR-TQC encode/propose/score interface."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .actor import TractorActor
from .belief_encoder import BeliefEncoder
from .contracts import (
    BeliefState, CandidateSet, DecisionContext, ReturnRiskOutput, TractorConfig, TractorInputs,
)
from .hazard_critic import CauseTimeRiskHead
from .nominal_rollout_adapter import NominalRolloutAdapter
from .return_critic import TractorReturnCritic
from .temporal_aggregator import CandidateTemporalAggregator
from .tube_interaction import CausalTubeOccupancyInteraction
from .tube_rasterizer import TubeRasterizer


class TractorTQC(nn.Module):
    def __init__(self, config: TractorConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.belief_encoder = BeliefEncoder(config)
        self.actor = TractorActor(config)
        self.rollout = NominalRolloutAdapter(config)
        self.tube_rasterizer = TubeRasterizer(config)
        self.interaction = CausalTubeOccupancyInteraction(config)
        self.temporal_aggregator = CandidateTemporalAggregator(config)
        self.return_critic = TractorReturnCritic(config)
        self.risk_head = CauseTimeRiskHead(config)

    def encode(self, inputs: TractorInputs) -> tuple[BeliefState, DecisionContext]:
        return self.belief_encoder(inputs)

    def propose(self, belief: BeliefState) -> CandidateSet:
        return self.actor.propose(belief)

    def score(
        self, belief: BeliefState, context: DecisionContext, candidates: CandidateSet
    ) -> ReturnRiskOutput:
        rollout = self.rollout(candidates, context, belief.plant_latent)
        tube = self.tube_rasterizer(rollout)
        interaction = self.interaction(belief, rollout, tube)
        interaction = self.temporal_aggregator(interaction)
        quantiles = self.return_critic(belief, interaction)
        hazard, clearance, stopping = self.risk_head(belief, interaction)
        candidate_valid = interaction.candidate_valid.all(dim=1)
        finite = (
            torch.isfinite(quantiles).all(dim=(-1, -2))
            & torch.isfinite(hazard).all(dim=(1, 2, 4, 5))
            & torch.isfinite(clearance).all(dim=(1, 2, 4, 5))
            & torch.isfinite(stopping).all(dim=(1, 2, 4, 5))
        )
        candidate_valid = candidate_valid & finite
        return ReturnRiskOutput(quantiles, hazard, clearance, stopping, candidate_valid)

    def forward(self, inputs: TractorInputs, candidates: CandidateSet | None = None):
        belief, context = self.encode(inputs)
        proposal = self.propose(belief) if candidates is None else candidates
        return proposal, self.score(belief, context, proposal), belief, context


class TractorValueTarget(nn.Module):
    """EMA target contains the complete value path and no actor/risk head."""

    def __init__(self, online: TractorTQC):
        super().__init__()
        self.config = online.config
        self.belief_encoder = copy.deepcopy(online.belief_encoder)
        self.rollout = copy.deepcopy(online.rollout)
        self.tube_rasterizer = copy.deepcopy(online.tube_rasterizer)
        self.interaction = copy.deepcopy(online.interaction)
        self.temporal_aggregator = copy.deepcopy(online.temporal_aggregator)
        self.return_critic = copy.deepcopy(online.return_critic)

    def encode(self, inputs: TractorInputs):
        return self.belief_encoder(inputs)

    def score(self, belief: BeliefState, context: DecisionContext, candidates: CandidateSet):
        rollout = self.rollout(candidates, context, belief.plant_latent)
        tube = self.tube_rasterizer(rollout)
        interaction = self.temporal_aggregator(self.interaction(belief, rollout, tube))
        return self.return_critic(belief, interaction)
