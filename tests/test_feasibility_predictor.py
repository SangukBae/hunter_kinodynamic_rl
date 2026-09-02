"""Phase 5 learned feasibility predictor (plan section 9.7). Torch-gated --
skips cleanly when torch is not installed. No shipped profile enables this
predictor -- it is a standalone, optional component this test exercises
directly."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.navigation.global_rl.feasibility_predictor import (  # noqa: E402
    FeasibilityLabel, FeasibilityPredictor, FeasibilityPredictorConfig, N_PREDICTOR_SCALARS,
)


def _cfg(**overrides):
    defaults = dict(map_crop_channels=3, map_crop_size_cells=8, cnn_channels=[4], feature_dim=8)
    defaults.update(overrides)
    cfg = FeasibilityPredictorConfig(**defaults)
    cfg.validate()
    return cfg


def test_predict_returns_valid_ranges():
    predictor = FeasibilityPredictor(_cfg(), device="cpu")
    map_crop = np.random.RandomState(0).rand(3, 8, 8).astype(np.float32)
    scalars = np.zeros(N_PREDICTOR_SCALARS, dtype=np.float32)
    success_prob, expected_steps, expected_risk = predictor.predict(map_crop, scalars)
    assert 0.0 <= success_prob <= 1.0
    assert expected_steps >= 0.0
    assert 0.0 <= expected_risk <= 1.0


def test_train_step_reduces_loss_on_a_tiny_overfit_case():
    cfg = _cfg()
    predictor = FeasibilityPredictor(cfg, device="cpu")
    rng = np.random.RandomState(1)
    batch_map = rng.rand(4, cfg.map_crop_channels, cfg.map_crop_size_cells, cfg.map_crop_size_cells).astype(np.float32)
    batch_scalar = rng.rand(4, N_PREDICTOR_SCALARS).astype(np.float32)
    labels = [
        FeasibilityLabel(success=True, local_steps=20, mean_risk=0.1),
        FeasibilityLabel(success=False, local_steps=150, mean_risk=0.8),
        FeasibilityLabel(success=True, local_steps=30, mean_risk=0.2),
        FeasibilityLabel(success=False, local_steps=140, mean_risk=0.7),
    ]
    first = predictor.train_step(batch_map, batch_scalar, labels)
    last = first
    for _ in range(30):
        last = predictor.train_step(batch_map, batch_scalar, labels)
    assert last["loss"] < first["loss"]


def test_config_validate_rejects_bad_values():
    with pytest.raises(Exception):
        FeasibilityPredictorConfig(map_crop_channels=0).validate()
    with pytest.raises(Exception):
        FeasibilityPredictorConfig(cnn_channels=[]).validate()
    with pytest.raises(Exception):
        FeasibilityPredictorConfig(learning_rate=0.0).validate()


def test_checkpoint_components_exposes_net_and_optimizer():
    predictor = FeasibilityPredictor(_cfg(), device="cpu")
    components = predictor.checkpoint_components()
    assert "net" in components and "optimizer" in components
