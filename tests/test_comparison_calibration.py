import torch

from hunter_kinodynamic_rl.evaluation.fit_comparison_calibration import calibration_metrics
from hunter_kinodynamic_rl.rl.networks.tractor.calibration import fit_platt


def test_calibration_metrics_and_platt_lineage_are_explicit():
    probability = torch.tensor([0.05, 0.15, 0.3, 0.7, 0.85, 0.95], dtype=torch.float64)
    target = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64)
    before = calibration_metrics(probability, target)
    calibration = fit_platt(
        probability, target, source_checkpoint_sha256="checkpoint",
        split_sha256="calibration-split", max_iter=20,
    )
    after = calibration_metrics(calibration.apply(probability), target)
    assert before["event_count"] == 3
    assert before["non_event_count"] == 3
    assert calibration.source_checkpoint_sha256 == "checkpoint"
    assert calibration.split_sha256 == "calibration-split"
    assert after["brier"] <= before["brier"]


def test_calibration_metrics_reject_empty_or_mismatched_inputs():
    import pytest

    with pytest.raises(ValueError):
        calibration_metrics(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError):
        calibration_metrics(torch.zeros(2), torch.zeros(3))
