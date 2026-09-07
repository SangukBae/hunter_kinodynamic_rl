import json
from pathlib import Path

import pytest

from hunter_kinodynamic_rl.evaluation.run_paper_comparison_campaign import (
    campaign_status, load_campaign, prepare_campaign,
)


CONFIG_ROOT = str(Path(__file__).resolve().parents[1] / "config")


def test_prepare_freezes_the_complete_11_method_5_seed_matrix(tmp_path):
    result = prepare_campaign(tmp_path / "campaign", CONFIG_ROOT)
    assert result["methods"] == [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "A7", "A8", "A9",
    ]
    assert result["seeds"] == [11, 23, 37, 53, 71]
    assert result["expected_locked_records"] == 11 * 5 * 176
    loaded, paths = load_campaign(tmp_path / "campaign", CONFIG_ROOT)
    assert loaded["campaign_manifest_sha256"] == result["campaign_manifest_sha256"]
    assert paths["scenario_manifest"].is_file()
    status = campaign_status(tmp_path / "campaign", CONFIG_ROOT)
    assert status["training_expected"] == 55
    assert status["calibration_expected"] == 20
    assert status["evaluation_expected"] == 55
    assert not status["formal_dataset_ready"]


def test_campaign_manifest_tampering_fails_before_any_run(tmp_path):
    prepare_campaign(tmp_path / "campaign", CONFIG_ROOT)
    manifest_path = tmp_path / "campaign" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seeds"] = [11]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_campaign(tmp_path / "campaign", CONFIG_ROOT)


def test_prepare_refuses_to_reuse_nonempty_campaign_root(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_campaign(root, CONFIG_ROOT)
