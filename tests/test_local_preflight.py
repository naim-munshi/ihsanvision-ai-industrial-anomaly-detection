import json

import pytest

from scripts.local_preflight import check_artifacts, environment_summary


def test_local_artifact_contract(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"trusted fake checkpoint")
    splits = {
        "train": ["/kaggle/input/train/0.png"],
        "val": ["/kaggle/input/test/1.png"],
        "test": ["/kaggle/input/test/2.png"],
    }
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    (tmp_path / "report.json").write_text(json.dumps({
        "model": "padim", "category": "bottle",
        "split_counts": {"train": 1, "val": 1, "test": 1},
        "evaluation": {"metrics": {"image_AUROC": 0.99}},
    }))
    (tmp_path / "smoke_report.json").write_text(json.dumps({
        "status": "single-image-inference-passed",
        "split_counts": {"train": 1, "val": 1, "test": 1},
    }))
    (tmp_path / "smoke_overlay.png").write_bytes(b"visualization stub")
    result = check_artifacts(ckpt)
    assert result["artifacts_complete"]
    assert result["model"] == "padim"
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["metric_names"] == ["image_AUROC"]


def test_local_preflight_detects_missing_artifacts(tmp_path):
    result = check_artifacts(tmp_path / "missing.ckpt")
    assert not result["checkpoint_exists"]
    assert not result["artifacts_complete"]


def test_local_preflight_detects_overlap(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({
        "train": ["overlap"], "val": ["overlap"], "test": ["other"],
    }))
    with pytest.raises(ValueError, match="leakage"):
        check_artifacts(tmp_path / "model.ckpt")


def test_local_preflight_detects_report_mismatch(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({
        "train": ["1"], "val": ["2"], "test": ["3"],
    }))
    (tmp_path / "report.json").write_text(json.dumps({
        "split_counts": {"train": 1, "val": 5, "test": 1},
    }))
    with pytest.raises(ValueError, match="disagree"):
        check_artifacts(tmp_path / "model.ckpt")


def test_environment_summary_has_expected_keys():
    result = environment_summary()
    assert "python" in result
    assert "torch" in result["packages"]
    assert "dependencies_present" in result
