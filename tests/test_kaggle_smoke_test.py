"""Offline tests: inferencer is stubbed; real model execution is separate."""
from io import BytesIO
from pathlib import Path
import base64
import json

from PIL import Image
import pytest

from scripts.kaggle_smoke_test import choose_defective_image, verify_checkpoint


def prepare(tmp_path: Path):
    data = tmp_path / "dataset"
    train = data / "bottle" / "train" / "good" / "000.png"
    val = data / "bottle" / "test" / "broken" / "001.png"
    test = data / "bottle" / "test" / "broken" / "002.png"
    for file in [train, val, test]:
        file.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(file)
    run = tmp_path / "runs" / "padim-bottle-seed42"
    run.mkdir(parents=True)
    (run / "model.ckpt").write_bytes(b"pretend trusted checkpoint")
    (run / "report.json").write_text(json.dumps({"evaluation": {"metrics": {}}}))
    (run / "splits.json").write_text(json.dumps({
        "train": [str(train)], "val": [str(val)], "test": [str(test)]
    }))
    return data, run


def test_smoke_test_writes_overlay_and_summary(tmp_path):
    data, run = prepare(tmp_path)
    png = BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    seen = []

    def stub(image, checkpoint, model, accelerator):
        seen.append((image, checkpoint, model, accelerator))
        return {
            "anomaly_score": 0.7,
            "predicted_anomaly": True,
            "latency_ms": 123.4,
            "heatmap_png_base64": base64.b64encode(png.getvalue()).decode(),
        }

    result = verify_checkpoint(data, tmp_path / "runs", predictor=stub)
    assert result["status"] == "single-image-inference-passed"
    # The first alphabetically discovered anomaly is a VALIDATION sample.
    # Smoke inference must instead use the independently held-out TEST sample.
    assert seen[0][0] == data / "bottle" / "test" / "broken" / "002.png"
    assert seen[0][1] == run / "model.ckpt"
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert (run / "smoke_overlay.png").is_file()
    assert json.loads((run / "smoke_report.json").read_text())["anomaly_score"] == 0.7


def test_refuses_missing_checkpoint(tmp_path):
    data, run = prepare(tmp_path)
    (run / "model.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="model.ckpt"):
        verify_checkpoint(data, tmp_path / "runs", predictor=lambda *args: None)


def test_refuses_overlapping_splits(tmp_path):
    data, run = prepare(tmp_path)
    (run / "splits.json").write_text(json.dumps({
        "train": ["same"], "val": ["same"], "test": ["different"]
    }))
    with pytest.raises(ValueError, match="Data leakage"):
        verify_checkpoint(data, tmp_path / "runs", predictor=lambda *args: None)


def test_pick_defective_only(tmp_path):
    data, _ = prepare(tmp_path)
    chosen = choose_defective_image(data, "bottle")
    assert chosen.parent.name != "good"



def test_smoke_fails_closed_if_only_validation_has_anomalies(tmp_path):
    data, run = prepare(tmp_path)
    # All defective images in the held-out test directory were assigned to
    # validation; do not silently fall back to the validation split.
    test = data / "bottle" / "test" / "broken" / "002.png"
    good = data / "bottle" / "test" / "good" / "004.png"
    good.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(good)
    (run / "splits.json").write_text(json.dumps({
        "train": [str(data / "bottle" / "train" / "good" / "000.png")],
        "val": [str(data / "bottle" / "test" / "broken" / "001.png")],
        "test": [str(good)],
    }))
    called = []
    def predictor(*args):
        called.append(args)
        raise AssertionError("Inference must never run on validation data")

    with pytest.raises(FileNotFoundError, match="held-out defective"):
        verify_checkpoint(data, tmp_path / "runs", predictor=predictor)
    assert called == []


def test_smoke_detects_duplicate_split_paths(tmp_path):
    data, run = prepare(tmp_path)
    val = str(data / "bottle" / "test" / "broken" / "001.png")
    (run / "splits.json").write_text(json.dumps({
        "train": ["train.png"], "val": [val, val], "test": ["test.png"],
    }))
    with pytest.raises(ValueError, match="Duplicate paths"):
        verify_checkpoint(data, tmp_path / "runs", predictor=lambda *args: None)
