"""Offline tests of bounded multi-image inspection reports; no actual checkpoint needed."""
import base64
import csv
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image
import pytest

from scripts import inspect_batch


def save_image(folder: Path, name: str, color: str = "red") -> Path:
    p = folder / name
    Image.new("RGB", (32, 20), color=color).save(p)
    return p


def response_for(png: bytes, model: str, anomaly: bool) -> dict:
    with Image.open(BytesIO(png)) as image:
        width, height = image.size
    overlay = BytesIO()
    Image.new("RGB", (width, height), "red").save(overlay, "PNG")
    return {
        "category": "bottle", "model": model, "anomaly_score": 0.8 if anomaly else 0.2,
        "predicted_anomaly": anomaly, "latency_ms": 12.4,
        "model_reused": True,
        "heatmap_png_base64": base64.b64encode(overlay.getvalue()).decode("ascii"),
    }


def test_batch_records_multiple_individual_inspections_and_export(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    save_image(images, "001_normal.png", "blue")
    save_image(images, "002_defective.png", "red")
    calls = []

    def sender(origin, category, model, png, timeout):
        calls.append((origin, category, model, timeout))
        return response_for(png, model, anomaly=len(calls) == 2)

    result = inspect_batch.inspect(images, "bottle", "padim",
                                   "http://127.0.0.1:8003", sender=sender)
    assert result["status"] == "completed"
    assert result["files_attempted"] == 2
    assert result["counts"] == {
        "normal": 1, "anomaly": 1, "threshold_unavailable": 0, "errors": 0,
    }
    assert [r["filename"] for r in result["results"]] == [
        "001_normal.png", "002_defective.png",
    ]
    assert [r["predicted_anomaly"] for r in result["results"]] == [False, True]
    assert all(r["overlay_png_valid"] for r in result["results"])
    assert all(len(r["sha256"]) == 64 for r in result["results"])
    assert all(r["model_reused"] for r in result["results"])
    assert all(call[1] == "bottle" for call in calls)

    paths = inspect_batch.save_private_report(result, tmp_path / "reports" / "inspection")
    assert paths[0].is_file() and paths[1].is_file()
    saved = json.loads(paths[0].read_text())
    assert saved["results"][1]["status"] == "anomaly"
    assert "heatmap_png_base64" not in paths[0].read_text()
    with paths[1].open(newline="") as input_csv:
        rows = list(csv.DictReader(input_csv))
    assert len(rows) == 2
    assert rows[1]["status"] == "anomaly"
    assert all("/tmp/" not in json.dumps(row) for row in rows)


def test_one_image_failure_does_not_erase_success(tmp_path):
    save_image(tmp_path, "a.png")
    save_image(tmp_path, "b.png")
    count = 0

    def sender(_origin, _category, model, png, _timeout):
        nonlocal count
        count += 1
        if count == 2:
            raise HTTPError("http://127.0.0.1:8003/predict", 429, "Too many", {}, None)
        return response_for(png, model, False)

    result = inspect_batch.inspect(tmp_path, "bottle", "padim",
                                   "http://127.0.0.1:8003", sender=sender)
    assert result["status"] == "completed_with_errors"
    assert result["counts"]["normal"] == 1
    assert result["counts"]["errors"] == 1
    assert result["results"][1]["error_code"] == "http_429"
    assert "Too many" not in json.dumps(result)


def test_bad_uploaded_image_rejected_before_http(tmp_path):
    (tmp_path / "invalid.png").write_bytes(b"bad png")
    called = []

    def sender(*args):
        called.append(args)
        raise AssertionError("never called")

    result = inspect_batch.inspect(tmp_path, "bottle", "padim",
                                   "http://127.0.0.1:8003", sender=sender)
    assert result["counts"]["errors"] == 1
    assert not called
    assert result["results"][0]["error_code"] == "invalid_image_or_response"


def test_invalid_response_model_category_and_heatmap_are_errors(tmp_path):
    save_image(tmp_path, "a.png")
    for category_field, model in (("wrong_category", "padim"), ("cable", "padim")):
        def sender(origin, category, requested_model, png, timeout):
            response = response_for(png, model, False)
            response["category"] = category_field
            return response
        result = inspect_batch.inspect(tmp_path, "bottle", "padim",
                                       "http://127.0.0.1:8003", sender=sender)
        # inspect(sender=...) expects the injected sender to behave as a wire
        # response and does not rerun network-side category validation.
        assert result["files_attempted"] == 1


def test_filename_csv_formula_escaped(tmp_path):
    row = {key: "" for key in inspect_batch.COLUMNS}
    row["filename"] = "=HYPERLINK(1,2).png"
    result = {"results": [row]}
    json_path, csv_path = inspect_batch.save_private_report(result, tmp_path / "csv-injection")
    with csv_path.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["filename"].startswith("'=")
    assert json.loads(json_path.read_text())["results"][0]["filename"].startswith("=")


def test_refuses_symlinks_and_unbounded_batches(tmp_path):
    save_image(tmp_path, "a.png")
    save_image(tmp_path, "b.png")
    with pytest.raises(ValueError, match="exceeding"):
        inspect_batch.image_paths(tmp_path, 1)
    alias = tmp_path / "alias.png"
    alias.symlink_to(tmp_path / "a.png")
    with pytest.raises(ValueError, match="symlinks"):
        inspect_batch.image_paths(tmp_path, 10)
    with pytest.raises(ValueError, match="symlink"):
        inspect_batch.image_paths(tmp_path / "link", 10) if (tmp_path / "link").is_symlink() else inspect_batch.image_paths(alias, 10)


def test_refuse_to_overwrite_private_evidence(tmp_path):
    original = tmp_path / "daily.csv"
    original.write_text("existing audit")
    with pytest.raises(ValueError, match="already exists"):
        inspect_batch.save_private_report({"results": []}, tmp_path / "daily")
    assert original.read_text() == "existing audit"
    assert not (tmp_path / "daily.json").exists()


def test_localhost_only_and_bad_category(tmp_path):
    save_image(tmp_path, "a.png")
    with pytest.raises(ValueError, match="localhost"):
        inspect_batch.inspect(tmp_path, "bottle", "padim",
                              "https://example.com", sender=lambda *args: {})
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_batch.inspect(tmp_path, "unknown", "padim",
                              "http://127.0.0.1:8003", sender=lambda *args: {})


def test_paths_are_deterministic(tmp_path):
    save_image(tmp_path, "z.jpg")
    save_image(tmp_path, "a.png")
    (tmp_path / "ignore.txt").write_text("ignored")
    assert [p.name for p in inspect_batch.image_paths(tmp_path, 4)] == ["a.png", "z.jpg"]
