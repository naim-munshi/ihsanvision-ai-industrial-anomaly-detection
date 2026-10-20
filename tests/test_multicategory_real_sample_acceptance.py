"""Offline tests for private normal/defect five-category HTTP inference."""
import base64
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import multicategory_real_sample_acceptance as real
from scripts.local_api_smoke import make_png


def create_private_samples(base: Path, categories: tuple[str, ...]) -> None:
    for category in categories:
        folder = base / category
        folder.mkdir(parents=True)
        # Distinct PNG files: image_sample rejects byte-identical pairs.
        (folder / "normal.png").write_bytes(make_png())
        (folder / "defective.png").write_bytes(
            make_png_image((22, 44, 66)))


def make_png_image(color):
    from io import BytesIO
    from PIL import Image
    output = BytesIO()
    Image.new("RGB", (256, 256), color).save(output, format="PNG")
    return output.getvalue()


def test_real_samples_two_models_both_kinds_cleanup(monkeypatch, tmp_path):
    create_private_samples(tmp_path, ("cable",))
    deleted = []
    requests = []

    monkeypatch.setattr(
        real, "verify_catalog",
        lambda origin, categories: {"status": "config_ready"})

    def fake_request(url, *, body=None, method="GET", headers=None, timeout=30):
        path = urlsplit(url).path
        if path == "/predict":
            assert method == "POST"
            assert headers["X-IhsanVision-Key"] == "X" * 32
            assert b"image/png" in body
            qs = parse_qs(urlsplit(url).query)
            model = qs["model"][0]
            category = qs["category"][0]
            requests.append((category, model))
            idx = len(requests)
            return {
                "category": category,
                "model": model,
                "anomaly_score": 0.125,
                "predicted_anomaly": idx % 2 == 0,
                "heatmap_png_base64": base64.b64encode(make_png()).decode("ascii"),
                "model_reused": False,
                "latency_ms": 41.7,
                "inspection_id": f"{idx:032x}",
            }
        if path.startswith("/inspections/"):
            assert method == "DELETE"
            marker = path.split("/")[-1]
            deleted.append(marker)
            return {"deleted": True, "inspection_id": marker}
        raise AssertionError("Unexpected path")

    monkeypatch.setattr(real, "request_json", fake_request)
    monkeypatch.setattr(real, "cleanup_inspection", lambda origin, key, ident:
                        deleted.append(ident))

    result = real.verify_real_samples(
        "http://127.0.0.1:8004", ["cable"], tmp_path, "X" * 32)
    assert result["status"] == "real_sample_api_integration_passed"
    assert result["tested_requests"] == 4
    assert result["threshold_label_matches"] == 4
    assert result["threshold_label_mismatches"] == 0
    assert requests == [("cable", "padim"), ("cable", "padim"),
                        ("cable", "patchcore"), ("cable", "patchcore")]
    assert deleted == [f"{idx:032x}" for idx in range(1, 5)]
    assert all(row["overlay_png_valid"] for row in result["rows"])
    assert not any("png_bytes" in row or "inspection_id" in row
                   for row in result["rows"])


def test_label_mismatch_is_recorded_without_faking_accuracy(monkeypatch, tmp_path):
    create_private_samples(tmp_path, ("capsule",))
    monkeypatch.setattr(real, "verify_catalog", lambda origin, categories: None)

    def fake_request(url, *, body=None, method="GET", headers=None, timeout=30):
        qs = parse_qs(urlsplit(url).query)
        return {
            "category": qs["category"][0],
            "model": qs["model"][0],
            "anomaly_score": 0.6,
            "predicted_anomaly": False,
            "heatmap_png_base64": base64.b64encode(make_png()).decode("ascii"),
        }

    monkeypatch.setattr(real, "request_json", fake_request)
    result = real.verify_real_samples(
        "http://127.0.0.1:8004", ["capsule"], tmp_path, "X" * 32)
    assert result["status"] == "real_sample_api_integration_passed"
    assert result["threshold_label_matches"] == 2
    assert result["threshold_label_mismatches"] == 2


def test_rejects_missing_or_identical_real_samples(tmp_path):
    folder = tmp_path / "carpet"
    folder.mkdir()
    (folder / "normal.png").write_bytes(make_png())
    with pytest.raises(ValueError, match="Missing"):
        real.sample_pair(tmp_path, "carpet")
    (folder / "defective.png").write_bytes(make_png())
    with pytest.raises(ValueError, match="share bytes"):
        real.sample_pair(tmp_path, "carpet")


def test_rejects_bad_key_and_remote_host(tmp_path):
    with pytest.raises(ValueError, match="localhost"):
        real.verify_real_samples("http://remote.example:8004", ["cable"],
                                 tmp_path, "X" * 32)
    with pytest.raises(ValueError, match="operator key"):
        real.verify_real_samples("http://127.0.0.1:8004", ["cable"],
                                 tmp_path, "pasted\ncommand")
