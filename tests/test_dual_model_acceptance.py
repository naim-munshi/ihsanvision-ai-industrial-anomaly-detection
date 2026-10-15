"""Offline contract tests for the real local two-model acceptance harness."""
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts import dual_model_acceptance as acceptance


def sample_paths(tmp_path: Path) -> tuple[Path, Path]:
    normal = tmp_path / "normal_bottle.png"
    defect = tmp_path / "defective_bottle.png"
    Image.new("RGB", (18, 14), "white").save(normal, format="PNG")
    Image.new("RGB", (18, 14), "black").save(defect, format="PNG")
    return normal, defect


def overlay_for(png_bytes: bytes) -> str:
    with Image.open(BytesIO(png_bytes)) as image:
        dims = image.size
    out = BytesIO()
    Image.new("RGB", dims, "red").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def stub_model_api(monkeypatch, *, miss_defective=False):
    calls = []
    monkeypatch.setattr(acceptance, "discover_models", lambda origin: ("padim", "patchcore"))

    def fake_predict(origin, model, png_bytes, timeout):
        with Image.open(BytesIO(png_bytes)) as image:
            # Normal fixture is white, defective fixture is black.
            defective = image.convert("RGB").getpixel((0, 0)) == (0, 0, 0)
        calls.append((model, defective))
        return {
            "model": model,
            "anomaly_score": 0.6 if defective else 0.1,
            "predicted_anomaly": defective and not (miss_defective and model == "patchcore"),
            "latency_ms": 12.3,
            "model_reused": bool(len(calls) % 2 == 0),
            "heatmap_png_base64": overlay_for(png_bytes),
        }

    monkeypatch.setattr(acceptance, "predict_png", fake_predict)
    return calls


def test_four_model_image_combinations_pass_without_binary_leak(monkeypatch, tmp_path):
    normal, defect = sample_paths(tmp_path)
    calls = stub_model_api(monkeypatch)
    result = acceptance.accept("http://127.0.0.1:8003", normal, defect)
    assert result["status"] == "passed"
    assert calls == [
        ("padim", False), ("padim", True),
        ("patchcore", False), ("patchcore", True),
    ]
    assert all(r["overlay_png_valid"] and r["predicted_as_expected"] for r in result["results"])
    assert len(result["results"]) == 4
    serialized = json.dumps(result)
    assert "heatmap_png_base64" not in serialized
    assert str(tmp_path) not in serialized
    assert result["sample_evidence"]["normal"]["dimensions"] == [18, 14]
    assert len(result["sample_evidence"]["defective"]["sha256"]) == 64


def test_missed_defect_is_recorded_not_misrepresented(monkeypatch, tmp_path):
    normal, defect = sample_paths(tmp_path)
    stub_model_api(monkeypatch, miss_defective=True)
    result = acceptance.accept("http://localhost:8003/", normal, defect)
    assert result["status"] == "label_mismatch"
    failures = [r for r in result["results"] if not r["predicted_as_expected"]]
    assert [(r["model"], r["sample"]) for r in failures] == [("patchcore", "defective")]


def test_only_one_available_model_refuses_incomplete_comparison(monkeypatch, tmp_path):
    normal, defect = sample_paths(tmp_path)
    monkeypatch.setattr(acceptance, "discover_models", lambda _: ("padim",))
    with pytest.raises(RuntimeError, match="Both padim and patchcore"):
        acceptance.accept("http://127.0.0.1:8003", normal, defect)


def test_identical_images_refused(monkeypatch, tmp_path):
    normal, _ = sample_paths(tmp_path)
    monkeypatch.setattr(acceptance, "discover_models", lambda _: ("padim", "patchcore"))
    with pytest.raises(ValueError, match="identical"):
        acceptance.accept("http://127.0.0.1:8003", normal, normal)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8003",
        "http://example.com:8003",
        "http://user:pass@localhost:8003",
        "http://localhost:8003/evil",
        "http://localhost:8003?model=padim",
        "http://localhost",
        "http://127.0.0.1:invalid",
    ],
)
def test_endpoint_restricted_to_local_service(url):
    with pytest.raises(ValueError):
        acceptance.local_origin(url)


def test_reject_wrong_model_and_invalid_anomaly_values():
    sample = {
        "model": "patchcore",
        "anomaly_score": 0.7,
        "predicted_anomaly": True,
        "latency_ms": 1.2,
        "heatmap_png_base64": overlay_for(make_png_fixture()),
    }
    with pytest.raises(RuntimeError, match="Model mismatch"):
        acceptance.validate_prediction(sample, "padim", (18, 14))
    for bad in (float("nan"), float("inf"), "0.7", True):
        with pytest.raises(RuntimeError, match="anomaly score"):
            acceptance.validate_prediction({**sample, "anomaly_score": bad}, "patchcore", (18, 14))


def make_png_fixture():
    stream = BytesIO()
    Image.new("RGB", (18, 14), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_reject_bad_overlay_or_dimensions():
    base = {
        "model": "padim",
        "anomaly_score": 0.5,
        "predicted_anomaly": False,
        "latency_ms": 1.2,
        "heatmap_png_base64": "notbase64",
    }
    with pytest.raises(RuntimeError, match="Malformed anomaly overlay"):
        acceptance.validate_prediction(base, "padim", (18, 14))
    with pytest.raises(RuntimeError, match="Heatmap must be a PNG"):
        acceptance.validate_prediction(
            {**base, "heatmap_png_base64": overlay_for(make_png_fixture())},
            "padim", (5, 5),
        )


def test_existing_report_is_preserved(monkeypatch, tmp_path):
    normal, defect = sample_paths(tmp_path)
    output = tmp_path / "evidence.json"
    output.write_text('{"historic": true}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        acceptance.main([
            "--normal", str(normal),
            "--defective", str(defect),
            "--output", str(output),
        ])
    assert exc.value.code == 2
    assert json.loads(output.read_text()) == {"historic": True}


def test_completed_cli_saves_evidence_and_missing_label_exits_failure(monkeypatch, tmp_path):
    normal, defect = sample_paths(tmp_path)
    stub_model_api(monkeypatch, miss_defective=True)
    output = tmp_path / "evidence.json"
    with pytest.raises(SystemExit) as exc:
        acceptance.main([
            "--normal", str(normal),
            "--defective", str(defect),
            "--output", str(output),
        ])
    assert exc.value.code == 1
    saved = json.loads(output.read_text())
    assert saved["status"] == "label_mismatch"
    assert len(saved["results"]) == 4
