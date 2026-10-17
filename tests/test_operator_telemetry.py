"""Operator-private telemetry contract, window bounds, and mocked API behavior."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ihsanvision.api import create_app
from ihsanvision.telemetry import InferenceTelemetry


def test_metrics_window_is_bounded_and_quantiles_are_empirical():
    metrics = InferenceTelemetry(max_samples=3)
    for elapsed in (1, 2, 3, 99):
        metrics.record(success=elapsed != 99, duration_ms=elapsed)
    result = metrics.snapshot()
    assert result["inference_completed"] == 4
    assert result["inference_successful"] == 3
    assert result["inference_failed"] == 1
    assert result["sample_window_count"] == 3
    assert result["post_admission_http_p50_ms"] == 3
    assert result["post_admission_http_p95_ms"] == 99
    assert result["scope"] == "single_process"


def test_metrics_disabled_without_private_api_key(monkeypatch):
    monkeypatch.delenv("IHSANVISION_API_KEY", raising=False)
    assert TestClient(create_app(predict_fn=lambda *_: {})).get(
        "/operator-metrics"
    ).status_code == 503


def test_private_api_metrics_include_success_and_validation_failure(monkeypatch, tmp_path):
    for key in ("IHSANVISION_MODEL_REGISTRY", "IHSANVISION_APPROVED_MODELS",
                "IHSANVISION_PADIM_CHECKPOINT", "IHSANVISION_PATCHCORE_CHECKPOINT",
                "IHSANVISION_HISTORY_DB"):
        monkeypatch.delenv(key, raising=False)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"private-marker")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    key = "really-private-operator-key"
    monkeypatch.setenv("IHSANVISION_API_KEY", key)

    def predict(_path, _checkpoint, model):
        return {
            "model": model, "anomaly_score": 0.23,
            "predicted_anomaly": False, "latency_ms": 5.0,
            "heatmap_png_base64": None,
        }

    client = TestClient(create_app(predict_fn=predict))
    assert client.get("/operator-metrics").status_code == 401
    headers = {"X-IhsanVision-Key": key}
    png = BytesIO()
    Image.new("RGB", (10, 10), "green").save(png, format="PNG")
    ok = client.post("/predict", files={"file": ("private.png", png.getvalue(), "image/png")},
                     headers=headers)
    assert ok.status_code == 200, ok.text
    failed = client.post("/predict", files={"file": ("bad.png", b"not-an-image", "image/png")},
                         headers=headers)
    assert failed.status_code == 415

    response = client.get("/operator-metrics", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["inference_completed"] == 2
    assert data["inference_successful"] == 1
    assert data["inference_failed"] == 1
    assert data["post_admission_http_p95_ms"] is not None
    for secret in (key, str(checkpoint), "private.png", "bad.png"):
        assert secret not in response.text
    assert "anomaly_score" not in response.text
    # In-memory per-process state resets on app restart (not a durable SLO store).
    fresh = TestClient(create_app(predict_fn=predict))
    assert fresh.get("/operator-metrics", headers=headers).json()["inference_completed"] == 0
