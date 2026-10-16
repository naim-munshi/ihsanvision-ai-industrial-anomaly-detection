from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.api import create_app


def example_png():
    out = BytesIO()
    Image.new("RGB", (10, 10), "blue").save(out, "PNG")
    return out.getvalue()


def test_health_without_checkpoint(monkeypatch):
    monkeypatch.delenv("IHSANVISION_CHECKPOINT", raising=False)
    client = TestClient(create_app())
    assert client.get("/health").json()["model_configured"] is False


def test_inference_requires_local_checkpoint(monkeypatch):
    monkeypatch.delenv("IHSANVISION_CHECKPOINT", raising=False)
    client = TestClient(create_app())
    response = client.post("/predict", files={"file": ("test.png", example_png(), "image/png")})
    assert response.status_code == 503


def test_inference_mocked(monkeypatch, tmp_path):
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    called = []

    def fake_predict(path, checkpoint, model):
        called.append((path.read_bytes(), checkpoint, model))
        return {"model": model, "anomaly_score": 0.72,
                "predicted_anomaly": True, "latency_ms": 12.0,
                "heatmap_png_base64": None}

    client = TestClient(create_app(predict_fn=fake_predict))
    response = client.post("/predict", files={"file": ("sample.png", example_png(), "image/png")})
    assert response.status_code == 200
    assert response.json()["anomaly_score"] == 0.72
    assert called[0][1] == ckpt
    assert called[0][2] == "padim"


def test_reject_non_image(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    client = TestClient(create_app())
    r = client.post("/predict", files={"file": ("evil.png", b"not an image", "image/png")})
    assert r.status_code == 415


def test_reject_large_upload(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    client = TestClient(create_app())
    r = client.post("/predict", files={"file": ("huge.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")})
    assert r.status_code == 413


def test_demo_route():
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "IhsanVision" in response.text


def test_unknown_model_returns_clear_error_before_prediction(monkeypatch, tmp_path):
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    calls = []

    def fake_predict(*args):
        calls.append(args)
        raise AssertionError("Prediction must not run for an invalid model")

    client = TestClient(create_app(predict_fn=fake_predict))
    response = client.post(
        "/predict?model=unsupported",
        files={"file": ("example.png", example_png(), "image/png")},
    )
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]
    assert len(response.headers["X-Request-ID"]) == 32
    assert calls == []


def test_empty_upload_returns_400_without_prediction(monkeypatch, tmp_path):
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")

    def fake_predict(*args):
        raise AssertionError("Prediction must not run on empty image")

    client = TestClient(create_app(predict_fn=fake_predict))
    response = client.post(
        "/predict?model=padim",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty upload"
    assert len(response.headers["X-Request-ID"]) == 32
