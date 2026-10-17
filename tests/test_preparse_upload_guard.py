"""Regression tests for pre-multipart authentication and bounded ASGI streams."""
import asyncio
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ihsanvision.api import create_app
from ihsanvision.upload_guard import PredictBodyGuard


def png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "green").save(output, format="PNG")
    return output.getvalue()


def test_key_required_before_multipart_parse(monkeypatch):
    monkeypatch.setenv("IHSANVISION_API_KEY", "secure-key-for-private-demo")
    client = TestClient(create_app(predict_fn=lambda *_: None))
    response = client.post(
        "/predict", data=b"not a multipart body",
        headers={"Content-Type": "multipart/form-data; boundary=invalid"},
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


def test_valid_key_runs_normal_prediction(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    monkeypatch.setenv("IHSANVISION_API_KEY", "secure-key-for-private-demo")
    called = []

    def predict(path, weight, model):
        called.append(path.read_bytes())
        return {
            "model": model,
            "anomaly_score": 0.23,
            "predicted_anomaly": False,
            "latency_ms": 1.0,
            "heatmap_png_base64": None,
        }

    client = TestClient(create_app(predict_fn=predict))
    response = client.post(
        "/predict",
        files={"file": ("sample.png", png(), "image/png")},
        headers={"X-IhsanVision-Key": "secure-key-for-private-demo"},
    )
    assert response.status_code == 200, response.text
    assert len(called) == 1
    assert called[0] == png()


def test_wrong_key_never_invokes_model(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_API_KEY", "secure-key-for-private-demo")
    calls = []

    client = TestClient(create_app(predict_fn=lambda *_: calls.append(1)))
    response = client.post(
        "/predict", files={"file": ("sample.png", png(), "image/png")},
        headers={"X-IhsanVision-Key": "wrong"},
    )
    assert response.status_code == 401
    assert calls == []


def test_reject_oversized_content_length_before_parser(monkeypatch):
    monkeypatch.delenv("IHSANVISION_API_KEY", raising=False)
    client = TestClient(create_app(predict_fn=lambda *_: None))
    result = client.post("/predict", content=b"x" * (10 * 1024 * 1024 + 65537),
                         headers={"Content-Type": "application/octet-stream"})
    assert result.status_code == 413


def test_chunked_body_without_length_is_bounded(monkeypatch):
    monkeypatch.delenv("IHSANVISION_API_KEY", raising=False)
    invoked = []
    sent = []
    messages = iter([
        {"type": "http.request", "body": b"aaaa", "more_body": True},
        {"type": "http.request", "body": b"bbbbb", "more_body": False},
    ])

    async def app(scope, receive, send):
        invoked.append(True)

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(PredictBodyGuard(app, max_body_bytes=8)(
        {"type": "http", "method": "POST", "path": "/predict", "headers": []},
        receive, send,
    ))
    assert invoked == []
    assert sent[0]["status"] == 413


def test_reject_duplicate_key_headers_before_parser(monkeypatch):
    monkeypatch.setenv("IHSANVISION_API_KEY", "secure-key-for-private-demo")
    client = TestClient(create_app(predict_fn=lambda *_: None))
    result = client.post("/predict", content=b"",
                         headers=[("X-IhsanVision-Key", "secure-key-for-private-demo"),
                                  ("X-IhsanVision-Key", "secure-key-for-private-demo")])
    assert result.status_code == 400
