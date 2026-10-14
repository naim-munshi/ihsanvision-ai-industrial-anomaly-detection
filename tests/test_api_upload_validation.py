"""Validate untrusted image uploads while keeping the event loop responsive."""
import asyncio
from io import BytesIO
import threading

import httpx
from fastapi.testclient import TestClient
from PIL import Image

import src.api as api


def image_bytes(format="PNG", size=(12, 10)):
    b = BytesIO()
    Image.new("RGB", size, (100, 140, 180)).save(b, format=format)
    return b.getvalue()


def configured_client(monkeypatch, tmp_path, fn):
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.write_bytes(b"stub")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    return TestClient(api.create_app(predict_fn=fn))


def test_jpeg_bytes_cannot_be_disguised_by_filename(monkeypatch, tmp_path):
    seen = []
    def fake_predict(image, checkpoint, model):
        seen.append((image.suffix, image.read_bytes(), model))
        return {"model": model, "anomaly_score": 0.2}
    client = configured_client(monkeypatch, tmp_path, fake_predict)
    jpeg = image_bytes("JPEG")
    result = client.post("/predict", files={"file": ("misleading.png", jpeg, "image/png")})
    assert result.status_code == 200
    assert seen == [(".jpg", jpeg, "padim")]


def test_supported_bmp_bytes_preserve_file_format(monkeypatch, tmp_path):
    seen = []
    client = configured_client(monkeypatch, tmp_path, lambda image, *_: (
        seen.append(image.suffix) or {"model": "padim", "anomaly_score": 0.3}
    ))
    result = client.post("/predict", files={
        "file": ("image.png", image_bytes("BMP"), "image/png")
    })
    assert result.status_code == 200
    assert seen == [".bmp"]


def test_gif_is_explicitly_rejected(monkeypatch, tmp_path):
    client = configured_client(monkeypatch, tmp_path, lambda *_: {"model": "padim", "anomaly_score": 0.3})
    result = client.post("/predict", files={
        "file": ("image.png", image_bytes("GIF"), "image/png")
    })
    assert result.status_code == 415
    assert "Supported formats" in result.text


def test_reject_more_than_25_million_pixels_before_inference(monkeypatch, tmp_path):
    client = configured_client(monkeypatch, tmp_path, lambda *_: (
        (_ for _ in ()).throw(AssertionError("Must reject before inference"))
    ))
    # Mode '1' uses one bit/pixel, so the fixture does not require ~100 MB.
    b = BytesIO()
    Image.new("1", (5001, 5000)).save(b, format="PNG")
    result = client.post("/predict", files={
        "file": ("large.png", b.getvalue(), "image/png")
    })
    assert result.status_code == 413


def test_health_remains_responsive_during_image_verification(monkeypatch, tmp_path):
    """The event loop must remain free even if PIL decoding takes time."""
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.write_bytes(b"stub")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    started = threading.Event()
    released = threading.Event()
    original = api._validate_image

    def slow_verify(payload):
        started.set()
        if not released.wait(3):
            raise RuntimeError("verification not released")
        return original(payload)

    monkeypatch.setattr(api, "_validate_image", slow_verify)

    async def exercise():
        app = api.create_app(predict_fn=lambda *_: {"model": "padim", "anomaly_score": 0.3})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            request = asyncio.create_task(client.post(
                "/predict", files={"file": ("input.png", image_bytes(), "image/png")}
            ))
            try:
                for _ in range(200):
                    if started.is_set():
                        break
                    await asyncio.sleep(.01)
                assert started.is_set()
                health = await asyncio.wait_for(client.get("/health"), timeout=2)
                assert health.status_code == 200
                assert health.json()["model_configured"] is True
            finally:
                released.set()
            assert (await asyncio.wait_for(request, timeout=3)).status_code == 200

    asyncio.run(exercise())
