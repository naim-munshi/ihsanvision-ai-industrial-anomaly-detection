"""Service concurrency contracts with dependency-isolated model predictions."""
import asyncio
import threading
import time

from fastapi.testclient import TestClient
import httpx
from PIL import Image
from io import BytesIO

from src.api import create_app


def png_bytes():
    output = BytesIO()
    Image.new("RGB", (12, 10), "white").save(output, format="PNG")
    return output.getvalue()


def test_prediction_failure_hides_internal_exception(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted test stub")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))

    def broken_prediction(*args):
        raise RuntimeError("secret checkpoint details should not leak")

    client = TestClient(create_app(predict_fn=broken_prediction))
    response = client.post("/predict", files={"file": ("bottle.png", png_bytes(), "image/png")})
    assert response.status_code == 500
    assert "secret checkpoint" not in response.text
    assert "Inference failed" in response.text


def test_health_responsive_while_inference_in_progress(monkeypatch, tmp_path):
    """A synchronous, slow model call must not block FastAPI's event loop."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted test stub")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    started = threading.Event()
    released = threading.Event()

    def slow_predict(*args):
        started.set()
        if not released.wait(timeout=4):
            raise TimeoutError("Waiting for test release")
        return {"anomaly_score": 0.7, "model": "padim"}

    async def exercise():
        app = create_app(predict_fn=slow_predict)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            task = asyncio.create_task(
                client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")})
            )
            try:
                for _ in range(200):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert started.is_set()
                # Without threadpool offloading, the event loop would be blocked.
                assert not task.done()
                response = await asyncio.wait_for(client.get("/health"), timeout=2)
                assert response.status_code == 200
                assert response.json()["model_configured"] is True
            finally:
                released.set()
            assert (await asyncio.wait_for(task, timeout=3)).status_code == 200

    asyncio.run(exercise())


def test_concurrent_inference_is_serialized(monkeypatch, tmp_path):
    """Current Engine.predict implementation must not race on one process."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"trusted test stub")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def counted_predict(*args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.035)
        with lock:
            active -= 1
        return {"anomaly_score": 0.8}

    async def exercise():
        app = create_app(predict_fn=counted_predict)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            a, b = await asyncio.gather(
                client.post("/predict", files={"file": ("one.png", png_bytes(), "image/png")}),
                client.post("/predict", files={"file": ("two.png", png_bytes(), "image/png")}),
            )
            assert a.status_code == b.status_code == 200

    asyncio.run(exercise())
    assert max_active == 1
