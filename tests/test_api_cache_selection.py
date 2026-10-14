"""Feature flag chooses cached model by default, with a cold inference fallback."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import src.api as api


def png_bytes():
    b = BytesIO()
    Image.new("RGB", (12, 12), "white").save(b, format="PNG")
    return b.getvalue()


def test_cache_disabled_uses_known_cold_path(monkeypatch, tmp_path):
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("IHSANVISION_MODEL_CACHE", "0")
    calls = []
    def fake_cold(image, checkpoint, model):
        calls.append((image.is_file(), checkpoint, model))
        return {"model": model, "anomaly_score": 0.5}
    monkeypatch.setattr(api, "predict_image", fake_cold)
    client = TestClient(api.create_app())
    r = client.post("/predict", files={"file": ("image.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    assert calls == [(True, ckpt, "padim")]


def test_default_predictor_uses_persistent_session(monkeypatch, tmp_path):
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(ckpt))
    monkeypatch.delenv("IHSANVISION_MODEL_CACHE", raising=False)
    created = []
    class StubSession:
        def __init__(self):
            created.append(self)
        def __call__(self, image, checkpoint, model):
            return {"model": model, "anomaly_score": 0.7, "model_reused": False}
    monkeypatch.setattr(api, "PersistentModelPredictor", StubSession)
    app = api.create_app()
    client = TestClient(app)
    for _ in range(2):
        r = client.post("/predict", files={"file": ("image.png", png_bytes(), "image/png")})
        assert r.status_code == 200
    assert len(created) == 1
