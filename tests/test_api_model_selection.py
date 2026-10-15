"""Offline API checks for server-side model selection and checkpoint isolation."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ihsanvision.api import create_app


def png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (9, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def clean_model_env(monkeypatch):
    for name in (
        "IHSANVISION_CHECKPOINT",
        "IHSANVISION_MODEL",
        "IHSANVISION_PADIM_CHECKPOINT",
        "IHSANVISION_PATCHCORE_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def mock_client():
    requests = []

    def fake_predict(image, checkpoint, model):
        requests.append((checkpoint, model))
        return {
            "model": model, "anomaly_score": 0.8,
            "predicted_anomaly": True, "latency_ms": 12.0,
            "heatmap_png_base64": None,
        }

    return TestClient(create_app(predict_fn=fake_predict)), requests


def test_discovery_empty_without_config(monkeypatch):
    clean_model_env(monkeypatch)
    client, _ = mock_client()
    assert client.get("/models").json() == {
        "available_models": [], "default_model": None,
    }
    assert client.get("/health").json()["model_configured"] is False


def test_legacy_single_model_still_works(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    checkpoint = tmp_path / "legacy.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    client, called = mock_client()
    assert client.get("/models").json() == {
        "available_models": ["padim"], "default_model": "padim",
    }
    result = client.post("/predict", files={"file": ("bottle.png", png(), "image/png")})
    assert result.status_code == 200
    assert result.json()["model"] == "padim"
    assert called == [(checkpoint, "padim")]


def test_two_models_dispatch_exact_checkpoint(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    padim = tmp_path / "padim.ckpt"
    patchcore = tmp_path / "patchcore.ckpt"
    padim.touch()
    patchcore.touch()
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(padim))
    monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(patchcore))
    client, called = mock_client()
    available = client.get("/models")
    assert available.status_code == 200
    assert available.json() == {
        "available_models": ["padim", "patchcore"], "default_model": "padim",
    }
    assert str(tmp_path) not in available.text
    assert client.get("/health").json()["model_configured"] is True

    for model in ("patchcore", "padim"):
        result = client.post(
            "/predict?model=" + model,
            files={"file": ("bottle.png", png(), "image/png")},
        )
        assert result.status_code == 200
        assert result.json()["model"] == model
    assert called == [(patchcore, "patchcore"), (padim, "padim")]


def test_preferred_default_patchcore_without_query(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    padim = tmp_path / "padim.ckpt"
    patchcore = tmp_path / "patchcore.ckpt"
    padim.touch()
    patchcore.touch()
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(padim))
    monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(patchcore))
    monkeypatch.setenv("IHSANVISION_MODEL", "patchcore")
    client, called = mock_client()
    assert client.get("/models").json()["default_model"] == "patchcore"
    assert client.post(
        "/predict", files={"file": ("bottle.png", png(), "image/png")}
    ).status_code == 200
    assert called == [(patchcore, "patchcore")]


def test_invalid_choice_and_unavailable_model_fail_closed(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    checkpoint = tmp_path / "padim.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(checkpoint))
    client, called = mock_client()
    files = {"file": ("bottle.png", png(), "image/png")}
    assert client.post("/predict?model=invalid", files=files).status_code == 400
    assert client.post("/predict?model=patchcore", files=files).status_code == 503
    assert called == []


def test_explicit_missing_model_does_not_fallback_to_legacy(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    checkpoint = tmp_path / "legacy.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(tmp_path / "missing.ckpt"))
    client, _ = mock_client()
    assert client.get("/models").json()["available_models"] == []
    assert client.get("/health").json()["model_configured"] is False


def test_non_checkpoint_file_not_exposed(monkeypatch, tmp_path):
    clean_model_env(monkeypatch)
    checkpoint = tmp_path / "not_a_checkpoint.txt"
    checkpoint.touch()
    monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(checkpoint))
    client, _ = mock_client()
    assert client.get("/models").json()["available_models"] == []


def test_dashboard_exposes_accessible_model_selector():
    client, _ = mock_client()
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="model-choice"' in page.text
    assert 'for="model-choice"' in page.text
    assert "/predict?model=" in page.text
