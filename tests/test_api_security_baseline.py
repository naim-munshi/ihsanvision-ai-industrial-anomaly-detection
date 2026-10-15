"""Offline security baseline contracts: auth, admission, limits, audit privacy."""
import asyncio
from io import BytesIO
import logging

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from ihsanvision import security


def png():
    stream = BytesIO()
    Image.new("RGB", (8, 9), "white").save(stream, "PNG")
    return stream.getvalue()


def clean_env(monkeypatch):
    for name in (
        "IHSANVISION_API_KEY",
        "IHSANVISION_CHECKPOINT",
        "IHSANVISION_MODEL",
        "IHSANVISION_PADIM_CHECKPOINT",
        "IHSANVISION_PATCHCORE_CHECKPOINT",
        "IHSANVISION_RATE_LIMIT_PER_MINUTE",
        "IHSANVISION_MAX_PENDING",
        "IHSANVISION_QUEUE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def fixture_client(monkeypatch, tmp_path):
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.touch()
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(ckpt))
    calls = []

    def fake_predict(image, checkpoint, model):
        calls.append((checkpoint, model))
        return {"model": model, "anomaly_score": 0.25,
                "predicted_anomaly": False, "latency_ms": 1,
                "heatmap_png_base64": None}

    return TestClient(create_app(predict_fn=fake_predict)), calls


def predict(client, key=None):
    headers = {} if key is None else {"X-IhsanVision-Key": key}
    return client.post(
        "/predict?model=padim",
        headers=headers,
        files={"file": ("normal.png", png(), "image/png")},
    )


def test_local_unconfigured_key_keeps_backwards_compatibility(monkeypatch, tmp_path):
    clean_env(monkeypatch)
    client, calls = fixture_client(monkeypatch, tmp_path)
    result = predict(client)
    assert result.status_code == 200
    assert len(calls) == 1
    assert len(result.headers["x-request-id"]) == 32


def test_key_rejects_missing_invalid_and_empty(monkeypatch, tmp_path):
    clean_env(monkeypatch)
    monkeypatch.setenv("IHSANVISION_API_KEY", "long-local-test-key")
    client, calls = fixture_client(monkeypatch, tmp_path)
    missing = predict(client)
    wrong = predict(client, "bad")
    assert missing.status_code == wrong.status_code == 401
    assert "long-local-test-key" not in missing.text + wrong.text
    assert len(calls) == 0
    accepted = predict(client, "long-local-test-key")
    assert accepted.status_code == 200
    assert len(calls) == 1
    monkeypatch.setenv("IHSANVISION_API_KEY", "")
    assert predict(client).status_code == 401


def test_global_rate_limit_429_and_retry_after(monkeypatch, tmp_path):
    clean_env(monkeypatch)
    monkeypatch.setenv("IHSANVISION_RATE_LIMIT_PER_MINUTE", "2")
    client, calls = fixture_client(monkeypatch, tmp_path)
    assert predict(client).status_code == 200
    assert predict(client).status_code == 200
    limited = predict(client)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert len(calls) == 2


def test_request_ids_unique_and_audit_log_scrubbed(monkeypatch, tmp_path, caplog):
    clean_env(monkeypatch)
    monkeypatch.setenv("IHSANVISION_API_KEY", "TOPSECRET-CREDENTIAL")
    client, calls = fixture_client(monkeypatch, tmp_path)
    with caplog.at_level(logging.INFO, logger="ihsanvision.api"):
        a = predict(client, "TOPSECRET-CREDENTIAL")
        b = predict(client, "bad")
    assert a.status_code == 200
    assert b.status_code == 401
    assert a.headers["x-request-id"] != b.headers["x-request-id"]
    assert "TOPSECRET-CREDENTIAL" not in caplog.text
    assert "normal.png" not in caplog.text
    assert "anomaly_score" not in caplog.text
    assert "endpoint=/predict" in caplog.text
    assert len(calls) == 1


def test_web_form_avoids_persisting_auth_secrets(monkeypatch):
    clean_env(monkeypatch)
    client = TestClient(create_app(predict_fn=lambda *_: {}))
    html = client.get("/").text
    assert 'id="api-key"' in html
    assert 'autocomplete="off"' in html
    assert 'headers["X-IhsanVision-Key"]' in html
    assert 'localStorage' not in html and 'sessionStorage' not in html


def test_invalid_limit_configuration_fails_closed(monkeypatch):
    clean_env(monkeypatch)
    for name, value in (
        ("IHSANVISION_RATE_LIMIT_PER_MINUTE", "0"),
        ("IHSANVISION_MAX_PENDING", "-1"),
        ("IHSANVISION_QUEUE_TIMEOUT_SECONDS", "0"),
        ("IHSANVISION_QUEUE_TIMEOUT_SECONDS", "60"),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            create_app(predict_fn=lambda *_: {})
        monkeypatch.delenv(name)


def test_window_expires_and_state_is_bounded(monkeypatch):
    counter = [0.0]
    monkeypatch.setattr(security, "monotonic", lambda: counter[0])
    limiter = security.GlobalSlidingWindow(limit=2, interval=10)

    async def action():
        await limiter.require_slot()
        await limiter.require_slot()
        assert len(limiter._times) == 2
        with pytest.raises(Exception) as exc:
            await limiter.require_slot()
        assert exc.value.status_code == 429
        counter[0] = 11.0
        await limiter.require_slot()
        assert len(limiter._times) == 1
    asyncio.run(action())
