"""Health is liveness; /ready checks real configured paths and opted-in hashes."""
import hashlib
import json

from fastapi.testclient import TestClient

from ihsanvision.api import create_app


def setup_keyless(monkeypatch):
    for name in ("IHSANVISION_CHECKPOINT", "IHSANVISION_PADIM_CHECKPOINT",
                 "IHSANVISION_PATCHCORE_CHECKPOINT",
                 "IHSANVISION_MODEL_REGISTRY", "IHSANVISION_APPROVED_MODELS"):
        monkeypatch.delenv(name, raising=False)


def test_no_models_is_live_but_not_ready(monkeypatch):
    setup_keyless(monkeypatch)
    client = TestClient(create_app(predict_fn=lambda *_: None))
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert "checkpoint" in response.text


def test_local_checkpoint_config_ready_but_not_inference_verified(monkeypatch, tmp_path):
    setup_keyless(monkeypatch)
    file = tmp_path / "model.ckpt"
    file.write_bytes(b"offline-marker")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(file))
    response = TestClient(create_app(predict_fn=lambda *_: None)).get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "config_ready", "model_pairs": 1,
        "approved_sha256_verified": False, "inference_verified": False,
    }
    assert str(file) not in response.text


def test_empty_checkpoint_is_not_ready(monkeypatch, tmp_path):
    setup_keyless(monkeypatch)
    file = tmp_path / "model.ckpt"
    file.touch()
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(file))
    assert TestClient(create_app(predict_fn=lambda *_: None)).get("/ready").status_code == 503


def test_approved_integrity_ready_then_tampered_is_unready(monkeypatch, tmp_path):
    setup_keyless(monkeypatch)
    file = tmp_path / "model.ckpt"
    file.write_bytes(b"offline-marker")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(file))
    monkeypatch.setenv("IHSANVISION_API_KEY", "private-long-random-key")
    manifest = tmp_path / "approvals.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "approvals": [{
            "category": "bottle", "model": "padim",
            "checkpoint_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }],
    }))
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", str(manifest))
    client = TestClient(create_app(predict_fn=lambda *_: None))
    first = client.get("/ready")
    assert first.status_code == 200
    assert first.json()["approved_sha256_verified"] is True
    assert first.json()["inference_verified"] is False
    file.write_bytes(b"tampered")
    assert client.get("/ready").status_code == 503


def test_ready_requires_all_approved_pairs_to_be_configured(monkeypatch, tmp_path):
    setup_keyless(monkeypatch)
    file = tmp_path / "model.ckpt"
    file.write_bytes(b"offline-marker")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(file))
    monkeypatch.setenv("IHSANVISION_API_KEY", "private-long-random-key")
    manifest = tmp_path / "approvals.json"
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({
        "version": 1,
        "approvals": [
            {"category": "bottle", "model": "padim", "checkpoint_sha256": digest},
            {"category": "bottle", "model": "patchcore", "checkpoint_sha256": digest},
        ],
    }))
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", str(manifest))
    response = TestClient(create_app(predict_fn=lambda *_: None)).get("/ready")
    assert response.status_code == 503
