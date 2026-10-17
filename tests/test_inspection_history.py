"""Private opt-in inspection ledger: real FastAPI requests, mocked inference only."""
from io import BytesIO
import os
from pathlib import Path
import sqlite3
import stat

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from ihsanvision.history import InspectionHistory


def image_bytes():
    data = BytesIO()
    Image.new("RGB", (24, 20), "yellow").save(data, "PNG")
    return data.getvalue()


def configure(monkeypatch, tmp_path, *, history=True):
    for env in ("IHSANVISION_CHECKPOINT", "IHSANVISION_PADIM_CHECKPOINT",
                "IHSANVISION_PATCHCORE_CHECKPOINT", "IHSANVISION_MODEL_REGISTRY",
                "IHSANVISION_HISTORY_DB", "IHSANVISION_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    model = tmp_path / "trusted.ckpt"
    model.write_bytes(b"model is a stub; we use mocked prediction")
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(model))
    if history:
        monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(tmp_path / "private.sqlite3"))
        monkeypatch.setenv("IHSANVISION_API_KEY", "private-test-operator-secret")
    return tmp_path / "private.sqlite3"


def predictor(_image, _checkpoint, model):
    return {
        "model": model, "anomaly_score": 0.73,
        "predicted_anomaly": True, "latency_ms": 8.5,
        "model_reused": False,
        "heatmap_png_base64": None,
    }


def post(client, name="private-business-image.png", headers=None):
    return client.post(
        "/predict?model=padim",
        files={"file": (name, image_bytes(), "image/png")},
        headers=headers or {"X-IhsanVision-Key": "private-test-operator-secret"},
    )


def test_disabled_history_keeps_inference_and_exposes_no_records(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, history=False)
    client = TestClient(create_app(predict_fn=predictor))
    result = post(client)
    assert result.status_code == 200
    assert result.json()["inspection_id"] is None
    assert client.get("/inspections").status_code == 503
    assert client.delete("/inspections/" + "1" * 32).status_code == 503
    assert not (tmp_path / "private.sqlite3").exists()


def test_history_requires_operator_key_and_absolute_private_path(monkeypatch, tmp_path):
    db = configure(monkeypatch, tmp_path)
    monkeypatch.delenv("IHSANVISION_API_KEY")
    with pytest.raises(ValueError, match="IHSANVISION_API_KEY"):
        create_app(predict_fn=predictor)
    monkeypatch.setenv("IHSANVISION_API_KEY", "short")
    with pytest.raises(ValueError, match="at least 16"):
        create_app(predict_fn=predictor)
    monkeypatch.setenv("IHSANVISION_API_KEY", "private-test-operator-secret")
    monkeypatch.setenv("IHSANVISION_HISTORY_DB", "relative.sqlite3")
    with pytest.raises(ValueError, match="absolute path"):
        create_app(predict_fn=predictor)
    assert not db.exists()


def test_secure_db_rejects_existing_world_readable_and_symlink(monkeypatch, tmp_path):
    path = configure(monkeypatch, tmp_path)
    path.write_bytes(b"")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        create_app(predict_fn=predictor)
    path.unlink()
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        create_app(predict_fn=predictor)


def test_record_list_pagination_auth_and_deletion(monkeypatch, tmp_path):
    db = configure(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predictor))
    assert stat.S_IMODE(db.stat().st_mode) == 0o600

    assert client.get("/inspections").status_code == 401
    assert client.get("/inspections", headers={"X-IhsanVision-Key": "bad"}).status_code == 401
    assert post(client, headers={"X-IhsanVision-Key": "bad"}).status_code == 401

    first = post(client)
    second = post(client, name="another-private-name.png")
    assert first.status_code == second.status_code == 200
    one_id, two_id = first.json()["inspection_id"], second.json()["inspection_id"]
    assert len(one_id) == len(two_id) == 32 and one_id != two_id

    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    page = client.get("/inspections?limit=1&offset=0", headers=headers)
    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert page.json()["limit"] == 1 and len(page.json()["records"]) == 1
    assert set(page.json()["records"][0]) == {
        "id", "created_at", "category", "model",
        "predicted_anomaly", "anomaly_score", "latency_ms", "model_reused"
    }
    assert page.json()["records"][0]["predicted_anomaly"] is True
    assert page.json()["records"][0]["model_reused"] is False
    assert client.get("/inspections?limit=101", headers=headers).status_code == 422
    assert client.get("/inspections?offset=-1", headers=headers).status_code == 422
    assert client.get("/inspections?limit=1&offset=1", headers=headers).json()["total"] == 2

    # Simulates server restart: the private records remain in a separate app instance.
    restarted = TestClient(create_app(predict_fn=predictor))
    assert restarted.get("/inspections", headers=headers).json()["total"] == 2

    assert client.delete("/inspections/" + first.json()["inspection_id"]).status_code == 401
    deletion = client.delete("/inspections/" + one_id, headers=headers)
    assert deletion.status_code == 200 and deletion.json()["deleted"] is True
    assert client.delete("/inspections/" + one_id, headers=headers).status_code == 404
    assert client.delete("/inspections/not-a-uuid", headers=headers).status_code == 404
    assert restarted.get("/inspections", headers=headers).json()["total"] == 1

    # No pixels, heatmaps, original filenames, model paths or operator API key persisted.
    binary = db.read_bytes()
    assert b"private-business-image" not in binary
    assert b"another-private-name" not in binary
    assert b"private-test-operator-secret" not in binary
    assert b"trusted.ckpt" not in binary
    assert image_bytes() not in binary


def test_failed_or_invalid_inference_is_never_saved(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    def bad_predict(*args):
        raise RuntimeError("checkpoint-secret-path")
    client = TestClient(create_app(predict_fn=bad_predict))
    response = post(client)
    assert response.status_code == 500
    assert "checkpoint-secret-path" not in response.text
    assert client.get("/inspections", headers=headers).json()["total"] == 0

    def invalid_predict(*args):
        return {
            "model": "padim", "anomaly_score": float("nan"),
            "predicted_anomaly": False, "latency_ms": 1,
        }
    client = TestClient(create_app(predict_fn=invalid_predict))
    assert post(client).status_code == 500
    assert client.get("/inspections", headers=headers).json()["total"] == 0


def test_model_mismatch_never_creates_history_record(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    def wrong_model(_image, _checkpoint, _model):
        return {"model": "patchcore", "anomaly_score": 0.9,
                "predicted_anomaly": True, "latency_ms": 2.0}
    client = TestClient(create_app(predict_fn=wrong_model))
    assert post(client).status_code == 500
    assert client.get("/inspections", headers={
        "X-IhsanVision-Key": "private-test-operator-secret"
    }).json()["total"] == 0


def test_history_ui_ships_metadata_only_operator_view(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, history=False)
    client = TestClient(create_app(predict_fn=predictor))
    page = client.get("/history")
    assert page.status_code == 200
    assert 'id="api-key"' in page.text
    assert 'id="rows"' in page.text
    assert 'id="load"' in page.text
    assert 'href="/batch"' in page.text
    assert '"/inspections?limit="' in page.text
    assert 'window.confirm' in page.text
    assert 'localStorage' not in page.text.split("<script>")[-1]
    assert 'sessionStorage' not in page.text.split("<script>")[-1]



def test_threshold_unavailable_remains_null(monkeypatch, tmp_path):
    db = configure(monkeypatch, tmp_path)
    def no_threshold(_image, _checkpoint, model):
        return {"model": model, "anomaly_score": 0.15,
                "predicted_anomaly": None, "latency_ms": 0.0,
                "model_reused": True}
    client = TestClient(create_app(predict_fn=no_threshold))
    assert post(client).status_code == 200
    records = client.get("/inspections", headers={
        "X-IhsanVision-Key": "private-test-operator-secret"
    }).json()["records"]
    assert records[0]["predicted_anomaly"] is None
    assert records[0]["model_reused"] is True
