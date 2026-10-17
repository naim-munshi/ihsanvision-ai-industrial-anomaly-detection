"""Offline checkpoint identity and per-inspection model traceability regression tests."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from ihsanvision.history import InspectionHistory
from ihsanvision.provenance import CheckpointFingerprinter


KEY = "private-test-operator-secret"
HEADERS = {"X-IhsanVision-Key": KEY}


def image_bytes():
    out = BytesIO()
    Image.new("RGB", (12, 10), "white").save(out, "PNG")
    return out.getvalue()


def setup(monkeypatch, tmp_path, *, history=True, two=False):
    for env in (
        "IHSANVISION_CHECKPOINT", "IHSANVISION_PADIM_CHECKPOINT",
        "IHSANVISION_PATCHCORE_CHECKPOINT", "IHSANVISION_HISTORY_DB",
        "IHSANVISION_API_KEY", "IHSANVISION_MODEL_REGISTRY"
    ):
        monkeypatch.delenv(env, raising=False)
    path = tmp_path / "padim.ckpt"
    path.write_bytes(b"first-trusted-test-checkpoint")
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(path))
    second = tmp_path / "patchcore.ckpt"
    if two:
        second.write_bytes(b"separate-trusted-test-checkpoint")
        monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(second))
    db = tmp_path / "private.sqlite3"
    if history:
        monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(db))
        monkeypatch.setenv("IHSANVISION_API_KEY", KEY)
    return path, second, db


def predictor(_image, _checkpoint, model):
    return {
        "model": model, "anomaly_score": 0.4,
        "predicted_anomaly": False, "latency_ms": 11.0,
        "model_reused": True,
    }


def post(client, model="padim"):
    return client.post(
        "/predict?model=" + model,
        files={"file": ("unretained-image.png", image_bytes(), "image/png")},
        headers=HEADERS,
    )


def test_hash_is_actual_file_digest_and_cached_until_metadata_changes(tmp_path):
    path = tmp_path / "trusted.ckpt"
    path.write_bytes(b"checkpoint-one")
    identifier = CheckpointFingerprinter()
    one = identifier.identify(path)
    assert one.digest == sha256(b"checkpoint-one").hexdigest()
    assert identifier.identify(path) is one
    identifier.verify_unchanged(path, one)

    replacement = tmp_path / "new.ckpt"
    replacement.write_bytes(b"checkpoint-two")
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="changed"):
        identifier.verify_unchanged(path, one)
    two = identifier.identify(path)
    assert two.digest == sha256(b"checkpoint-two").hexdigest()
    assert two != one

    path.write_bytes(b"changed-again")
    with pytest.raises(RuntimeError, match="changed"):
        identifier.verify_unchanged(path, two)


def test_private_inference_persists_real_hash_and_never_raw_bytes(monkeypatch, tmp_path):
    path, second, db = setup(monkeypatch, tmp_path, two=True)
    client = TestClient(create_app(predict_fn=predictor))
    first = post(client)
    again = post(client)
    other = post(client, model="patchcore")
    assert all(response.status_code == 200 for response in (first, again, other))
    sha_first = sha256(path.read_bytes()).hexdigest()
    sha_second = sha256(second.read_bytes()).hexdigest()
    assert first.json()["checkpoint_sha256"] == sha_first
    assert again.json()["checkpoint_sha256"] == sha_first
    assert other.json()["checkpoint_sha256"] == sha_second
    assert sha_first != sha_second

    rows = client.get("/inspections", headers=HEADERS).json()["records"]
    assert len(rows) == 3
    assert {row["checkpoint_sha256"] for row in rows} == {sha_first, sha_second}
    assert all(row["checkpoint_sha256"] and len(row["checkpoint_sha256"]) == 64 for row in rows)
    assert all(row["model"] in ("padim", "patchcore") for row in rows)

    binary = db.read_bytes()
    assert b"unretained-image.png" not in binary
    assert b"private-test-operator-secret" not in binary
    assert b"first-trusted-test-checkpoint" not in binary
    assert b"separate-trusted-test-checkpoint" not in binary
    assert str(tmp_path).encode() not in binary
    assert image_bytes() not in binary


def test_disabled_history_has_no_hash_overhead_or_db(monkeypatch, tmp_path):
    _, _, db = setup(monkeypatch, tmp_path, history=False)
    result = post(TestClient(create_app(predict_fn=predictor)))
    assert result.status_code == 200
    assert result.json()["checkpoint_sha256"] is None
    assert result.json()["inspection_id"] is None
    assert not db.exists()


def test_checkpoint_mutated_during_prediction_is_not_attributed(monkeypatch, tmp_path):
    path, _, _ = setup(monkeypatch, tmp_path)
    def mutate(_image, checkpoint, model):
        assert checkpoint == path
        checkpoint.write_bytes(b"changed-during-model-run")
        return predictor(_image, checkpoint, model)
    client = TestClient(create_app(predict_fn=mutate))
    result = post(client)
    assert result.status_code == 500
    assert "changed-during-model-run" not in result.text
    assert client.get("/inspections", headers=HEADERS).json()["total"] == 0


def test_old_lot_and_review_database_migrates_without_overwriting(monkeypatch, tmp_path):
    _, _, db = setup(monkeypatch, tmp_path)
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE inspections (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                category TEXT NOT NULL, model TEXT NOT NULL,
                predicted_anomaly INTEGER, anomaly_score REAL NOT NULL,
                latency_ms REAL, model_reused INTEGER,
                operator_verdict TEXT, reviewed_at TEXT, lot_id TEXT
            );
        """)
        con.execute(
            "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("a" * 32, "2026-10-17T01:00:00+00:00", "bottle", "padim",
             1, 0.91, 10.0, 1, "confirmed_anomaly",
             "2026-10-17T02:00:00+00:00", "LOT-004"),
        )
    db.chmod(0o600)
    client = TestClient(create_app(predict_fn=predictor))
    row = client.get("/inspections", headers=HEADERS).json()["records"][0]
    assert row["checkpoint_sha256"] is None
    assert row["operator_verdict"] == "confirmed_anomaly"
    assert row["lot_id"] == "LOT-004"
    assert row["anomaly_score"] == 0.91
    assert post(client).status_code == 200
    restarted = TestClient(create_app(predict_fn=predictor))
    rows = restarted.get("/inspections", headers=HEADERS).json()["records"]
    assert len(rows) == 2
    assert sum(row["checkpoint_sha256"] is None for row in rows) == 1


def test_history_rejects_fake_hash_and_browser_displays_legacy_unknown(monkeypatch, tmp_path):
    _, _, db = setup(monkeypatch, tmp_path)
    ledger = InspectionHistory(db)
    with pytest.raises(ValueError, match="SHA-256"):
        ledger.record("bottle", "padim", {
            "anomaly_score": 0.2, "predicted_anomaly": True,
            "checkpoint_sha256": "not-a-hash"
        })
    with pytest.raises(ValueError, match="SHA-256"):
        ledger.record("bottle", "padim", {
            "anomaly_score": 0.2, "predicted_anomaly": True
        })
    assert ledger.list_records()["total"] == 0
    page = TestClient(create_app(predict_fn=predictor)).get("/history")
    assert page.status_code == 200
    assert "Checkpoint SHA-256" in page.text
    assert "Unknown (legacy)" in page.text
    assert "version.title=digest" in page.text
