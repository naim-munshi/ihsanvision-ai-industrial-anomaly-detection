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
        "predicted_anomaly", "anomaly_score", "latency_ms", "model_reused",
        "operator_verdict", "reviewed_at", "lot_id"
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


def test_review_auth_validation_and_original_prediction_immutability(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predictor))
    recorded = post(client)
    assert recorded.status_code == 200
    inspection_id = recorded.json()["inspection_id"]
    endpoint = "/inspections/" + inspection_id + "/review"
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}

    assert client.patch(endpoint, json={"verdict": "confirmed_anomaly"}).status_code == 401
    assert client.patch(endpoint, headers={"X-IhsanVision-Key": "wrong"},
                        json={"verdict": "confirmed_anomaly"}).status_code == 401
    assert client.patch(endpoint, headers=headers,
                        json={"verdict": "maybe"}).status_code == 422
    assert client.patch(endpoint, headers=headers,
                        json={"verdict": "confirmed_normal", "secret_note": "PII"}).status_code == 422
    assert client.patch("/inspections/" + "0" * 32 + "/review", headers=headers,
                        json={"verdict": "confirmed_normal"}).status_code == 404

    change = client.patch(endpoint, headers=headers, json={"verdict": "confirmed_normal"})
    assert change.status_code == 200
    assert change.json() == {"updated": True, "inspection_id": inspection_id,
                              "verdict": "confirmed_normal"}

    row = client.get("/inspections", headers=headers).json()["records"][0]
    assert row["operator_verdict"] == "confirmed_normal"
    assert row["reviewed_at"] is not None
    assert row["predicted_anomaly"] is True
    assert row["anomaly_score"] == 0.73
    assert row["model"] == "padim"

    # Simulate an application restart; human decision persists independently.
    restarted = TestClient(create_app(predict_fn=predictor))
    assert restarted.get("/inspections", headers=headers).json()["records"][0][
        "operator_verdict"
    ] == "confirmed_normal"

    assert restarted.patch(endpoint, headers=headers,
                           json={"verdict": "needs_followup"}).status_code == 200
    assert restarted.get("/inspections", headers=headers).json()["records"][0][
        "operator_verdict"
    ] == "needs_followup"
    assert restarted.patch(endpoint, headers=headers,
                           json={"verdict": None}).status_code == 200
    reset = restarted.get("/inspections", headers=headers).json()["records"][0]
    assert reset["operator_verdict"] is None
    assert reset["reviewed_at"] is None
    assert reset["predicted_anomaly"] is True


def test_review_migrates_existing_private_history_without_loss(monkeypatch, tmp_path):
    db = configure(monkeypatch, tmp_path)
    # Existing PR #28 database has no operator-review fields.
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE inspections (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                category TEXT NOT NULL, model TEXT NOT NULL,
                predicted_anomaly INTEGER, anomaly_score REAL NOT NULL,
                latency_ms REAL, model_reused INTEGER
            );
        """)
        connection.execute(
            "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?)",
            ("a" * 32, "2026-10-16T01:00:00+00:00", "bottle", "padim", 0, 0.15, 20.0, 1),
        )
    db.chmod(0o600)
    client = TestClient(create_app(predict_fn=predictor))
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    rows = client.get("/inspections", headers=headers).json()["records"]
    assert len(rows) == 1
    assert rows[0]["anomaly_score"] == 0.15
    assert rows[0]["operator_verdict"] is None
    assert client.patch("/inspections/" + "a" * 32 + "/review",
                        headers=headers, json={"verdict": "confirmed_normal"}).status_code == 200
    # Running the migration a second time must be idempotent.
    newer = TestClient(create_app(predict_fn=predictor))
    persisted = newer.get("/inspections", headers=headers).json()["records"]
    assert persisted[0]["operator_verdict"] == "confirmed_normal"
    assert persisted[0]["predicted_anomaly"] is False


def test_history_page_has_human_review_controls(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, history=False)
    response = TestClient(create_app(predict_fn=predictor)).get("/history")
    assert response.status_code == 200
    assert "Human review" in response.text
    assert "Confirm anomaly" in response.text
    assert "Needs follow-up" in response.text
    assert 'method:"PATCH"' in response.text
    assert "saveReview" in response.text
    assert 'localStorage' not in response.text.split("<script>")[-1]


def test_opt_in_lots_link_inspections_and_analyze_human_disagreements(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    decisions = iter((True, False, None, True, False))

    def variable_prediction(_image, _checkpoint, model):
        return {
            "model": model, "anomaly_score": 0.54,
            "predicted_anomaly": next(decisions),
            "latency_ms": 3.3, "model_reused": True,
        }

    client = TestClient(create_app(predict_fn=variable_prediction))
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    ids = []
    for lot in ("LOT-A01", "LOT-A01", "LOT-A01", "LOT-B02", None):
        url = "/predict?model=padim" + (f"&lot_id={lot}" if lot else "")
        res = client.post(
            url, headers=headers,
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        assert res.status_code == 200
        ids.append(res.json()["inspection_id"])

    # Human labels explicitly differ from the two known AI labels in lot A.
    assert client.patch(
        "/inspections/" + ids[0] + "/review",
        json={"verdict": "confirmed_normal"}, headers=headers,
    ).status_code == 200
    assert client.patch(
        "/inspections/" + ids[1] + "/review",
        json={"verdict": "confirmed_anomaly"}, headers=headers,
    ).status_code == 200
    assert client.patch(
        "/inspections/" + ids[2] + "/review",
        json={"verdict": "needs_followup"}, headers=headers,
    ).status_code == 200

    private = client.get("/inspections?lot_id=LOT-A01&limit=10", headers=headers)
    assert private.status_code == 200
    assert private.json()["total"] == 3
    assert private.json()["lot_id"] == "LOT-A01"
    assert all(row["lot_id"] == "LOT-A01" for row in private.json()["records"])

    another = client.get("/inspections?lot_id=LOT-B02", headers=headers)
    assert another.json()["total"] == 1
    assert client.get("/inspections", headers=headers).json()["total"] == 5

    summary = client.get("/inspection-lots", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["total"] == 2  # the unassigned fifth row is excluded
    lots = {row["lot_id"]: row for row in summary.json()["lots"]}
    assert set(lots) == {"LOT-A01", "LOT-B02"}
    assert {
        k: lots["LOT-A01"][k] for k in (
            "inspected", "ai_flagged", "ai_normal", "ai_unclassified",
            "human_normal", "human_anomaly", "needs_followup",
            "unreviewed", "human_ai_disagreements",
        )
    } == {
        "inspected": 3, "ai_flagged": 1, "ai_normal": 1, "ai_unclassified": 1,
        "human_normal": 1, "human_anomaly": 1, "needs_followup": 1,
        "unreviewed": 0, "human_ai_disagreements": 2,
    }
    assert lots["LOT-B02"]["unreviewed"] == 1
    assert lots["LOT-B02"]["human_ai_disagreements"] == 0
    assert client.get("/inspection-lots?limit=1&offset=1", headers=headers).json()["total"] == 2
    assert client.get("/inspection-lots", headers={"X-IhsanVision-Key": "bad"}).status_code == 401
    assert client.get("/inspection-lots?limit=101", headers=headers).status_code == 422

    # Data and manual verdicts survive app recreation without losing old records.
    restarted = TestClient(create_app(predict_fn=variable_prediction))
    assert restarted.get("/inspection-lots", headers=headers).json()["total"] == 2


def test_lot_format_and_disabled_history_fail_closed(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predictor))
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    assert client.post(
        "/predict?model=padim&lot_id=bad%20lot", headers=headers,
        files={"file": ("x.png", image_bytes(), "image/png")},
    ).status_code == 422
    assert client.post(
        "/predict?model=padim&lot_id=xx", headers=headers,
        files={"file": ("x.png", image_bytes(), "image/png")},
    ).status_code == 422
    assert client.post(
        "/predict?model=padim&lot_id=%3Dcmd", headers=headers,
        files={"file": ("x.png", image_bytes(), "image/png")},
    ).status_code == 422
    assert client.get("/inspections?lot_id=not%20valid", headers=headers).status_code == 422
    assert client.get("/inspections", headers=headers).json()["total"] == 0
    configure(monkeypatch, tmp_path, history=False)
    disabled = TestClient(create_app(predict_fn=predictor))
    assert disabled.post(
        "/predict?model=padim&lot_id=LOT-010",
        files={"file": ("x.png", image_bytes(), "image/png")},
    ).status_code == 503
    assert disabled.get("/inspection-lots").status_code == 503


def test_lot_migration_preserves_existing_reviewed_records(monkeypatch, tmp_path):
    db = configure(monkeypatch, tmp_path)
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE inspections (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                category TEXT NOT NULL, model TEXT NOT NULL,
                predicted_anomaly INTEGER, anomaly_score REAL NOT NULL,
                latency_ms REAL, model_reused INTEGER,
                operator_verdict TEXT, reviewed_at TEXT
            );
        """)
        connection.execute(
            "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("b"*32, "2026-10-17T01:00:00+00:00", "bottle", "patchcore",
             1, 0.91, 45.0, 1, "confirmed_anomaly", "2026-10-17T02:00:00+00:00"),
        )
    db.chmod(0o600)
    client = TestClient(create_app(predict_fn=predictor))
    headers = {"X-IhsanVision-Key": "private-test-operator-secret"}
    original = client.get("/inspections", headers=headers).json()["records"][0]
    assert original["lot_id"] is None
    assert original["operator_verdict"] == "confirmed_anomaly"
    assert original["anomaly_score"] == 0.91
    assert client.get("/inspection-lots", headers=headers).json()["total"] == 0
    restarted = TestClient(create_app(predict_fn=predictor))
    assert restarted.get("/inspections", headers=headers).json()["total"] == 1


def test_history_dashboard_has_lot_reporting_controls(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, history=False)
    html = TestClient(create_app(predict_fn=predictor)).get("/history").text
    assert 'id="lot-filter"' in html
    assert 'id="lot-summary"' in html
    assert 'id="lot-rows"' in html
    assert '"/inspection-lots?limit=100&offset=0"' in html
    assert "encodeURIComponent(key)" in html
