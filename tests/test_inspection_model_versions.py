"""Functional private model-version reporting, using test stubs not ML weights."""
from hashlib import sha256
from io import BytesIO
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image

from ihsanvision.api import create_app


KEY = {"X-IhsanVision-Key": "private-test-operator-secret"}


def payload():
    out = BytesIO()
    Image.new("RGB", (12, 10), "white").save(out, "PNG")
    return out.getvalue()


def prepare(monkeypatch, tmp_path, history=True):
    for name in (
        "IHSANVISION_CHECKPOINT", "IHSANVISION_MODEL",
        "IHSANVISION_PADIM_CHECKPOINT", "IHSANVISION_PATCHCORE_CHECKPOINT",
        "IHSANVISION_MODEL_REGISTRY", "IHSANVISION_HISTORY_DB",
        "IHSANVISION_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    padim = tmp_path / "p.ckpt"
    patchcore = tmp_path / "c.ckpt"
    padim.write_bytes(b"version-a")
    patchcore.write_bytes(b"patchcore-a")
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(padim))
    monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(patchcore))
    db = tmp_path / "private.sqlite3"
    if history:
        monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(db))
        monkeypatch.setenv("IHSANVISION_API_KEY", KEY["X-IhsanVision-Key"])
    return padim, patchcore, db


def predict(_image, _checkpoint, model):
    return {
        "model": model,
        "anomaly_score": 0.7,
        "predicted_anomaly": True,
        "latency_ms": 10.0,
        "model_reused": True,
    }


def upload(client, model="padim", lot=None):
    query = "/predict?model=" + model + (f"&lot_id={lot}" if lot else "")
    response = client.post(query, headers=KEY,
                           files={"file": ("not-persisted.png", payload(), "image/png")})
    assert response.status_code == 200
    return response.json()


def test_private_model_version_groups_and_lot_scope(monkeypatch, tmp_path):
    padim, patchcore, db = prepare(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predict))

    a1 = upload(client, "padim", "LOT-A01")
    a2 = upload(client, "padim", "LOT-A01")
    p1 = upload(client, "patchcore", "LOT-A01")
    padim.write_bytes(b"version-b")
    a3 = upload(client, "padim", "LOT-B02")
    expected_a = sha256(b"version-a").hexdigest()
    expected_b = sha256(b"version-b").hexdigest()
    expected_patch = sha256(b"patchcore-a").hexdigest()
    assert [p["checkpoint_sha256"] for p in (a1, a2, p1, a3)] == [
        expected_a, expected_a, expected_patch, expected_b,
    ]

    # Review disagrees with one AI anomaly, without modifying AI's result.
    assert client.patch(
        "/inspections/" + a1["inspection_id"] + "/review",
        json={"verdict": "confirmed_normal"}, headers=KEY,
    ).status_code == 200
    response = client.get("/inspection-model-versions", headers=KEY)
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 3
    groups = {(x["model"], x["checkpoint_sha256"]): x for x in result["groups"]}
    assert set(groups) == {
        ("padim", expected_a), ("padim", expected_b), ("patchcore", expected_patch)
    }
    version_a = groups["padim", expected_a]
    assert version_a["inspected"] == 2
    assert version_a["ai_flagged"] == 2
    assert version_a["human_normal"] == 1
    assert version_a["human_ai_disagreements"] == 1
    assert version_a["unreviewed"] == 1
    assert groups["padim", expected_b]["inspected"] == 1
    assert groups["padim", expected_b]["human_ai_disagreements"] == 0

    lot = client.get("/inspection-model-versions?lot_id=LOT-A01", headers=KEY).json()
    assert lot["total"] == 2
    assert lot["lot_id"] == "LOT-A01"
    assert {x["checkpoint_sha256"] for x in lot["groups"]} == {
        expected_a, expected_patch
    }
    only_b = client.get(
        "/inspection-model-versions?lot_id=LOT-B02", headers=KEY
    ).json()
    assert only_b["total"] == 1
    assert only_b["groups"][0]["checkpoint_sha256"] == expected_b
    assert client.get(
        "/inspection-model-versions?limit=1&offset=1", headers=KEY
    ).json()["total"] == 3

    # No raw images, original model paths, weights, filenames or secret key
    # are returned in aggregate JSON.
    assert str(tmp_path) not in response.text
    assert "not-persisted.png" not in response.text
    assert "private-test-operator-secret" not in response.text
    assert "version-a" not in response.text
    assert "patchcore-a" not in response.text
    assert db.exists()


def test_absent_auth_invalid_query_and_disabled_history(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predict))
    assert client.get("/inspection-model-versions").status_code == 401
    assert client.get(
        "/inspection-model-versions", headers={"X-IhsanVision-Key": "incorrect"}
    ).status_code == 401
    assert client.get(
        "/inspection-model-versions?lot_id=bad%20lot", headers=KEY
    ).status_code == 422
    assert client.get(
        "/inspection-model-versions?limit=101", headers=KEY
    ).status_code == 422
    assert client.get(
        "/inspection-model-versions?offset=-1", headers=KEY
    ).status_code == 422
    prepare(monkeypatch, tmp_path, history=False)
    disabled = TestClient(create_app(predict_fn=predict))
    assert disabled.get("/inspection-model-versions").status_code == 503


def test_unknown_legacy_fingerprint_remains_separate(monkeypatch, tmp_path):
    _, _, db = prepare(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predict))
    new = upload(client)
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO inspections
               (id, created_at, category, model, predicted_anomaly,
                anomaly_score, latency_ms, model_reused)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("a" * 32, "2026-10-16T01:00:00+00:00", "bottle",
             "padim", 0, 0.1, 30.0, 1),
        )
    result = client.get("/inspection-model-versions", headers=KEY).json()
    assert result["total"] == 2
    assert {group["checkpoint_sha256"] for group in result["groups"]} == {
        None, new["checkpoint_sha256"]
    }
    unknown = next(group for group in result["groups"]
                   if group["checkpoint_sha256"] is None)
    assert unknown["inspected"] == 1
    assert unknown["ai_normal"] == 1
    assert unknown["human_ai_disagreements"] == 0


def test_operator_dashboard_exposes_model_version_summary(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path, history=False)
    page = TestClient(create_app(predict_fn=predict)).get("/history")
    assert page.status_code == 200
    assert 'id="version-summary"' in page.text
    assert 'id="version-rows"' in page.text
    assert "/inspection-model-versions?limit=100&offset=0" in page.text
    assert "encodeURIComponent(key)" in page.text
    assert "group.checkpoint_sha256" in page.text
    assert 'localStorage' not in page.text.split("<script>")[-1]
