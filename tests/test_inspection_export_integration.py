"""Full mocked-checkpoint operator journey and bounded private CSV export."""
import csv
import hashlib
from io import BytesIO, StringIO

from fastapi.testclient import TestClient
from PIL import Image

from ihsanvision.api import create_app
from ihsanvision.history import InspectionHistory
from ihsanvision.report_export import _csv_safe, export_history_csv


def png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (12, 9), color="orange").save(out, "PNG")
    return out.getvalue()


def configure_private(monkeypatch, tmp_path):
    for name in ("IHSANVISION_PADIM_CHECKPOINT", "IHSANVISION_PATCHCORE_CHECKPOINT",
                 "IHSANVISION_MODEL_REGISTRY", "IHSANVISION_APPROVED_MODELS"):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "operator.ckpt"
    checkpoint.write_bytes(b"trusted-local-fixture")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    monkeypatch.setenv("IHSANVISION_API_KEY", "operator-only-long-private-key")
    path = tmp_path / "private.sqlite3"
    monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(path))
    return checkpoint, path


def predictor(image_path, checkpoint, model):
    assert image_path.read_bytes() == png()
    return {
        "model": model, "anomaly_score": 0.71,
        "predicted_anomaly": True, "latency_ms": 7.0,
        "heatmap_png_base64": None, "model_reused": False,
    }


def test_inspection_review_lot_export_and_restart(monkeypatch, tmp_path):
    checkpoint, path = configure_private(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predictor))
    key = {"X-IhsanVision-Key": "operator-only-long-private-key"}

    assert client.get("/inspections/export.csv").status_code == 401
    response = client.post(
        "/predict?category=bottle&model=padim&lot_id=LOT-TEST-1",
        files={"file": ("sensitive-filename.png", png(), "image/png")},
        headers=key,
    )
    assert response.status_code == 200, response.text
    first = response.json()
    record_id = first["inspection_id"]
    assert first["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert len(record_id) == 32

    row = client.get("/inspections?lot_id=LOT-TEST-1", headers=key)
    assert row.status_code == 200
    assert row.json()["total"] == 1
    assert row.json()["records"][0]["operator_verdict"] is None

    update = client.patch(
        f"/inspections/{record_id}/review",
        json={"verdict": "confirmed_normal"}, headers=key
    )
    assert update.status_code == 200
    lots = client.get("/inspection-lots", headers=key).json()
    assert lots["lots"][0]["human_ai_disagreements"] == 1
    versions = client.get("/inspection-model-versions", headers=key).json()
    assert versions["groups"][0]["checkpoint_sha256"] == first["checkpoint_sha256"]

    export = client.get("/inspections/export.csv?lot_id=LOT-TEST-1", headers=key)
    assert export.status_code == 200
    assert export.headers["cache-control"] == "no-store"
    assert export.headers["x-export-rows"] == "1"
    assert export.headers["x-export-truncated"] == "false"
    parsed = list(csv.DictReader(StringIO(export.text)))
    assert len(parsed) == 1
    assert parsed[0]["id"] == record_id
    assert parsed[0]["operator_verdict"] == "confirmed_normal"
    assert parsed[0]["predicted_anomaly"] == "true"
    assert parsed[0]["lot_id"] == "LOT-TEST-1"
    for secret in ("operator-only-long-private-key", "sensitive-filename.png", str(checkpoint)):
        assert secret not in export.text

    # Re-create the API against the same private DB, simulating service restart.
    restarted = TestClient(create_app(predict_fn=predictor))
    assert restarted.get("/inspections", headers=key).json()["total"] == 1
    assert restarted.delete(f"/inspections/{record_id}", headers=key).status_code == 200
    assert restarted.get("/inspections/export.csv", headers=key).headers["x-export-rows"] == "0"
    assert path.is_file()


def test_export_limit_and_truncation(tmp_path):
    history = InspectionHistory(tmp_path / "private.sqlite3")
    for index in range(3):
        history.record(
            "bottle", "padim",
            {"anomaly_score": -0.5 + index, "predicted_anomaly": False,
             "latency_ms": 2.0, "checkpoint_sha256": "a" * 64},
            lot_id="LOT-1",
        )
    text, count, truncated = export_history_csv(history, limit=2, lot_id="LOT-1")
    assert count == 2
    assert truncated is True
    assert len(list(csv.DictReader(StringIO(text)))) == 2
    assert export_history_csv(history, limit=5, lot_id="NO-LOT")[1] == 0


def test_csv_formula_prevention_and_numeric_integrity():
    assert _csv_safe("=2+3") == "'=2+3"
    assert _csv_safe(" +SUM(A1)") == "' +SUM(A1)"
    assert _csv_safe("@cmd") == "'@cmd"
    assert _csv_safe("\t=SUM(A1)") == "'\t=SUM(A1)"
    assert _csv_safe(-0.5) == "-0.5"
    assert _csv_safe(False) == "false"
    assert _csv_safe(None) == ""


def test_export_rejects_large_limit(monkeypatch, tmp_path):
    configure_private(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=predictor))
    r = client.get("/inspections/export.csv?limit=501",
                   headers={"X-IhsanVision-Key": "operator-only-long-private-key"})
    assert r.status_code == 422


def test_disabled_history_export_is_unavailable(monkeypatch):
    monkeypatch.delenv("IHSANVISION_HISTORY_DB", raising=False)
    r = TestClient(create_app(predict_fn=predictor)).get("/inspections/export.csv")
    assert r.status_code == 503


def test_export_keeps_one_snapshot_during_concurrent_insert(monkeypatch, tmp_path):
    from contextlib import closing

    history = InspectionHistory(tmp_path / "snapshot.sqlite3")
    original_connect = history._connect
    with closing(original_connect()) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    result = {"anomaly_score": 0.5, "predicted_anomaly": False,
              "checkpoint_sha256": "a" * 64}
    expected = {history.record("bottle", "padim", result, lot_id="LOT-1")
                for _ in range(101)}
    inserted = []

    def connect_with_writer():
        connection = original_connect()

        def row_factory(cursor, row):
            # A second connection commits after the reader has started reading
            # its rows. WAL permits that write without blocking the reader.
            if len(row) > 1 and not inserted:
                with closing(original_connect()) as writer, writer:
                    new_id = "f" * 32
                    writer.execute(
                        "INSERT INTO inspections (id, created_at, category, model, "
                        "anomaly_score, lot_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (new_id, "9999-01-01", "bottle", "padim", 0.8, "LOT-1"),
                    )
                inserted.append(new_id)
            return row

        connection.row_factory = row_factory
        return connection

    monkeypatch.setattr(history, "_connect", connect_with_writer)
    text, count, truncated = export_history_csv(history, limit=101, lot_id="LOT-1")
    ids = [row["id"] for row in csv.DictReader(StringIO(text))]
    assert inserted, "The concurrent write must actually commit during export"
    assert count == 101
    assert len(set(ids)) == 101
    assert set(ids) == expected
    assert truncated is False
