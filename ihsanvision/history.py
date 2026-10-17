"""Opt-in, private SQLite inspection ledger for a single local IhsanVision operator.

No raw image, filenames, image hashes, heatmap bytes, checkpoint paths or
credentials are persisted. The only records are successful model predictions.
This is not a multi-tenant database, immutable compliance log or production
event store. An operator must explicitly configure an absolute private path
and an API key.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from uuid import uuid4


_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    model TEXT NOT NULL,
    predicted_anomaly INTEGER,
    anomaly_score REAL NOT NULL,
    latency_ms REAL,
    model_reused INTEGER,
    operator_verdict TEXT,
    reviewed_at TEXT,
    lot_id TEXT,
    checkpoint_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS inspections_created_at
ON inspections(created_at DESC, id DESC);
"""
_COLUMNS = ("id", "created_at", "category", "model",
            "predicted_anomaly", "anomaly_score", "latency_ms", "model_reused",
            "operator_verdict", "reviewed_at", "lot_id", "checkpoint_sha256")
_OPERATOR_VERDICTS = {"confirmed_normal", "confirmed_anomaly", "needs_followup"}
_ID = re.compile(r"[0-9a-f]{32}\Z")
_LOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,39}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _optional_bool(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{field} must be bool or null")
    return int(value)


def _finite(value: object, field: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return float(value)


class InspectionHistory:
    """Short, transactional metadata records with explicit local opt-in."""

    def __init__(self, path: Path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            raise ValueError("IHSANVISION_HISTORY_DB must be an absolute path")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise ValueError("Create a private, non-symlink history directory first")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("History database cannot be a symlink or nonregular file")
        if path.exists():
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise ValueError("History database must be private (chmod 600)")
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                raise ValueError("History database appeared concurrently; retry safely") from None
            else:
                os.close(descriptor)
        self.path = path
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            # Preserve PR #28 databases without discarding existing evidence.
            # SQLite ADD COLUMN is safe and idempotent for these nullable fields.
            existing = {
                info[1] for info in connection.execute("PRAGMA table_info(inspections)")
            }
            for column in ("operator_verdict", "reviewed_at", "lot_id", "checkpoint_sha256"):
                if column not in existing:
                    connection.execute(f"ALTER TABLE inspections ADD COLUMN {column} TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS inspections_lot_created "
                "ON inspections(lot_id, created_at DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    @staticmethod
    def validate_lot(lot_id: str | None) -> str | None:
        """Opaque non-PII operator lot key; no freeform names or paths."""
        if lot_id is None:
            return None
        if not isinstance(lot_id, str) or not _LOT.fullmatch(lot_id):
            raise ValueError(
                "Lot ID must be 3-40 ASCII characters: letters, digits, hyphen, underscore"
            )
        return lot_id

    def record(self, category: str, model: str, result: dict,
               lot_id: str | None = None) -> str:
        """Only persist validated successful inference metadata."""
        if not isinstance(category, str) or not category or not isinstance(model, str) or not model:
            raise ValueError("Missing model or category")
        lot_id = self.validate_lot(lot_id)
        score = _finite(result.get("anomaly_score"), "anomaly_score")
        latency_raw = result.get("latency_ms")
        latency = None if latency_raw is None else _finite(
            latency_raw, "latency_ms", nonnegative=True
        )
        prediction = _optional_bool(result.get("predicted_anomaly"), "predicted_anomaly")
        reused = _optional_bool(result.get("model_reused"), "model_reused")
        digest = result.get("checkpoint_sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("History requires a verified SHA-256 checkpoint fingerprint")
        record_id = uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO inspections
                   (id, created_at, category, model, predicted_anomaly,
                    anomaly_score, latency_ms, model_reused, lot_id, checkpoint_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, timestamp, category, model, prediction, score, latency, reused, lot_id, digest),
            )
        return record_id

    @staticmethod
    def _row(row: tuple) -> dict:
        record = dict(zip(_COLUMNS, row))
        for field in ("predicted_anomaly", "model_reused"):
            value = record[field]
            record[field] = None if value is None else bool(value)
        return record

    def list_records(self, limit: int = 50, offset: int = 0,
                     lot_id: str | None = None) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 1_000_000:
            raise ValueError("offset must be between 0 and 1000000")
        lot_id = self.validate_lot(lot_id)
        where = " WHERE lot_id = ?" if lot_id is not None else ""
        params: tuple = (lot_id,) if lot_id is not None else ()
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM inspections" + where, params
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM inspections" + where
                + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return {
            "records": [self._row(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "lot_id": lot_id,
        }

    def list_lots(self, limit: int = 50, offset: int = 0) -> dict:
        """Read-only per-lot counts; never turn AI predictions into ground truth."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 1_000_000:
            raise ValueError("offset must be between 0 and 1000000")
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(DISTINCT lot_id) FROM inspections WHERE lot_id IS NOT NULL"
            ).fetchone()[0]
            rows = connection.execute(
                """SELECT lot_id, COUNT(*) AS inspected,
                          MAX(created_at) AS last_inspection_utc,
                          SUM(CASE WHEN predicted_anomaly = 1 THEN 1 ELSE 0 END) AS ai_flagged,
                          SUM(CASE WHEN predicted_anomaly = 0 THEN 1 ELSE 0 END) AS ai_normal,
                          SUM(CASE WHEN predicted_anomaly IS NULL THEN 1 ELSE 0 END) AS ai_unclassified,
                          SUM(CASE WHEN operator_verdict = 'confirmed_normal'
                                   THEN 1 ELSE 0 END) AS human_normal,
                          SUM(CASE WHEN operator_verdict = 'confirmed_anomaly'
                                   THEN 1 ELSE 0 END) AS human_anomaly,
                          SUM(CASE WHEN operator_verdict = 'needs_followup'
                                   THEN 1 ELSE 0 END) AS needs_followup,
                          SUM(CASE WHEN operator_verdict IS NULL
                                   THEN 1 ELSE 0 END) AS unreviewed,
                          SUM(CASE WHEN
                            (operator_verdict = 'confirmed_normal' AND predicted_anomaly = 1)
                            OR (operator_verdict = 'confirmed_anomaly' AND predicted_anomaly = 0)
                            THEN 1 ELSE 0 END) AS human_ai_disagreements
                   FROM inspections WHERE lot_id IS NOT NULL
                   GROUP BY lot_id
                   ORDER BY last_inspection_utc DESC, lot_id ASC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        fields = (
            "lot_id", "inspected", "last_inspection_utc", "ai_flagged",
            "ai_normal", "ai_unclassified", "human_normal", "human_anomaly",
            "needs_followup", "unreviewed", "human_ai_disagreements"
        )
        return {
            "lots": [dict(zip(fields, row)) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def update_review(self, record_id: str, verdict: str | None) -> bool:
        """Human adjudication is separate from immutable model prediction fields.

        A null verdict clears the previous local review. No reviewer identity,
        freeform notes or other personal details are persisted.
        """
        if not isinstance(record_id, str) or not _ID.fullmatch(record_id):
            return False
        if verdict is not None and verdict not in _OPERATOR_VERDICTS:
            raise ValueError("Unsupported operator review verdict")
        reviewed_at = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            if verdict is not None else None
        )
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE inspections
                   SET operator_verdict = ?, reviewed_at = ?
                   WHERE id = ?""",
                (verdict, reviewed_at, record_id),
            )
            return result.rowcount == 1

    def delete_record(self, record_id: str) -> bool:
        if not isinstance(record_id, str) or not _ID.fullmatch(record_id):
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM inspections WHERE id = ?", (record_id,)
            )
            return cursor.rowcount == 1
