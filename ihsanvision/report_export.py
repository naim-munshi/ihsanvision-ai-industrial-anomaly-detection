"""Bounded, spreadsheet-safe private inspection metadata export.

No images, filenames, model paths, credentials, heatmaps or base64 content.
Human verdicts remain separate from raw model predictions.
"""
from __future__ import annotations

import csv
from io import StringIO

EXPORT_COLUMNS = (
    "id", "created_at", "lot_id", "category", "model",
    "predicted_anomaly", "anomaly_score", "latency_ms",
    "model_reused", "operator_verdict", "reviewed_at",
    "checkpoint_sha256",
)
MAX_EXPORT_ROWS = 500


def _csv_safe(value) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    text = str(value)
    # Prevent spreadsheet formula execution even if a corrupted historic
    # record contains a malicious string. CSV escaping alone is insufficient.
    if text.lstrip().startswith(("=", "+", "-", "@")) or (
        text and text[0] in ("\t", "\r", "\n")
    ):
        return "'" + text
    return text


def export_history_csv(history, *, limit: int = 100, lot_id: str | None = None) -> tuple[str, int, bool]:
    """Return a bounded CSV plus rows written and whether records were omitted."""
    if type(limit) is not int or not 1 <= limit <= MAX_EXPORT_ROWS:
        raise ValueError("Export limit must be 1-500")
    lot_id = history.validate_lot(lot_id)
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(EXPORT_COLUMNS)
    written, total = 0, None
    while written < limit:
        batch = history.list_records(limit=min(100, limit - written),
                                     offset=written, lot_id=lot_id)
        if total is None:
            total = batch["total"]
        rows = batch["records"]
        if not rows:
            break
        for row in rows:
            writer.writerow([_csv_safe(row.get(column)) for column in EXPORT_COLUMNS])
        written += len(rows)
        if len(rows) < min(100, limit - (written - len(rows))):
            break
    return buffer.getvalue(), written, bool(total and total > written)
