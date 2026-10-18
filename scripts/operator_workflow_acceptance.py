"""Run a real authenticated operator inspection journey against localhost only.

This is an opt-in DESTRUCTIVE-TO-TEST-RECORDS acceptance check: it creates
exactly one randomly identified test inspection, reviews it, verifies private
CSV and per-lot history, then deletes only the inspection it just created.
No existing records, checkpoints or images are modified; failure cleanup is
best-effort and explicitly reported. Use a disposable/private operator DB.
"""
from __future__ import annotations

import csv
from io import StringIO
import json
import os
import re
from urllib import parse, request
from uuid import uuid4

from .dual_model_acceptance import local_origin
from .local_api_smoke import multipart_png


MAX_RESPONSE = 2 * 1024 * 1024
INSPECTION_ID = re.compile(r"[0-9a-f]{32}\Z")
CHECKPOINT_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _request_local(origin: str, route: str, *, method: str = "GET",
                   body: bytes | None = None, content_type: str | None = None) -> bytes:
    key = os.getenv("IHSANVISION_CLIENT_API_KEY")
    if not key:
        raise ValueError("A private IHSANVISION_CLIENT_API_KEY is required")
    headers = {"X-IhsanVision-Key": key, "Cache-Control": "no-store"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    call = request.Request(
        local_origin(origin) + route, data=body, headers=headers, method=method
    )
    with request.urlopen(call, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError("Operator acceptance HTTP request failed")
        raw = response.read(MAX_RESPONSE + 1)
    if len(raw) > MAX_RESPONSE:
        raise RuntimeError("Operator acceptance response exceeds size limit")
    return raw


def _json_request(origin: str, route: str, *, method: str = "GET",
                  body: bytes | None = None, content_type: str | None = None) -> dict:
    raw = _request_local(origin, route, method=method, body=body,
                         content_type=content_type)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("Operator endpoint returned a non-object JSON response")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Operator acceptance failed: " + message)


def cleanup_accepted_inspections(origin: str, ids: list[str]) -> None:
    """Delete only records created by this acceptance run, trying every ID."""
    origin = local_origin(origin)
    if len(ids) != len(set(ids)) or any(
        not isinstance(item, str) or INSPECTION_ID.fullmatch(item) is None
        for item in ids
    ):
        raise ValueError("Refusing invalid or duplicate temporary inspection IDs")
    failed = 0
    for record_id in ids:
        try:
            reply = _json_request(origin, "/inspections/" + record_id,
                                  method="DELETE")
            if reply.get("deleted") is not True or reply.get("inspection_id") != record_id:
                failed += 1
        except Exception:
            failed += 1
    if failed:
        raise RuntimeError(
            f"Could not clean {failed} of {len(ids)} temporary private inspections. "
            "Inspect the private history using the IDs retained only in process "
            "and the latest run time; never bulk-delete existing operator data."
        )


def verify_operator_workflow(origin: str, normal_png: bytes) -> dict:
    """Perform a real private end-to-end operator flow and clean up its own row."""
    origin = local_origin(origin)
    if not os.getenv("IHSANVISION_CLIENT_API_KEY"):
        raise ValueError("Operator acceptance requires IHSANVISION_CLIENT_API_KEY")
    if not normal_png or len(normal_png) > 10 * 1024 * 1024:
        raise ValueError("A nonempty normal PNG within 10 MiB is required")

    # A unique synthetic lot avoids touching or filtering an existing real lot.
    lot = "PORTFOLIO-" + uuid4().hex[:12]
    query_lot = parse.quote(lot, safe="")
    inspection_id = None
    cleanup_ok = False

    try:
        body, content_type = multipart_png(normal_png)
        prediction = _json_request(
            origin, "/predict?model=padim&category=bottle&lot_id=" + query_lot,
            method="POST", body=body, content_type=content_type,
        )
        raw_id = prediction.get("inspection_id")
        _require(isinstance(raw_id, str) and bool(INSPECTION_ID.fullmatch(raw_id)),
                 "private history did not record the test inspection")
        inspection_id = raw_id
        digest = prediction.get("checkpoint_sha256")
        _require(isinstance(digest, str) and bool(CHECKPOINT_HASH.fullmatch(digest)),
                 "missing trusted checkpoint identity")
        _require(prediction.get("model") == "padim" and
                 prediction.get("category") == "bottle", "model/category mismatch")
        _require(prediction.get("predicted_anomaly") is False,
                 "operator normal-image sample not classified as normal")

        records = _json_request(origin, "/inspections?lot_id=" + query_lot)
        _require(records.get("total") == 1 and isinstance(records.get("records"), list)
                 and len(records["records"]) == 1, "new inspection not found")
        row = records["records"][0]
        _require(row.get("id") == inspection_id and
                 row.get("checkpoint_sha256") == digest and
                 row.get("operator_verdict") is None, "initial record mismatch")
        original_score = row.get("anomaly_score")

        review = _json_request(
            origin, "/inspections/" + inspection_id + "/review",
            method="PATCH", body=b'{"verdict":"confirmed_normal"}',
            content_type="application/json",
        )
        _require(review.get("updated") is True and
                 review.get("verdict") == "confirmed_normal", "human review not stored")

        updated = _json_request(origin, "/inspections?lot_id=" + query_lot)
        _require(updated.get("total") == 1 and
                 updated["records"][0].get("operator_verdict") == "confirmed_normal"
                 and updated["records"][0].get("anomaly_score") == original_score,
                 "review altered or lost original AI observation")

        lots = _json_request(origin, "/inspection-lots?limit=100")
        _require(any(item.get("lot_id") == lot and item.get("inspected") == 1 and
                     item.get("human_normal") == 1
                     for item in lots.get("lots", [])),
                 "lot-level private summary missing test inspection")

        versions = _json_request(
            origin, "/inspection-model-versions?lot_id=" + query_lot
        )
        _require(any(item.get("checkpoint_sha256") == digest and
                     item.get("model") == "padim" and item.get("inspected") == 1
                     for item in versions.get("groups", [])),
                 "checkpoint-version traceability missing")

        raw_csv = _request_local(
            origin, "/inspections/export.csv?limit=5&lot_id=" + query_lot
        )
        rows = list(csv.DictReader(StringIO(raw_csv.decode("utf-8-sig"))))
        _require(len(rows) == 1 and rows[0].get("id") == inspection_id and
                 rows[0].get("operator_verdict") == "confirmed_normal" and
                 rows[0].get("checkpoint_sha256") == digest,
                 "private metadata CSV disagrees with recorded inspection")

        telemetry = _json_request(origin, "/operator-metrics")
        _require(telemetry.get("inference_successful", 0) >= 1,
                 "private inference telemetry did not count successful request")
        # Return only aggregate contract evidence. No raw image, score, secret,
        # internal path, checkpoint hash or synthetic lot ID in the saved report.
        return {
            "status": "operator_workflow_passed",
            "observed_workflow": [
                "authenticated_live_prediction", "checkpoint_sha256_linkage",
                "private_inspection_history", "independent_human_review",
                "production_lot_summary", "per_model_version_summary",
                "metadata_only_csv", "private_inference_telemetry",
                "test_record_cleanup",
            ],
            "sample_type": "operator_supplied_normal_bottle",
            "persisted_after_service_restart_verified": False,
            "browser_interaction_verified": False,
        }
    finally:
        if inspection_id is not None:
            try:
                deleted = _json_request(
                    origin, "/inspections/" + inspection_id, method="DELETE"
                )
                cleanup_ok = deleted.get("deleted") is True and (
                    deleted.get("inspection_id") == inspection_id
                )
                if cleanup_ok:
                    after = _json_request(origin, "/inspections?lot_id=" + query_lot)
                    cleanup_ok = after.get("total") == 0
            except Exception:
                cleanup_ok = False
            if not cleanup_ok:
                raise RuntimeError(
                    "Automatic cleanup failed; manually remove the temporary "
                    "test inspection belonging to lot " + lot + ". "
                    "Never delete other operator records."
                )
