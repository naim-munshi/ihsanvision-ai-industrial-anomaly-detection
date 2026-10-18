"""One-command REAL localhost stack acceptance (no training, no model loading here).

Requires an already-running Bottle PaDiM/PatchCore API and two operator-owned
sample images. Checks liveness, config readiness, both dashboards, model routing,
and actual four-prediction inference/PNG overlays with an existing harness.
This is NOT multi-category accuracy, stress testing or industrial certification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib import error, request

from .dual_model_acceptance import accept, image_sample, local_origin
from .operator_workflow_acceptance import verify_operator_workflow
from .local_api_smoke import make_png, multipart_png

MAX_UI_BYTES = 2 * 1024 * 1024


def _get(origin: str, path: str) -> bytes:
    with request.urlopen(origin + path, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Local {path} failed")
        payload = response.read(MAX_UI_BYTES + 1)
    if len(payload) > MAX_UI_BYTES:
        raise RuntimeError(f"Local {path} response exceeds bounded acceptance size")
    return payload


def assert_unauthenticated_prediction_denied(origin: str) -> None:
    """Verify key enforcement occurs before model work; no operator key is sent."""
    body, content_type = multipart_png(make_png())
    call = request.Request(
        origin + "/predict",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with request.urlopen(call, timeout=10):
            pass
    except error.HTTPError as exc:
        if exc.code == 401:
            return
        raise RuntimeError("Unauthenticated prediction was not rejected with 401") from None
    raise RuntimeError("Unauthenticated prediction unexpectedly succeeded")


def verify_stack(origin: str, normal: Path, defective: Path, *,
                 timeout: float = 180, require_auth: bool = False,
                 check_operator: bool = False) -> dict:
    origin = local_origin(origin)
    if check_operator and not require_auth:
        raise ValueError("Operator workflow requires --require-auth and a private DB")
    health = json.loads(_get(origin, "/health"))
    ready = json.loads(_get(origin, "/ready"))
    catalog = json.loads(_get(origin, "/models"))
    if health.get("status") != "ok" or health.get("model_configured") is not True:
        raise RuntimeError("Local API is not live with configured weights")
    if ready.get("status") != "config_ready" or ready.get("model_pairs", 0) < 2:
        raise RuntimeError("Checkpoint readiness did not pass")
    if not {"padim", "patchcore"}.issubset(
        set(catalog.get("models_by_category", {}).get("bottle", []))
    ):
        raise RuntimeError("Both bottle model families must be locally configured")
    for route, marker in (
        ("/", b"IhsanVision"), ("/batch", b"Inspect a product batch"),
        ("/history", b"Private inspection history"),
    ):
        if marker not in _get(origin, route):
            raise RuntimeError(f"Local {route} dashboard is incomplete")
    if require_auth:
        assert_unauthenticated_prediction_denied(origin)
        if not os.getenv("IHSANVISION_CLIENT_API_KEY"):
            raise RuntimeError("Set private IHSANVISION_CLIENT_API_KEY for authenticated acceptance")
    predictions = accept(origin, normal, defective, timeout=timeout)
    operator_result = None
    if check_operator and predictions["status"] == "passed":
        sample = image_sample(normal)
        operator_result = verify_operator_workflow(origin, sample["png_bytes"])
        if operator_result.get("status") != "operator_workflow_passed":
            raise RuntimeError("Real private operator workflow acceptance failed")
    return {
        "schema_version": 1,
        "status": ("local_stack_acceptance_passed"
                   if predictions["status"] == "passed" else "prediction_label_mismatch"),
        "checks": {
            "liveness": "passed",
            "checkpoint_config_readiness": "passed",
            "bottle_model_discovery": "passed",
            "single_dashboard": "passed",
            "batch_dashboard": "passed",
            "private_history_shell": "passed",
            "missing_api_key_refused": "passed" if require_auth else "not_checked",
        },
        "inference": predictions,
        "operator_workflow": operator_result,
        "limitations": [
            "Only manually selected bottle samples; no new defect accuracy measurement.",
            "Readiness verifies config/integrity, not inference; actual predictions are tested separately.",
            ("Browser interaction, post-restart persistence and Docker internals are not "
             "exercised." if check_operator else
             "Browser interaction, DB mutation/retention and Docker internals are not exercised."),
            "This verifies only the specified localhost service; it does not prove cloud deployment.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--defective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument(
        "--operator-workflow", action="store_true",
        help="Opt in to write, review, CSV-export and delete one private test inspection",
    )
    args = parser.parse_args(argv)
    if args.operator_workflow and not args.require_auth:
        parser.error("--operator-workflow requires --require-auth and a private history DB")
    destination = args.output.expanduser()
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        parser.error("Evidence output must be a new absolute private path")
    try:
        report = verify_stack(
            args.url, args.normal, args.defective,
            timeout=args.timeout, require_auth=args.require_auth,
            check_operator=args.operator_workflow,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, error.URLError, KeyError, TypeError) as exc:
        raise SystemExit(f"Local stack acceptance FAILED: {exc}") from None
    print("Saved private local-stack acceptance evidence.")
    print("Result:", report["status"])
    if report["status"] != "local_stack_acceptance_passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
