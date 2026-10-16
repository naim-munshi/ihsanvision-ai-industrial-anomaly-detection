"""Final local portfolio-MVP acceptance: browser, API errors, and both real models.

Run against a localhost server already started by scripts.run_local_demo.
Uses two personally chosen local images; never trains, loads checkpoints in the
client, sends to a remote host, or publishes source images/overlays.

    python -m scripts.final_local_acceptance \
      --normal "$HOME/Downloads/bottle-ui-test/normal_bottle.png" \
      --defective "$HOME/Downloads/bottle-ui-test/defective_bottle.png" \
      --output "$HOME/ihsanvision-artifacts/final_local_acceptance.json"

A pass is a *local demonstration acceptance*, not release approval, dataset
provenance verification, a production security audit, or a model accuracy study.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib import error, request

from .dual_model_acceptance import accept, discover_models, local_origin
from .local_api_smoke import multipart_png


def _api_headers(content_type: str) -> dict[str, str]:
    headers = {"Content-Type": content_type}
    key = os.getenv("IHSANVISION_CLIENT_API_KEY")
    if key:
        headers["X-IhsanVision-Key"] = key
    return headers


def _expect_http_error(origin: str, label: str, endpoint: str,
                       payload: bytes, content_type: str, expected: int) -> dict:
    req = request.Request(
        origin + endpoint, data=payload, method="POST",
        headers=_api_headers(content_type),
    )
    try:
        with request.urlopen(req, timeout=15) as response:
            status = response.status
    except error.HTTPError as exc:
        status = exc.code
        exc.close()
    if status != expected:
        if status == 401:
            raise RuntimeError(
                f"{label}: local inference API requires authentication (HTTP 401). "
                "Set IHSANVISION_CLIENT_API_KEY to the private key configured on "
                "the local server; do not put the key in commands or reports."
            )
        raise RuntimeError(
            f"{label}: expected HTTP {expected}, received {status}"
        )
    return {"case": label, "expected_status": expected, "actual_status": status}


def check_service_and_errors(origin: str) -> dict:
    """Read-only service checks plus three small deliberately rejected uploads."""
    origin = local_origin(origin)
    with request.urlopen(origin + "/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Health endpoint is not HTTP 200")
        health = json.load(response)
    if not isinstance(health, dict) or health.get("status") != "ok" or health.get("model_configured") is not True:
        raise RuntimeError("Health endpoint is not configured and ready")

    models = discover_models(origin)
    if set(models) != {"padim", "patchcore"}:
        raise RuntimeError("Both model choices must be registered")

    with request.urlopen(origin + "/", timeout=10) as response:
        html = response.read(512 * 1024)
        if response.status != 200 or b"IhsanVision" not in html:
            raise RuntimeError("Browser dashboard is missing or malformed")

    valid_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
        b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    # No model inference is reached in any of the negative cases.
    valid_body, valid_type = multipart_png(valid_png)
    invalid_body, invalid_type = multipart_png(b"not an image")
    checks = [
        _expect_http_error(
            origin, "unsupported_model", "/predict?model=unknown",
            valid_body, valid_type, 400,
        ),
        _expect_http_error(
            origin, "invalid_image", "/predict?model=padim",
            invalid_body, invalid_type, 415,
        ),
        _expect_http_error(
            origin, "missing_upload", "/predict?model=padim",
            b"", "application/x-www-form-urlencoded", 422,
        ),
    ]
    return {
        "health": "ok",
        "dashboard": "served",
        "available_models": list(models),
        "rejected_request_checks": checks,
    }


def run_acceptance(origin: str, normal: Path, defective: Path, timeout: float = 180) -> dict:
    origin = local_origin(origin)
    service = check_service_and_errors(origin)
    predictions = accept(origin, normal, defective, timeout=timeout)
    passed = predictions["status"] == "passed"
    return {
        "schema_version": 1,
        "status": "local_mvp_acceptance_passed" if passed else "local_mvp_acceptance_failed",
        "scope": "localhost_portfolio_demo_only",
        "service_and_error_states": service,
        "dual_model_real_image_acceptance": predictions,
        "release_authorized": False,
        "limitations": [
            "Four user-selected images/model combinations are not a held-out accuracy study.",
            "Sample split membership and scientific provenance are not independently verified.",
            "A local pass does not verify Docker, public security, licensing, or deployment.",
            "The browser endpoint was checked over HTTP, not with a visual or accessibility audit.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--defective", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, help="New private JSON path (never overwritten)")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior evidence")
    try:
        report = run_acceptance(args.url, args.normal, args.defective, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, error.URLError) as exc:
        raise SystemExit(f"Final local acceptance failed: {exc}") from None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved local acceptance evidence: {args.output}")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "local_mvp_acceptance_passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
