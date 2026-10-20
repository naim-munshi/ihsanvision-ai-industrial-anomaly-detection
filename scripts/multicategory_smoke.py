"""Authenticated synthetic-PNG localhost smoke test for five MVTec categories.

No external photos, training, or dataset needed. Tests the real configured
checkpoint, finite-score response, routing, and PNG heatmap for each pair.
If private history is on, deletes ONLY the record produced by that request.
This is NOT a model-accuracy, defect-label, or manual-browser test.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from urllib import error, parse, request

from PIL import Image

from .dual_model_acceptance import local_origin
from .local_api_smoke import make_png, multipart_png

CATEGORIES = ("bottle", "cable", "carpet", "metal_nut", "capsule")
MODELS = ("padim", "patchcore")
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
ID_PATTERN = re.compile(r"[0-9a-f]{32}\\Z")


def request_json(url: str, *, body: bytes | None = None,
                 headers: dict[str, str] | None = None,
                 method: str = "GET", timeout: float = 30) -> dict:
    req = request.Request(url, data=body, headers=headers or {}, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Local API JSON response exceeds limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Expected local API JSON object")
    return value


def validate_response(data: dict, *, category: str, model: str,
                      dimensions: tuple[int, int]) -> dict:
    if data.get("category") != category or data.get("model") != model:
        raise ValueError("Category or model routing mismatch")
    score = data.get("anomaly_score")
    if type(score) not in (float, int) or not math.isfinite(score):
        raise ValueError("Expected finite anomaly score")
    label = data.get("predicted_anomaly")
    if label is not None and type(label) is not bool:
        raise ValueError("Invalid prediction label type")
    encoded = data.get("heatmap_png_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Missing generated heatmap")
    try:
        overlay = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(overlay)) as im:
            if im.format != "PNG" or im.size != dimensions:
                raise ValueError("Heatmap format or dimensions mismatch")
            im.verify()
    except (OSError, ValueError, binascii.Error) as exc:
        raise ValueError("Invalid heatmap PNG") from exc
    reused = data.get("model_reused")
    if reused is not None and type(reused) is not bool:
        raise ValueError("Invalid model cache flag")
    latency = data.get("latency_ms")
    if latency is not None and (type(latency) not in (float, int)
                                or not math.isfinite(latency) or latency < 0):
        raise ValueError("Invalid model latency")
    return {"category": category, "model": model, "overlay_png_valid": True,
            "label_present": type(label) is bool,
            "model_reused": reused, "latency_ms": latency}


def verify_catalog(origin: str, categories: list[str]) -> dict:
    catalog = request_json(origin + "/models")
    ready = request_json(origin + "/ready", timeout=180)
    if (ready.get("status") != "config_ready"
            or ready.get("approved_sha256_verified") is not True
            or ready.get("model_pairs", 0) < 2 * len(categories)):
        raise RuntimeError("Server readiness/approved checkpoint integrity failed")
    available = catalog.get("models_by_category")
    if not isinstance(available, dict):
        raise RuntimeError("Missing model category catalog")
    for category in categories:
        if set(available.get(category, [])) != set(MODELS):
            raise RuntimeError(f"Both trusted checkpoints unavailable for {category}")
    return ready


def cleanup_inspection(origin: str, key: str, inspection_id: str) -> None:
    if not ID_PATTERN.fullmatch(inspection_id):
        raise ValueError("Unexpected private inspection identifier format")
    data = request_json(
        f"{origin}/inspections/{inspection_id}",
        headers={"X-IhsanVision-Key": key}, method="DELETE", timeout=30)
    if data.get("deleted") is not True or data.get("inspection_id") != inspection_id:
        raise RuntimeError("Synthetic inspection cleanup did not confirm deletion")


def verify_pairs(origin: str, categories: list[str], key: str,
                 *, timeout: float = 300) -> list[dict]:
    if not key or any(ch in key for ch in "\\r\\n") or len(key) < 16:
        raise ValueError("Set a valid private operator key (never paste it in chat)")
    if not 1 <= timeout <= 600:
        raise ValueError("Timeout must be between 1 and 600 seconds")
    origin = local_origin(origin)
    verify_catalog(origin, categories)
    png = make_png()
    with Image.open(BytesIO(png)) as im:
        dimensions = im.size
    outcomes = []
    for category in categories:
        for model in MODELS:
            body, content_type = multipart_png(png)
            query = parse.urlencode({"category": category, "model": model})
            result = request_json(
                origin + "/predict?" + query, body=body,
                headers={"Content-Type": content_type, "X-IhsanVision-Key": key},
                method="POST", timeout=timeout)
            # Private history opt-in writes rows; delete ONLY this test's own row.
            inspection_id = result.get("inspection_id")
            try:
                evidence = validate_response(result, category=category,
                                             model=model, dimensions=dimensions)
            finally:
                if inspection_id is not None:
                    if not isinstance(inspection_id, str):
                        raise ValueError("Invalid private history record identifier")
                    cleanup_inspection(origin, key, inspection_id)
            outcomes.append(evidence)
            print(f"PASSED: {category}/{model} (PNG overlay validated)", flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES,
                        default=list(CATEGORIES))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--key-file", type=Path,
                        help="Private operator-key file; otherwise use IHSANVISION_CLIENT_API_KEY")
    args = parser.parse_args(argv)
    if len(args.categories) != len(set(args.categories)):
        parser.error("Duplicate categories not allowed")
    if args.key_file is not None:
        key_path = args.key_file.expanduser()
        if not key_path.is_file() or key_path.is_symlink():
            parser.error("Key file must be a regular non-symlink file")
        if key_path.stat().st_mode & 0o077:
            parser.error("Key file must not be accessible by group/others")
        key = key_path.read_text(encoding="utf-8").strip()
    else:
        key = os.getenv("IHSANVISION_CLIENT_API_KEY", "")
    try:
        outcomes = verify_pairs(args.url, args.categories, key, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, error.URLError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Synthetic localhost acceptance FAILED: {exc}\\n")
    print(json.dumps({"status": "synthetic_inference_passed",
                      "tested_pairs": len(outcomes),
                      "rows": outcomes,
                      "limitation": "Synthetic connectivity/PNG test only, NOT defect accuracy or manual browser acceptance."}, indent=2))


if __name__ == "__main__":
    main()
