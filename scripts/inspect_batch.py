"""Inspect a bounded local image batch and export private JSON/CSV inspection results.

Usage:
 python -m scripts.inspect_batch --directory /path/to/product-images \
   --category bottle --model patchcore \
   --output-prefix "$HOME/ihsanvision-artifacts/my-inspection"

This calls a localhost IhsanVision API with existing trusted model weights.
It never trains a model or stores source images, model weights or heatmaps.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from urllib import error, parse, request

from PIL import Image

from ihsanvision.registry import SUPPORTED_CATEGORIES
from .dual_model_acceptance import local_origin, validate_prediction
from .local_api_smoke import make_png, multipart_png

MODELS = ("padim", "patchcore")
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
COLUMNS = ("filename", "sha256", "status", "category", "model",
           "predicted_anomaly", "anomaly_score_raw", "latency_ms",
           "model_reused", "overlay_png_valid", "error_code")


def discover_pair(origin: str, category: str, model: str) -> None:
    with request.urlopen(origin + "/models", timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid model discovery response")
    category_models = payload.get("models_by_category")
    if isinstance(category_models, dict):
        available = category_models.get(category, [])
    elif category == "bottle":
        available = payload.get("available_models", [])
    else:
        available = []
    if not isinstance(available, list) or model not in available:
        raise RuntimeError(f"No configured checkpoint for {category}/{model}")


def image_paths(directory: Path, maximum: int) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Input must be a real local directory, not a symlink")
    if type(maximum) is not int or not 1 <= maximum <= 200:
        raise ValueError("Maximum image count must be from 1 to 200")
    images = sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES),
        key=lambda p: p.name,
    )
    if any(p.is_symlink() or not p.is_file() for p in images):
        raise ValueError("Image symlinks and nonregular entries are not supported")
    if not images:
        raise ValueError("No supported image files found in the directory")
    if len(images) > maximum:
        raise ValueError(f"Found {len(images)} images, exceeding --max-images {maximum}")
    return images


def send_prediction(origin: str, category: str, model: str,
                    png: bytes, timeout: float) -> dict:
    body, content_type = multipart_png(png)
    endpoint = (origin + "/predict?model=" + parse.quote(model)
                + ("" if category == "bottle" else "&category=" + parse.quote(category)))
    headers = {"Content-Type": content_type}
    if os.getenv("IHSANVISION_CLIENT_API_KEY"):
        headers["X-IhsanVision-Key"] = os.environ["IHSANVISION_CLIENT_API_KEY"]
    call = request.Request(endpoint, data=body, headers=headers, method="POST")
    with request.urlopen(call, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("model") != model:
        raise RuntimeError("Prediction returned the wrong model")
    if category != "bottle" and result.get("category") != category:
        raise RuntimeError("Prediction returned the wrong category")
    if result.get("category") not in (None, category):
        raise RuntimeError("Prediction returned the wrong category")
    return result


def _sample_png(path: Path) -> tuple[bytes, str, tuple[int, int]]:
    if path.stat().st_size == 0 or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or exceeds the 10 MiB limit")
    with path.open("rb") as reader:
        digest = sha256(reader.read()).hexdigest()
    with Image.open(path) as img:
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions exceed the pixel limit")
        dimensions = img.size
    png = make_png(path)
    if len(png) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds upload limit after PNG conversion")
    return png, digest, dimensions


def inspect(directory: Path, category: str, model: str, url: str,
            max_images: int = 100, timeout: float = 180,
            sender=None) -> dict:
    """Record one structured outcome per image; no unbounded remote queues."""
    if category not in SUPPORTED_CATEGORIES or model not in MODELS:
        raise ValueError("Unsupported category/model")
    if not 1 <= timeout <= 600 or not math.isfinite(timeout):
        raise ValueError("Timeout must be 1–600 seconds")
    origin = local_origin(url)
    samples = image_paths(directory.expanduser(), max_images)
    if sender is None:
        discover_pair(origin, category, model)
        sender = send_prediction

    rows = []
    counts = {"normal": 0, "anomaly": 0, "threshold_unavailable": 0, "errors": 0}
    for path in samples:
        item = {
            "filename": path.name,
            "sha256": None,
            "status": "error",
            "category": category,
            "model": model,
            "predicted_anomaly": None,
            "anomaly_score_raw": None,
            "latency_ms": None,
            "model_reused": None,
            "overlay_png_valid": False,
            "error_code": None,
        }
        try:
            png, item["sha256"], dimensions = _sample_png(path)
            response = sender(origin, category, model, png, timeout)
            evidence = validate_prediction(response, model, dimensions)
            # The strict acceptance validator requires a saved classification
            # threshold; a missing label is an explicit error, not "normal".
            label = evidence["predicted_anomaly"]
            item.update({
                "status": "anomaly" if label else "normal",
                "predicted_anomaly": label,
                "anomaly_score_raw": evidence["anomaly_score_raw"],
                "latency_ms": evidence["latency_ms_including_possible_load"],
                "model_reused": evidence["model_reused"],
                "overlay_png_valid": evidence["overlay_png_valid"],
            })
            counts[item["status"]] += 1
        except error.HTTPError as exc:
            item["error_code"] = f"http_{exc.code}"
            counts["errors"] += 1
        except error.URLError:
            item["error_code"] = "connection_error"
            counts["errors"] += 1
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            # No paths, server stack traces, secrets or raw response bodies
            # are included in the saved inspection report.
            item["error_code"] = "invalid_image_or_response"
            counts["errors"] += 1
        rows.append(item)
    return {
        "schema_version": 1,
        "type": "local_category_batch_inspection",
        "status": "completed" if counts["errors"] == 0 else "completed_with_errors",
        "category": category,
        "model": model,
        "files_attempted": len(samples),
        "counts": counts,
        "results": rows,
        "notes": [
            "Anomaly decisions use the saved checkpoint's threshold; scores are not probabilities.",
            "PNG overlays were validated but not exported or scored against pixel masks.",
            "This report does not establish model accuracy or membership of any evaluation split.",
            "Report is private and may contain sensitive input filenames; do not commit it to Git.",
        ],
    }


def _excel_safe(value: object) -> object:
    """Avoid interpreting user-controlled file names as spreadsheet formulas."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def save_private_report(result: dict, prefix: Path) -> tuple[Path, Path]:
    prefix = prefix.expanduser()
    if prefix.suffix:
        raise ValueError("Provide a filename prefix without an extension")
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    if json_path.exists() or csv_path.exists():
        raise ValueError("A report already exists; refusing to overwrite existing evidence")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_created = False
    try:
        with json_path.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
        json_created = True
        with csv_path.open("x", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in result["results"]:
                writer.writerow({name: _excel_safe(row.get(name)) for name in COLUMNS})
    except Exception:
        # Preserve previously existing artifacts; remove only the newly created
        # JSON when writing the CSV failed.
        if json_created and not csv_path.exists():
            json_path.unlink()
        raise
    return json_path, csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--category", choices=SUPPORTED_CATEGORIES, default="bottle")
    parser.add_argument("--model", choices=MODELS, default="padim")
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_prefix.with_suffix(".json").exists() or args.output_prefix.with_suffix(".csv").exists():
        parser.error("A report already exists; choose a new output prefix")
    try:
        result = inspect(args.directory, args.category, args.model, args.url,
                         args.max_images, args.timeout)
        json_path, csv_path = save_private_report(result, args.output_prefix)
    except (error.URLError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Inspection batch could not finish: {exc}") from None
    print(json.dumps({
        "status": result["status"], "files_attempted": result["files_attempted"],
        "counts": result["counts"], "json": str(json_path), "csv": str(csv_path),
    }, indent=2))
    if result["counts"]["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
