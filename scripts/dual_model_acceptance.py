"""Repeatable, real-checkpoint local API acceptance test for two bottle images.

The command sends each user-supplied image to both PaDiM and PatchCore via the
running localhost API. It checks model routing, response types, image-level
labels, and genuine PNG overlays. It NEVER trains or loads weights itself.

Usage:
    python -m scripts.dual_model_acceptance \
      --url http://127.0.0.1:8003 \
      --normal /path/to/normal_bottle.png \
      --defective /path/to/defective_bottle.png \
      --output /private/path/dual_model_acceptance.json

Inputs are manually selected demonstration examples, not a held-out dataset
accuracy study. File hashes are recorded but saved split membership is NOT
inferred. Local images/checkpoints and base64 overlays are never stored in the
JSON report. Only run against an API you control on localhost.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
import math
from pathlib import Path
import sys
from urllib import error, parse, request

from PIL import Image

from .local_api_smoke import make_png, multipart_png


MODELS = ("padim", "patchcore")
MAX_UPLOAD = 10 * 1024 * 1024


def local_origin(base_url: str) -> str:
    """Refuse remote endpoints and ambiguous userinfo, path or query."""
    url = parse.urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"localhost", "127.0.0.1"}
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("Only localhost HTTP origins are supported")
    try:
        port = url.port
    except ValueError:
        raise ValueError("Invalid local server port") from None
    if port is None:
        raise ValueError("Specify localhost's server port, e.g. :8003")
    return f"http://{url.hostname}:{port}"


def discover_models(origin: str) -> tuple[str, ...]:
    with request.urlopen(origin + "/models", timeout=10) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("available_models"), list):
        raise RuntimeError("Invalid /models JSON")
    models = payload["available_models"]
    if not all(name in MODELS for name in models) or len(set(models)) != len(models):
        raise RuntimeError("Invalid /models list")
    return tuple(models)


def predict_png(origin: str, model: str, png_bytes: bytes, timeout: float) -> dict:
    body, content_type = multipart_png(png_bytes)
    call = request.Request(
        origin + "/predict?model=" + parse.quote(model),
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with request.urlopen(call, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("Expected a prediction JSON object")
    return result


def validate_prediction(result: dict, model: str, expected_size: tuple[int, int]) -> dict:
    """Return only bounded, non-sensitive evidence, not raw images/overlays."""
    if result.get("model") != model:
        raise RuntimeError(f"Model mismatch; requested {model}")
    score = result.get("anomaly_score")
    if type(score) not in (int, float) or not math.isfinite(score):
        raise RuntimeError("Missing or nonfinite anomaly score")
    label = result.get("predicted_anomaly")
    if type(label) is not bool:
        raise RuntimeError("No boolean model prediction")
    latency = result.get("latency_ms")
    if latency is not None and (
        type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0
    ):
        raise RuntimeError("Invalid prediction latency")
    reused = result.get("model_reused")
    if reused is not None and type(reused) is not bool:
        raise RuntimeError("Invalid model cache indicator")
    encoded = result.get("heatmap_png_base64")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("Missing anomaly heatmap")
    try:
        binary = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(binary)) as img:
            if img.format != "PNG" or img.size != expected_size:
                raise RuntimeError("Heatmap must be a PNG with input-image dimensions")
            img.verify()
    except (ValueError, OSError, base64.binascii.Error) as exc:
        raise RuntimeError("Malformed anomaly overlay PNG") from exc
    return {
        "anomaly_score_raw": float(score),
        "predicted_anomaly": label,
        "latency_ms_including_possible_load": latency,
        "model_reused": reused,
        "overlay_png_valid": True,
    }


def image_sample(path: Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample image: {path}")
    raw = path.read_bytes()
    png_bytes = make_png(path)
    if len(png_bytes) > MAX_UPLOAD:
        raise ValueError(f"Transcoded image exceeds the server upload limit: {path.name}")
    with Image.open(BytesIO(png_bytes)) as image:
        size = image.size
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "png_bytes": png_bytes,
        "dimensions": size,
    }


def accept(origin: str, normal_path: Path, defective_path: Path, timeout: float = 180) -> dict:
    """Perform four real localhost HTTP requests; no training or dataset reads."""
    if not 1 <= timeout <= 600:
        raise ValueError("timeout must be between 1 and 600 seconds")
    origin = local_origin(origin)
    available = discover_models(origin)
    if not all(model in available for model in MODELS):
        raise RuntimeError("Both padim and patchcore checkpoints must be configured")

    samples = {
        "normal": image_sample(normal_path),
        "defective": image_sample(defective_path),
    }
    if samples["normal"]["sha256"] == samples["defective"]["sha256"]:
        raise ValueError("Normal and defective inputs have identical bytes")

    rows = []
    for model in MODELS:
        for kind in ("normal", "defective"):
            sample = samples[kind]
            prediction = predict_png(origin, model, sample["png_bytes"], timeout=timeout)
            evidence = validate_prediction(prediction, model, sample["dimensions"])
            expected = kind == "defective"
            rows.append({
                "model": model,
                "sample": kind,
                "predicted_as_expected": evidence["predicted_anomaly"] is expected,
                **evidence,
            })

    passed = all(row["predicted_as_expected"] for row in rows)
    return {
        "status": "passed" if passed else "label_mismatch",
        "test_type": "local_dual_model_manual_sample_acceptance",
        "model_names": list(MODELS),
        "sample_evidence": {
            kind: {
                "filename": sample["filename"],
                "sha256": sample["sha256"],
                "dimensions": list(sample["dimensions"]),
            }
            for kind, sample in samples.items()
        },
        "results": rows,
        "limitations": [
            "Sample file membership of the saved test split is unverified.",
            "Four manually selected predictions do not establish accuracy or recall.",
            "Anomaly scores are uncalibrated and not comparable between models.",
            "Timings include possible checkpoint loading and are not p50/p95 or throughput.",
            "Overlay PNG validity does not prove pixel-level localization quality.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--defective", required=True, type=Path)
    parser.add_argument("--timeout", default=180.0, type=float)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; existing files are not overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists; refusing to overwrite: {args.output}")
    try:
        report = accept(args.url, args.normal, args.defective, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, error.URLError) as exc:
        raise SystemExit(f"Dual-model acceptance could not finish: {exc}") from None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2, allow_nan=False)
            writer.write("\n")
        print(f"Saved acceptance evidence: {args.output}")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
