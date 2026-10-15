"""Profile the localhost checkpoint-backed HTTP inference service.

This is a sequential, single-client benchmark, NOT a concurrent load test.
Warmup requests are discarded from measured distributions. Record both
client-observed round-trip and API-reported prediction durations. Runs only
against localhost and never saves checkpoint/image bytes or base64 heatmaps.

Example:
 python -m scripts.benchmark_serving --url http://127.0.0.1:8003 \
   --model patchcore --requests 20 --warmup 3 \
   --output "$HOME/ihsanvision-artifacts/patchcore_serving_profile.json"
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from urllib import request, parse

from PIL import Image

from .dual_model_acceptance import local_origin, discover_models
from .local_api_smoke import make_png, multipart_png

MODELS = ("padim", "patchcore")


def nearest_rank(values: list[float], percentile: float) -> float:
    """Report a deterministic nearest-rank quantile; do not interpolate tiny samples."""
    if not values or not 0 < percentile <= 100:
        raise ValueError("Need observations and valid percentile")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile / 100) - 1]


def summary(values: list[float]) -> dict:
    if not values or not all(isinstance(n, (float, int)) and math.isfinite(n) and n >= 0 for n in values):
        raise ValueError("Invalid timing distribution")
    return {
        "samples": len(values),
        "min_ms": round(min(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p50_nearest_rank_ms": round(nearest_rank(values, 50), 3),
        "p95_nearest_rank_ms": round(nearest_rank(values, 95), 3),
        "max_ms": round(max(values), 3),
    }


def sample_png(image: Path | None) -> bytes:
    if image is not None:
        image = image.expanduser()
        if not image.is_file():
            raise FileNotFoundError(f"Missing image: {image}")
    payload = make_png(image)
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError("Image exceeds 10 MiB after PNG transcoding")
    return payload


def measure_once(origin: str, model: str, payload: bytes, timeout: float) -> dict:
    body, content_type = multipart_png(payload)
    call = request.Request(
        origin + "/predict?model=" + parse.quote(model),
        data=body,
        headers={
            "Content-Type": content_type,
            **({"X-IhsanVision-Key": os.environ["IHSANVISION_CLIENT_API_KEY"]}
               if os.getenv("IHSANVISION_CLIENT_API_KEY") else {}),
        },
        method="POST",
    )
    t0 = perf_counter()
    with request.urlopen(call, timeout=timeout) as response:
        answer = json.load(response)
    round_trip_ms = (perf_counter() - t0) * 1000
    if not isinstance(answer, dict) or answer.get("model") != model:
        raise RuntimeError("Model mismatch in API response")
    score = answer.get("anomaly_score")
    if type(score) not in (float, int) or not math.isfinite(score):
        raise RuntimeError("Invalid anomaly score in API response")
    duration = answer.get("latency_ms")
    if type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
        raise RuntimeError("API did not return finite prediction duration")
    reused = answer.get("model_reused")
    if reused is not None and type(reused) is not bool:
        raise RuntimeError("Invalid model cache status")
    return {
        "round_trip_ms": round_trip_ms,
        "model_predict_ms": float(duration),
        "model_reused": reused,
    }


def profile(
    url: str,
    model: str,
    image: Path | None = None,
    warmup: int = 3,
    requests: int = 20,
    timeout: float = 180.0,
) -> dict:
    if model not in MODELS:
        raise ValueError("Unsupported model")
    if type(warmup) is not int or not 0 <= warmup <= 50:
        raise ValueError("Warmup must be from 0 to 50")
    if type(requests) is not int or not 2 <= requests <= 500:
        raise ValueError("Use 2–500 measured requests")
    if not 1 <= timeout <= 600:
        raise ValueError("Timeout must be 1–600 seconds")
    origin = local_origin(url)
    available = discover_models(origin)
    if model not in available:
        raise RuntimeError(f"Model not configured on localhost: {model}")

    png_bytes = sample_png(image)
    warmup_rows = []
    for _ in range(warmup):
        warmup_rows.append(measure_once(origin, model, png_bytes, timeout))
    rows = []
    started = perf_counter()
    for _ in range(requests):
        rows.append(measure_once(origin, model, png_bytes, timeout))
    total_s = perf_counter() - started
    client = [row["round_trip_ms"] for row in rows]
    model_span = [row["model_predict_ms"] for row in rows]
    reused = sum(row["model_reused"] is True for row in rows)
    uncached = sum(row["model_reused"] is False for row in rows)
    unknown = requests - reused - uncached
    return {
        "schema_version": 1,
        "status": "sequential_profile_complete",
        "model": model,
        "sample_type": "synthetic" if image is None else "user_supplied",
        "sample_sha256_of_transmitted_png": sha256(png_bytes).hexdigest(),
        "input_image_dimensions": list(Image.open(BytesIO(png_bytes)).size),
        "system": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "warmup_requests": warmup,
        "measured_requests": requests,
        "total_measured_seconds": round(total_s, 3),
        "sequential_completed_requests_per_second": round(requests / total_s, 3) if total_s > 0 else None,
        "client_round_trip": summary(client),
        "server_reported_prediction": summary(model_span),
        "model_cache_reuse_counts": {"true": reused, "false": uncached, "unknown": unknown},
        "warmup_cache_reuse_counts": {
            "true": sum(row["model_reused"] is True for row in warmup_rows),
            "false": sum(row["model_reused"] is False for row in warmup_rows),
            "unknown": sum(row["model_reused"] is None for row in warmup_rows),
        },
        "limitations": [
            "Single-client sequential loop; completed requests per second is not concurrent load throughput.",
            "The client timing includes HTTP overhead; the model span may include checkpoint loading.",
            "Warmup results are excluded from the measured distribution, not automatically classified as cold/warm.",
            "Synthetic images only test integration; model accuracy is not measured.",
            "No results are comparable across machines without controlled environment and workload.",
            "Nearest-rank p95 is unstable with very small sample counts.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, help="New JSON report; never overwrite")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        evidence = profile(args.url, args.model, args.image, args.warmup, args.requests, args.timeout)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Serving profile failed: {exc}") from None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(evidence, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"Saved profile: {args.output}")
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
