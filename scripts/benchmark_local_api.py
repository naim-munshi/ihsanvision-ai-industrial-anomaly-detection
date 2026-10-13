"""Measure repeated inference through the already-running localhost FastAPI service.

For real Mac validation, start Uvicorn in another terminal, then run:
    python -m scripts.benchmark_local_api --url http://127.0.0.1:8002 --runs 10

The default input is a generated synthetic gray PNG. This measures only the
reported per-request model inference time; it is NOT an accuracy benchmark,
full HTTP round-trip benchmark, throughput test, or multi-user load test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from .local_api_smoke import make_png, smoke


def nearest_rank(values: list[float], percentile: float) -> float:
    """Report the specified empirical nearest-rank percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile of empty observations")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile / 100) - 1)
    return round(ordered[index], 2)


def summarize(results: list[dict]) -> dict:
    if len(results) < 2:
        raise ValueError("At least two successful requests are required")
    first = results[0]
    scores = []
    for index, result in enumerate(results, 1):
        if result.get("status") != "api-smoke-passed":
            raise ValueError(f"Request {index} did not pass integration checks")
        if result.get("model") != first.get("model"):
            raise ValueError("Model identity changed during benchmark")
        if result.get("predicted_anomaly") != first.get("predicted_anomaly"):
            raise ValueError("Predicted label changed for identical input")
        if result.get("overlay_png") != first.get("overlay_png"):
            raise ValueError("Overlay availability changed for identical input")
        score = result.get("anomaly_score")
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError(f"Invalid anomaly score for request {index}")
        scores.append(float(score))
        if not math.isclose(scores[0], float(score), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Raw model scores differ for identical input")

    samples = []
    warm = []
    for index, result in enumerate(results, 1):
        elapsed = result.get("latency_ms_including_load")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Invalid inference latency for request {index}")
        samples.append(round(float(elapsed), 2))
        if result.get("model_reused") is True:
            warm.append(float(elapsed))
    return {
        "status": "api-benchmark-passed",
        "model": first["model"],
        "requests": len(results),
        "inference_times_ms": samples,
        "first_request_ms": samples[0],
        "first_request_model_reused": first.get("model_reused"),
        "warm_reused_samples": len(warm),
        "warm_inference_p50_ms": round(median(warm), 2) if warm else None,
        "warm_inference_p95_ms": nearest_rank(warm, 95) if warm else None,
        "consistent_anomaly_score": scores[0],
        "predicted_anomaly": first.get("predicted_anomaly"),
        "overlay_png": first.get("overlay_png"),
        "notes": (
            "Model-reported inference times only. Synthetic input has no defect ground truth. "
            "Warm samples are requests with model_reused=true; none are inferred from position."
        ),
    }


def run(base_url: str, payload: bytes, runs: int = 10, timeout: float = 180) -> dict:
    if runs < 2 or runs > 100:
        raise ValueError("runs must be between 2 and 100")
    results = []
    for i in range(runs):
        result = smoke(base_url, payload, timeout=timeout)
        results.append(result)
        print(f"Request {i + 1}/{runs}: {result['latency_ms_including_load']} ms; "
              f"reused={result.get('model_reused')}", flush=True)
    return summarize(results)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, help="Optional local JSON report (not committed)")
    args = parser.parse_args(argv)
    try:
        result = run(args.url, make_png(args.image), runs=args.runs, timeout=args.timeout)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Benchmark failed: {exc}") from None
    formatted = json.dumps(result, indent=2)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
