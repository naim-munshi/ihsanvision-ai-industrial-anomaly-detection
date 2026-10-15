"""Controlled local, single-process-target concurrent HTTP load probe.

Sends a finite set of simultaneous inference requests to a localhost API.
Measures accepted/rejected responses instead of silently reporting only fast
successful predictions. No training, remote endpoints or checkpoint loading.

Example:
 python -m scripts.benchmark_concurrent \
   --url http://127.0.0.1:8003 --model patchcore \
   --concurrency 2 --requests 20 --warmup 2 \
   --output "$HOME/ihsanvision-artifacts/patchcore_concurrent.json"
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError

from .benchmark_serving import MODELS, measure_once, sample_png, summary
from .dual_model_acceptance import discover_models, local_origin


def attempt(origin: str, model: str, payload: bytes, timeout: float) -> dict:
    t0 = perf_counter()
    try:
        sample = measure_once(origin, model, payload, timeout)
        return {"status": "200", **sample}
    except HTTPError as exc:
        return {"status": str(exc.code), "round_trip_ms": round((perf_counter() - t0) * 1000, 3)}
    except (URLError, OSError, RuntimeError, ValueError):
        return {"status": "transport_or_response_error", "round_trip_ms": round((perf_counter() - t0) * 1000, 3)}


def run_load(
    url: str,
    model: str,
    image: Path | None = None,
    concurrency: int = 2,
    requests: int = 20,
    warmup: int = 2,
    timeout: float = 180.0,
) -> dict:
    if model not in MODELS:
        raise ValueError("Unsupported model")
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError("Concurrency must be 1–16")
    if type(requests) is not int or not concurrency <= requests <= 500:
        raise ValueError("Requests must be between concurrency and 500")
    if type(warmup) is not int or not 0 <= warmup <= 30:
        raise ValueError("Warmup must be 0–30")
    if not 1 <= timeout <= 600:
        raise ValueError("Timeout must be 1–600 seconds")
    origin = local_origin(url)
    if model not in discover_models(origin):
        raise RuntimeError(f"Model not configured: {model}")
    payload = sample_png(image)
    warmup_results = [attempt(origin, model, payload, timeout) for _ in range(warmup)]
    if any(row["status"] != "200" for row in warmup_results):
        raise RuntimeError("Warmup failed; refusing a misleading load profile")

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(attempt, origin, model, payload, timeout) for _ in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    total_seconds = perf_counter() - started
    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    successful = [row for row in results if row["status"] == "200"]
    failed = requests - len(successful)
    return {
        "schema_version": 1,
        "status": "all_requests_succeeded" if not failed else "completed_with_rejections_or_errors",
        "model": model,
        "test_type": "finite_single_client_local_concurrent_probe",
        "sample_type": "synthetic" if image is None else "user_supplied",
        "sample_sha256_of_transmitted_png": sha256(payload).hexdigest(),
        "concurrency": concurrency,
        "requests_submitted": requests,
        "warmup_successful": len(warmup_results),
        "results_successful": len(successful),
        "results_failed_or_rejected": failed,
        "response_status_counts": dict(sorted(status_counts.items())),
        "measured_seconds": round(total_seconds, 3),
        "observed_successful_requests_per_second": (
            round(len(successful) / total_seconds, 3) if total_seconds > 0 else None
        ),
        "successful_client_round_trip": (
            summary([row["round_trip_ms"] for row in successful]) if successful else None
        ),
        "successful_server_prediction": (
            summary([row["model_predict_ms"] for row in successful]) if successful else None
        ),
        "successful_cache_reuse_counts": {
            "true": sum(row["model_reused"] is True for row in successful),
            "false": sum(row["model_reused"] is False for row in successful),
            "unknown": sum(row["model_reused"] is None for row in successful),
        },
        "limitations": [
            "One machine/client and small finite workload; not a distributed or sustained capacity test.",
            "Server currently serializes one Anomalib inference per process; concurrency adds queue contention.",
            "Successful-only latency statistics exclude rejected/failed requests; total status counts must be considered.",
            "A 429 indicates rate-limit admission; a 503 may indicate admission/dependency failure.",
            "Observed completed requests per second is for this single run, not a scalable QPS guarantee.",
            "No production SLO, accuracy, memory limit, safety or cold-start claim is implied.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, help="New JSON file; never overwrites")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        result = run_load(
            args.url, args.model, image=args.image,
            concurrency=args.concurrency, requests=args.requests,
            warmup=args.warmup, timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Concurrent load check failed: {exc}") from None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2, allow_nan=False)
            writer.write("\n")
        print(f"Saved load evidence: {args.output}")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["results_failed_or_rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
