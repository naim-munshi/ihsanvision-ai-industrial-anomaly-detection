"""Bounded, metadata-free per-process inference telemetry.

Measures admitted API inference attempts only. No image, filename, score,
checkpoint path, user, lot ID or credential is retained or exported.
"""
from __future__ import annotations

from collections import deque
import math
from threading import Lock
from time import monotonic


class InferenceTelemetry:
    """Small in-memory window; resets on process restart."""

    def __init__(self, max_samples: int = 256) -> None:
        if type(max_samples) is not int or not 1 <= max_samples <= 10_000:
            raise ValueError("Invalid telemetry window")
        self._started = monotonic()
        self._lock = Lock()
        self._success = 0
        self._failed = 0
        self._latencies: deque[float] = deque(maxlen=max_samples)

    def record(self, *, success: bool, duration_ms: float) -> None:
        if type(success) is not bool or not isinstance(duration_ms, (int, float)) or (
            not math.isfinite(duration_ms) or duration_ms < 0
        ):
            raise ValueError("Invalid telemetry measurement")
        with self._lock:
            if success:
                self._success += 1
            else:
                self._failed += 1
            self._latencies.append(float(duration_ms))

    def snapshot(self) -> dict:
        with self._lock:
            ordered = sorted(self._latencies)
            success, failed = self._success, self._failed

        def quantile(fraction: float) -> float | None:
            if not ordered:
                return None
            rank = math.ceil(fraction * len(ordered)) - 1
            return round(ordered[max(0, rank)], 2)

        return {
            "scope": "single_process",
            "source": "admitted_predict_requests_only",
            "inference_successful": success,
            "inference_failed": failed,
            "inference_completed": success + failed,
            "sample_window_count": len(ordered),
            "post_admission_http_p50_ms": quantile(0.5),
            "post_admission_http_p95_ms": quantile(0.95),
            "uptime_seconds": round(max(0.0, monotonic() - self._started), 2),
            "limitations": (
                "Excludes rejected pre-admission requests, body upload and client network; "
                "includes image validation and serialized model inference. Not a full SLO."
            ),
        }
