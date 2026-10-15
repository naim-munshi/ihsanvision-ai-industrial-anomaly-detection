"""Small, bounded per-process admission controls for the local inference service.

These controls protect one Uvicorn process and do NOT replace a reverse proxy,
HTTPS, gateway-level upload limits, centralized identity or distributed limits.
"""
from __future__ import annotations

import asyncio
from collections import deque
import hmac
import os
from time import monotonic

from fastapi import HTTPException


def positive_int_env(name: str, default: int, maximum: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}") from None
    if not 1 <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return parsed


def require_predict_key(supplied: str | None) -> None:
    """Protect POST /predict when a key is explicitly configured."""
    configured = os.getenv("IHSANVISION_API_KEY")
    if configured is None:
        return  # Backwards-compatible localhost developer mode; NOT public-safe.
    if not configured or not hmac.compare_digest(supplied or "", configured):
        raise HTTPException(401, "Valid inference API key required", headers={"WWW-Authenticate": "ApiKey"})


class GlobalSlidingWindow:
    """Memory-bounded global 60-second request limit for a *single* process."""

    def __init__(self, limit: int = 30, interval: float = 60.0):
        if limit < 1 or interval <= 0:
            raise ValueError("Invalid rate limit")
        self.limit = limit
        self.interval = interval
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def require_slot(self) -> None:
        async with self._lock:
            now = monotonic()
            while self._times and now - self._times[0] >= self.interval:
                self._times.popleft()
            if len(self._times) >= self.limit:
                retry = max(1, int(self.interval - (now - self._times[0]) + 0.999))
                raise HTTPException(429, "Inference request limit reached", headers={"Retry-After": str(retry)})
            self._times.append(now)
