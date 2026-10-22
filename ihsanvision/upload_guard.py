"""Bound prediction uploads before multipart parsing and before memory admission.

Per-process pure-ASGI defense in depth. Authenticated POST /predict requests
must acquire finite admission before their bodies are buffered. This bounds
in-flight request payload memory in addition to per-request multipart limits.
Not a distributed quota, TLS proxy, or defense against stalled connections.
"""
from __future__ import annotations

import asyncio
import os

from fastapi import HTTPException
from starlette.responses import JSONResponse

from .security import positive_int_env, require_predict_key

MAX_MULTIPART_BYTES = 10 * 1024 * 1024 + 64 * 1024


class PredictBodyGuard:
    """Authenticate first, acquire admission, stream-bound and replay multipart."""

    def __init__(self, app, max_body_bytes: int = MAX_MULTIPART_BYTES,
                 max_inflight: int | None = None, timeout: float | None = None):
        if max_body_bytes < 1:
            raise ValueError("Invalid body limit")
        if max_inflight is None:
            max_inflight = positive_int_env("IHSANVISION_MAX_PENDING", 2, 100)
        if type(max_inflight) is not int or not 1 <= max_inflight <= 100:
            raise ValueError("Upload admission must be 1-100 requests")
        if timeout is None:
            timeout = float(os.getenv("IHSANVISION_QUEUE_TIMEOUT_SECONDS", "1.0"))
        if not 0 < timeout <= 30:
            raise ValueError("Invalid upload admission timeout")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.timeout = timeout
        self.admission = asyncio.BoundedSemaphore(max_inflight)

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("path") != "/predict"
                or scope.get("method") != "POST"):
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        credentials = [v for k, v in headers if k.lower() == b"x-ihsanvision-key"]
        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        if len(credentials) > 1 or len(lengths) > 1:
            await JSONResponse({"detail": "Duplicate request headers"}, status_code=400)(
                scope, receive, send
            )
            return
        try:
            supplied = credentials[0].decode("utf-8") if credentials else None
            require_predict_key(supplied, scope.get("state", {}).get("principal"))
        except (UnicodeError, HTTPException):
            await JSONResponse(
                {"detail": "Valid inference API key required"},
                status_code=401,
                headers={"WWW-Authenticate": "ApiKey"},
            )(scope, receive, send)
            return

        if lengths:
            try:
                declared = int(lengths[0].decode("ascii"))
            except (UnicodeError, ValueError):
                declared = -1
            if declared < 0:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return
            if declared > self.max_body_bytes:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return

        try:
            await asyncio.wait_for(self.admission.acquire(), timeout=self.timeout)
        except asyncio.TimeoutError:
            await JSONResponse(
                {"detail": "Upload capacity temporarily exhausted"}, status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return

        try:
            await self._accept_bounded(scope, receive, send)
        finally:
            self.admission.release()

    async def _accept_bounded(self, scope, receive, send):
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                await JSONResponse({"detail": "Invalid request stream"}, status_code=400)(
                    scope, receive, send
                )
                return
            part = message.get("body", b"")
            total += len(part)
            if total > self.max_body_bytes:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return
            chunks.append(part)
            if not message.get("more_body", False):
                break

        index = 0

        async def replay():
            nonlocal index
            if index < len(chunks):
                part = chunks[index]
                index += 1
                return {
                    "type": "http.request",
                    "body": part,
                    "more_body": index < len(chunks),
                }
            return await receive()

        await self.app(scope, replay, send)
