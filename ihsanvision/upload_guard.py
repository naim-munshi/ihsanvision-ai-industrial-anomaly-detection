"""Bound /predict request bodies and authenticate before multipart parsing.

Local defense-in-depth; not a substitute for TLS and gateway body limits.
The guard buffers at most MAX_MULTIPART_BYTES per in-flight request, then
replays the bytes for FastAPI's normal UploadFile validation.
"""
from __future__ import annotations

from fastapi import HTTPException
from starlette.responses import JSONResponse

from .security import require_predict_key

# The image itself is limited to 10 MiB by the existing /predict endpoint.
# Allow modest multipart boundary/headers overhead; reject extra huge fields.
MAX_MULTIPART_BYTES = 10 * 1024 * 1024 + 64 * 1024


class PredictBodyGuard:
    """Pure ASGI pre-parser guard, including requests without Content-Length."""

    def __init__(self, app, max_body_bytes: int = MAX_MULTIPART_BYTES):
        if max_body_bytes < 1:
            raise ValueError("Invalid body limit")
        self.app = app
        self.max_body_bytes = max_body_bytes

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
            require_predict_key(supplied)
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
