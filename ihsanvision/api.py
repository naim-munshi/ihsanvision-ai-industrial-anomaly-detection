"""FastAPI inference service (configure a trusted local checkpoint)."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from uuid import uuid4
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .config import MODELS
from .serving import PersistentModelPredictor
from .predict import predict_image
from .schemas import PredictionResponse
from .security import GlobalSlidingWindow, positive_int_env, require_predict_key

logger = logging.getLogger(__name__)

MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
UI = Path(__file__).resolve().parents[1] / "demo" / "index.html"


IMAGE_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp"}


def _configured_checkpoints() -> dict[str, Path]:
    """Resolve trusted server-side checkpoints; clients never supply paths.

    The original IHSANVISION_CHECKPOINT + IHSANVISION_MODEL configuration
    remains supported. Explicit per-model settings take precedence.
    """
    configured: dict[str, Path] = {}
    legacy_model = os.getenv("IHSANVISION_MODEL", "padim").strip().lower()
    legacy_path = os.getenv("IHSANVISION_CHECKPOINT", "").strip()
    if legacy_model in MODELS and legacy_path:
        candidate = Path(legacy_path).expanduser()
        if candidate.is_file() and candidate.suffix.lower() == ".ckpt":
            configured[legacy_model] = candidate

    for name in MODELS:
        configured_path = os.getenv(f"IHSANVISION_{name.upper()}_CHECKPOINT")
        if configured_path is None:
            continue
        # An explicitly configured path must not silently fall back to the
        # legacy path if it is missing or invalid.
        configured.pop(name, None)
        if configured_path.strip():
            candidate = Path(configured_path.strip()).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".ckpt":
                configured[name] = candidate
    return configured


def _default_model(configured: dict[str, Path]) -> str | None:
    preferred = os.getenv("IHSANVISION_MODEL", "padim").strip().lower()
    if preferred in configured:
        return preferred
    for name in MODELS:
        if name in configured:
            return name
    return None


def _validate_image(payload: bytes) -> str:
    """Inspect actual bytes, bound decoded pixels and return a trusted suffix."""
    try:
        with Image.open(BytesIO(payload)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(413, "Image has too many pixels")
            suffix = IMAGE_SUFFIXES.get(img.format)
            if suffix is None:
                raise HTTPException(415, "Supported formats: PNG, JPEG, WEBP, BMP")
            img.verify()
            return suffix
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "Invalid image content") from None


def _run_prediction(payload: bytes, suffix: str, checkpoint: Path, model_name: str, predict_fn) -> dict:
    """Run blocking ML work in a worker thread, keeping temporary bytes private."""
    with tempfile.TemporaryDirectory(prefix="ihsanvision-") as directory:
        image_path = Path(directory) / f"image{suffix}"
        image_path.write_bytes(payload)
        return predict_fn(image_path, checkpoint, model_name)


def create_app(predict_fn=None) -> FastAPI:
    app = FastAPI(
        title="IhsanVision AI",
        version="0.3.1",
        description="Defect detection and localization. Not for safety-critical decisions.",
    )
    if predict_fn is None:
        # Safe fallback to previously verified cold inference while the new
        # persistent predictor is undergoing real-checkpoint validation.
        use_cache = os.getenv("IHSANVISION_MODEL_CACHE", "1").strip() != "0"
        predict_fn = PersistentModelPredictor() if use_cache else predict_image

    # The default predictor lazily caches one checkpoint per process.
    # Serialize calls because Anomalib Engine/Trainer is not thread-safe.
    inference_semaphore = asyncio.Semaphore(1)
    # A small per-process admission budget prevents an unbounded ML work queue.
    admission_semaphore = asyncio.BoundedSemaphore(
        positive_int_env("IHSANVISION_MAX_PENDING", 2, 100)
    )
    queue_timeout = float(os.getenv("IHSANVISION_QUEUE_TIMEOUT_SECONDS", "1.0"))
    if not 0 < queue_timeout <= 30:
        raise ValueError("IHSANVISION_QUEUE_TIMEOUT_SECONDS must be in (0, 30]")
    limiter = GlobalSlidingWindow(
        positive_int_env("IHSANVISION_RATE_LIMIT_PER_MINUTE", 30, 10000)
    )

    @app.middleware("http")
    async def request_audit(request: Request, call_next):
        request_id = uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # Never log body bytes, full URLs/queries, filenames, auth headers or scores.
        logger.info(
            "request_id=%s method=%s endpoint=%s status=%s duration_ms=%.2f",
            request_id, request.method, request.url.path,
            response.status_code, (time.perf_counter() - started) * 1000
        )
        return response

    @app.get("/")
    def home():
        return FileResponse(UI, media_type="text/html")

    @app.get("/health")
    def health():
        configured = _configured_checkpoints()
        return {
            "status": "ok",
            "model_configured": bool(configured),
        }

    @app.get("/models")
    def models():
        """Expose model availability without disclosing local checkpoint paths."""
        configured = _configured_checkpoints()
        return {
            "available_models": [name for name in MODELS if name in configured],
            "default_model": _default_model(configured),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        model: str | None = Query(default=None),
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        require_predict_key(api_key)
        if model is not None and model not in MODELS:
            raise HTTPException(400, "Unsupported model. Choose padim or patchcore.")
        configured = _configured_checkpoints()
        model_name = model if model is not None else _default_model(configured)
        if model_name is None or model_name not in configured:
            raise HTTPException(503, "Requested model has no trusted checkpoint configured")
        checkpoint = configured[model_name]

        await limiter.require_slot()
        try:
            await asyncio.wait_for(admission_semaphore.acquire(), timeout=queue_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(
                503, "Inference capacity temporarily exhausted",
                headers={"Retry-After": "1"}
            ) from None

        try:
            return await _predict_admitted(file, checkpoint, model_name)
        finally:
            admission_semaphore.release()

    async def _predict_admitted(file: UploadFile, checkpoint: Path, model_name: str):
        payload = await file.read(MAX_UPLOAD + 1)
        if not payload:
            raise HTTPException(400, "Empty upload")
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Maximum upload size is 10 MiB")
        # Decode and verify in a worker thread, not the event loop. The temporary
        # file suffix is derived from inspected image bytes, never client filename.
        suffix = await run_in_threadpool(_validate_image, payload)

        # Image decoding and Anomalib checkpoint inference are synchronous. Avoid
        # blocking FastAPI's event loop, so /health stays responsive during inference.
        async with inference_semaphore:
            try:
                return await run_in_threadpool(
                    _run_prediction, payload, suffix, checkpoint, model_name, predict_fn
                )
            except Exception:
                logger.exception("Prediction failed")
                # Do not expose internal exception messages/checkpoint paths to clients.
                raise HTTPException(500, "Inference failed; inspect server logs") from None

    return app


app = create_app()
