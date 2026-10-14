"""FastAPI inference service (configure a trusted local checkpoint)."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .config import MODELS
from .serving import PersistentModelPredictor
from .predict import predict_image
from .schemas import PredictionResponse

logger = logging.getLogger(__name__)

MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
UI = Path(__file__).resolve().parents[1] / "demo" / "index.html"


IMAGE_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp"}


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

    @app.get("/")
    def home():
        return FileResponse(UI, media_type="text/html")

    @app.get("/health")
    def health():
        checkpoint = os.getenv("IHSANVISION_CHECKPOINT", "")
        return {
            "status": "ok",
            "model_configured": bool(checkpoint and Path(checkpoint).is_file()),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        checkpoint = os.getenv("IHSANVISION_CHECKPOINT")
        model_name = os.getenv("IHSANVISION_MODEL", "padim").lower()
        if not checkpoint or not Path(checkpoint).is_file() or model_name not in MODELS:
            raise HTTPException(
                503, "Set IHSANVISION_CHECKPOINT to a trusted local .ckpt and IHSANVISION_MODEL"
            )

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
                    _run_prediction, payload, suffix, Path(checkpoint), model_name, predict_fn
                )
            except Exception:
                logger.exception("Prediction failed")
                # Do not expose internal exception messages/checkpoint paths to clients.
                raise HTTPException(500, "Inference failed; inspect server logs") from None

    return app


app = create_app()
