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

logger = logging.getLogger(__name__)

MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
UI = Path(__file__).resolve().parents[1] / "demo" / "index.html"


def _validate_image(payload: bytes) -> None:
    """Validate real image bytes, format and pixel count before inference."""
    try:
        with Image.open(BytesIO(payload)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(413, "Image has too many pixels")
            if img.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
                raise HTTPException(415, "Supported formats: PNG, JPEG, WEBP, BMP")
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "Invalid image content") from None


def _run_prediction(payload: bytes, checkpoint: Path, model_name: str, predict_fn) -> dict:
    """Run blocking ML work in a worker thread, keeping temporary bytes private."""
    with tempfile.TemporaryDirectory(prefix="ihsanvision-") as directory:
        image_path = Path(directory) / "image.png"
        image_path.write_bytes(payload)
        return predict_fn(image_path, checkpoint, model_name)


def create_app(predict_fn=None) -> FastAPI:
    app = FastAPI(
        title="IhsanVision AI",
        version="0.3.0",
        description="Defect detection and localization. Not for safety-critical decisions.",
    )
    predict_fn = predict_fn if predict_fn is not None else PersistentModelPredictor()

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

    @app.post("/predict")
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
        _validate_image(payload)

        # Image decoding and Anomalib checkpoint inference are synchronous. Avoid
        # blocking FastAPI's event loop, so /health stays responsive during inference.
        async with inference_semaphore:
            try:
                return await run_in_threadpool(
                    _run_prediction, payload, Path(checkpoint), model_name, predict_fn
                )
            except Exception:
                logger.exception("Prediction failed")
                # Do not expose internal exception messages/checkpoint paths to clients.
                raise HTTPException(500, "Inference failed; inspect server logs") from None

    return app


app = create_app()
