"""FastAPI inference service (configure a trusted local checkpoint)."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import MODELS
from .predict import predict_image

MAX_UPLOAD = 10 * 1024 * 1024
UI = Path(__file__).resolve().parents[1] / "demo" / "index.html"


def create_app(predict_fn=None) -> FastAPI:
    app = FastAPI(title="IhsanVision AI", version="0.2.0",
                  description="Defect detection and localization. Do not use for safety-critical decisions.")
    predict_fn = predict_fn or predict_image

    @app.get("/")
    def home():
        return FileResponse(UI, media_type="text/html")

    @app.get("/health")
    def health():
        checkpoint = os.getenv("IHSANVISION_CHECKPOINT", "")
        return {"status": "ok",
                "model_configured": bool(checkpoint and Path(checkpoint).is_file())}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        checkpoint = os.getenv("IHSANVISION_CHECKPOINT")
        model_name = os.getenv("IHSANVISION_MODEL", "padim").lower()
        if not checkpoint or not Path(checkpoint).is_file() or model_name not in MODELS:
            raise HTTPException(503, "Set IHSANVISION_CHECKPOINT to a trusted local .ckpt and IHSANVISION_MODEL")
        payload = await file.read(MAX_UPLOAD + 1)
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Maximum upload size is 10 MiB")
        if not payload:
            raise HTTPException(400, "Empty upload")
        # Validate file content, not user-supplied name or content-type.
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(BytesIO(payload)) as img:
                img.verify()
            with Image.open(BytesIO(payload)) as img:
                if img.width * img.height > 25_000_000:
                    raise HTTPException(413, "Image has too many pixels")
                if img.format not in ("PNG", "JPEG", "WEBP", "BMP"):
                    raise HTTPException(415, "Supported formats: PNG, JPEG, WEBP, BMP")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(415, "Invalid image content") from None
        with tempfile.TemporaryDirectory(prefix="ihsanvision-") as tempdir:
            image_path = Path(tempdir) / "image.png"
            image_path.write_bytes(payload)
            try:
                return predict_fn(image_path, Path(checkpoint), model_name)
            except Exception:
                # Never return checkpoint paths or Python traceback to clients.
                raise HTTPException(500, "Inference failed; inspect server logs") from None

    return app


app = create_app()
