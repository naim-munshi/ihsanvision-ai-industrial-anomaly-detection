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
from .history import InspectionHistory
from .provenance import CheckpointFingerprinter
from .registry import SUPPORTED_CATEGORIES, model_catalog, public_catalog
from .serving import PersistentModelPredictor
from .predict import predict_image
from .schemas import PredictionResponse, InspectionReviewUpdate
from .security import GlobalSlidingWindow, positive_int_env, require_predict_key

logger = logging.getLogger(__name__)

MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
UI = Path(__file__).resolve().parents[1] / "demo" / "index.html"
BATCH_UI = Path(__file__).resolve().parents[1] / "demo" / "batch.html"
HISTORY_UI = Path(__file__).resolve().parents[1] / "demo" / "history.html"


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
        positive_int_env("IHSANVISION_RATE_LIMIT_PER_MINUTE", 120, 10000)
    )
    history_path = os.getenv("IHSANVISION_HISTORY_DB", "").strip()
    history = None
    if history_path:
        # Persistence is explicit and requires a configured operator API key.
        # An unprotected local development server must never accidentally
        # expose durable inspection records via the history endpoints.
        operator_key = os.getenv("IHSANVISION_API_KEY", "")
        if len(operator_key) < 16:
            raise ValueError(
                "IHSANVISION_HISTORY_DB requires a private IHSANVISION_API_KEY "
                "with at least 16 characters"
            )
        history = InspectionHistory(Path(history_path))
    fingerprinter = CheckpointFingerprinter() if history is not None else None

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

    @app.get("/batch")
    def batch_dashboard():
        """Local operator UI; actual predictions use protected /predict."""
        return FileResponse(BATCH_UI, media_type="text/html")

    @app.get("/history")
    def history_dashboard():
        """Public HTML shell only; saved records require operator authentication."""
        return FileResponse(HISTORY_UI, media_type="text/html")

    @app.get("/health")
    def health():
        catalog = model_catalog(_configured_checkpoints())
        return {
            "status": "ok",
            "model_configured": bool(catalog),
        }

    @app.get("/models")
    def models():
        """Expose available model/category IDs; never expose local checkpoint paths."""
        return public_catalog(
            model_catalog(_configured_checkpoints()),
            preferred_model=os.getenv("IHSANVISION_MODEL", "padim").strip().lower(),
        )

    @app.get("/inspections")
    async def inspections(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=1_000_000),
        lot_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$"),
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        if history is None:
            raise HTTPException(503, "Private inspection history is disabled")
        require_predict_key(api_key)
        return await run_in_threadpool(history.list_records, limit, offset, lot_id)

    @app.get("/inspection-lots")
    async def inspection_lots(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=1_000_000),
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        if history is None:
            raise HTTPException(503, "Private inspection history is disabled")
        require_predict_key(api_key)
        return await run_in_threadpool(history.list_lots, limit, offset)

    @app.get("/inspection-model-versions")
    async def inspection_model_versions(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0, le=1_000_000),
        lot_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$"),
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        """Private model-version observation; never expose checkpoints or paths."""
        if history is None:
            raise HTTPException(503, "Private inspection history is disabled")
        require_predict_key(api_key)
        return await run_in_threadpool(
            history.list_model_versions, limit, offset, lot_id
        )

    @app.delete("/inspections/{inspection_id}")
    async def delete_inspection(
        inspection_id: str,
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        if history is None:
            raise HTTPException(503, "Private inspection history is disabled")
        require_predict_key(api_key)
        if not await run_in_threadpool(history.delete_record, inspection_id):
            raise HTTPException(404, "Inspection record not found")
        return {"deleted": True, "inspection_id": inspection_id}

    @app.patch("/inspections/{inspection_id}/review")
    async def review_inspection(
        inspection_id: str,
        review: InspectionReviewUpdate,
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        if history is None:
            raise HTTPException(503, "Private inspection history is disabled")
        require_predict_key(api_key)
        if not await run_in_threadpool(
            history.update_review, inspection_id, review.verdict
        ):
            raise HTTPException(404, "Inspection record not found")
        return {"updated": True, "inspection_id": inspection_id, "verdict": review.verdict}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        model: str | None = Query(default=None),
        category: str | None = Query(default=None),
        lot_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$"),
        api_key: str | None = Header(default=None, alias="X-IhsanVision-Key"),
    ):
        require_predict_key(api_key)
        if lot_id is not None and history is None:
            raise HTTPException(503, "Inspection lot tracking requires private history")
        if model is not None and model not in MODELS:
            raise HTTPException(400, "Unsupported model. Choose padim or patchcore.")
        if category is not None and category not in SUPPORTED_CATEGORIES:
            raise HTTPException(400, "Unsupported product category")
        catalog = model_catalog(_configured_checkpoints())
        selected_category = category or ("bottle" if "bottle" in catalog else next(iter(catalog), "bottle"))
        configured = catalog.get(selected_category, {})
        model_name = model if model is not None else _default_model(configured)
        if model_name is None or model_name not in configured:
            raise HTTPException(503, "Requested category/model has no trusted checkpoint configured")
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
            return await _predict_admitted(file, checkpoint, model_name, selected_category, lot_id)
        finally:
            admission_semaphore.release()

    async def _predict_admitted(file: UploadFile, checkpoint: Path, model_name: str,
                                category: str, lot_id: str | None):
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
                # When history is enabled, bind this prediction to the exact
                # selected checkpoint contents. Only hash a previously unseen
                # on-disk snapshot; inspect in a worker to keep /health responsive.
                identity = (
                    await run_in_threadpool(fingerprinter.identify, checkpoint)
                    if fingerprinter is not None else None
                )
                result = await run_in_threadpool(
                    _run_prediction, payload, suffix, checkpoint, model_name, predict_fn
                )
                if identity is not None:
                    await run_in_threadpool(
                        fingerprinter.verify_unchanged, checkpoint, identity
                    )
                # The category is selected exclusively through the server's
                # trusted registry, never inferred from an arbitrary upload name.
                validated = PredictionResponse.model_validate({
                    **result, "category": category
                })
                if validated.model != model_name:
                    raise ValueError("Inference result model does not match selected checkpoint")
                if history is None:
                    return validated
                validated = validated.model_copy(
                    update={"checkpoint_sha256": identity.digest}
                )
                inspection_id = await run_in_threadpool(
                    history.record, category, model_name, validated.model_dump(), lot_id
                )
                return validated.model_copy(update={"inspection_id": inspection_id})
            except Exception:
                logger.exception("Prediction failed")
                # Do not expose internal exception messages/checkpoint paths to clients.
                raise HTTPException(500, "Inference failed; inspect server logs") from None

    return app


app = create_app()
