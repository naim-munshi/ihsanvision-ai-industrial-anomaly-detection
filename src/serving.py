"""Long-lived, per-process Anomalib inference session for the local FastAPI service.

The first request loads a *trusted* Lightning checkpoint through Anomalib's
Engine.predict; subsequent requests reuse the same model/Trainer in memory.
The API serializes calls to this instance with an asyncio semaphore.

This path needs real-checkpoint integration validation on macOS. Single-image
CLI inference in src.predict remains independent and unchanged.
"""
from __future__ import annotations

import base64
from pathlib import Path
import time

from PIL import Image

from .config import MODELS, make_model
from .predict import _first_scalar, _overlay


class PersistentModelPredictor:
    """Cache a trained inference session until its model or checkpoint changes.

    Caller must serialize access to this object; Anomalib Engine/Trainer is
    intentionally not assumed to be thread-safe. Checkpoint files must never
    come from untrusted user uploads.
    """

    def __init__(self, accelerator: str = "cpu") -> None:
        self.accelerator = accelerator
        self._key: tuple | None = None
        self._engine = None
        self._model = None
        self._ready = False

    def __call__(self, image_path: Path, checkpoint: Path, model_name: str) -> dict:
        image_path, checkpoint = Path(image_path), Path(checkpoint)
        if not image_path.is_file():
            raise FileNotFoundError("Missing inference image")
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".ckpt":
            raise FileNotFoundError("Expected a trusted local .ckpt checkpoint")
        if model_name not in MODELS:
            raise ValueError("Unsupported anomaly model")

        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")

        stat = checkpoint.stat()
        key = (str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns, model_name, self.accelerator)
        if key != self._key:
            # Lazy model creation; do not load pretrained weights during /health.
            from anomalib.engine import Engine

            self._engine = Engine(accelerator=self.accelerator, devices=1, logger=False)
            self._model = make_model(model_name, pre_trained=False)
            self._key = key
            self._ready = False

        # Pass the checkpoint exactly once for this cache key. Lightning restores
        # its state into the model on that first call; later predict() calls use
        # the already-loaded model. If prediction fails, retry checkpoint loading.
        checkpoint_to_load = None if self._ready else str(checkpoint)
        start = time.perf_counter()
        batches = self._engine.predict(
            model=self._model,
            data_path=image_path,
            ckpt_path=checkpoint_to_load,
            return_predictions=True,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if not batches:
            raise RuntimeError("Anomalib returned no predictions")
        pred = batches[0]
        score = _first_scalar(getattr(pred, "pred_score", None))
        label_num = _first_scalar(getattr(pred, "pred_label", None))
        if score is None:
            raise RuntimeError("Prediction contained no anomaly score")

        heatmap = _overlay(rgb, getattr(pred, "anomaly_map", None))
        self._ready = True
        return {
            "model": model_name,
            "anomaly_score": score,
            "predicted_anomaly": None if label_num is None else bool(label_num),
            "latency_ms": latency_ms,
            "heatmap_png_base64": heatmap,
            "notes": "Score is NOT a probability; heatmap is relative visualization.",
            "model_reused": checkpoint_to_load is None,
        }
