"""Checkpoint-backed inference. Load only checkpoints you trust."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import MODELS, make_model


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _first_scalar(value):
    arr = _numpy(value)
    return None if arr is None or arr.size == 0 else float(arr.reshape(-1)[0])


def _overlay(image: Image.Image, anomaly_map) -> str | None:
    """Return a visual contrast heatmap as base64 PNG (not a calibrated mask)."""
    arr = _numpy(anomaly_map)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D anomaly map; got {arr.shape}")
    arr = np.nan_to_num(arr.astype(np.float32))
    span = float(arr.max() - arr.min())
    scaled = np.zeros_like(arr) if span <= 1e-12 else (arr - arr.min()) / span
    gray = Image.fromarray((np.clip(scaled, 0, 1) * 255).astype("uint8"))
    gray = gray.resize(image.size, Image.Resampling.BILINEAR)
    intensity = np.asarray(gray, dtype=np.float32) / 255
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(original)
    color[..., 0] = 255 * intensity
    color[..., 1] = 50 * (1 - intensity)
    alpha = (0.58 * intensity)[..., None]
    blended = np.uint8(np.clip(original * (1 - alpha) + color * alpha, 0, 255))
    output = BytesIO()
    Image.fromarray(blended).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def predict_image(image_path: Path, checkpoint: Path, model_name: str = "padim",
                  accelerator: str = "cpu") -> dict:
    image_path, checkpoint = Path(image_path), Path(checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not checkpoint.is_file() or checkpoint.suffix != ".ckpt":
        raise FileNotFoundError("Missing Lightning .ckpt checkpoint")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    with Image.open(image_path) as img:
        img.verify()
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
    from anomalib.engine import Engine
    # Lightning checkpoints are pickle-based. NEVER accept them from user uploads.
    engine = Engine(accelerator=accelerator, devices=1, logger=False)
    start = time.perf_counter()
    batches = engine.predict(model=make_model(model_name, pre_trained=False),
                             data_path=image_path, ckpt_path=str(checkpoint),
                             return_predictions=True)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    if not batches:
        raise RuntimeError("Anomalib returned no predictions")
    pred = batches[0]
    # The first batch corresponds to the single supplied image.
    score = _first_scalar(getattr(pred, "pred_score", None))
    label_num = _first_scalar(getattr(pred, "pred_label", None))
    if score is None:
        raise RuntimeError("Prediction contained no anomaly score")
    return {
        "model": model_name,
        "anomaly_score": score,
        "predicted_anomaly": None if label_num is None else bool(label_num),
        "latency_ms": elapsed_ms,
        "heatmap_png_base64": _overlay(rgb, getattr(pred, "anomaly_map", None)),
        "notes": "Score is NOT a probability; heatmap is relative visualization.",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="padim")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    args = parser.parse_args(argv)
    result = predict_image(args.image, args.checkpoint, args.model, args.accelerator)
    import json
    encoded = result.pop("heatmap_png_base64")
    if encoded:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(base64.b64decode(encoded))
        result["heatmap_path"] = str(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
