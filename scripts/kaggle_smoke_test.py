"""Verify a Kaggle-trained checkpoint on a known defective MVTec image.

Run from the repository directory:
    python -m scripts.kaggle_smoke_test \
      --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
      --output-dir /kaggle/working/ihsanvision_runs

This runs REAL model inference using a trusted local Lightning checkpoint.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

from src.config import CATEGORIES, MODELS, save_json
from src.predict import predict_image

SUPPORTED_IMAGES = {".png", ".jpeg", ".jpg", ".bmp", ".webp"}


def choose_defective_image(dataset_root: Path, category: str) -> Path:
    test = dataset_root / category / "test"
    candidates = sorted(
        p for p in test.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_IMAGES
        and p.parent.name != "good"
    ) if test.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No defective test image found in {test}")
    return candidates[0]


def verify_checkpoint(
    dataset_root: Path,
    output_dir: Path,
    category: str = "bottle",
    model: str = "padim",
    seed: int = 42,
    accelerator: str = "cpu",
    predictor=None,
) -> dict:
    if model not in MODELS or category not in CATEGORIES:
        raise ValueError("Unknown model or MVTec category")
    if predictor is None:
        predictor = predict_image
    folder = output_dir / f"{model}-{category}-seed{seed}"
    checkpoint = folder / "model.ckpt"
    report = folder / "report.json"
    manifest = folder / "splits.json"
    required = (checkpoint, report, manifest)
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Training artifact missing: {p}")
    import json
    splits = json.loads(manifest.read_text(encoding="utf-8"))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if not set(splits[a]).isdisjoint(splits[b]):
            raise ValueError(f"Data leakage between {a} and {b}")

    image = choose_defective_image(dataset_root, category)
    result = predictor(image, checkpoint, model, accelerator)
    if result["anomaly_score"] is None:
        raise RuntimeError("Model inference returned no anomaly score")
    overlay = result.get("heatmap_png_base64")
    if not overlay:
        raise RuntimeError("No localization overlay produced")
    raw = base64.b64decode(overlay, validate=True)
    from PIL import Image
    from io import BytesIO
    with Image.open(BytesIO(raw)) as vis:
        if vis.format != "PNG":
            raise RuntimeError("Localization overlay must be a PNG")
        vis.verify()

    overlay_path = folder / "smoke_overlay.png"
    overlay_path.write_bytes(raw)
    summary = {
        "image": str(image),
        "checkpoint": str(checkpoint),
        "overlay": str(overlay_path),
        "model": model,
        "category": category,
        "anomaly_score": float(result["anomaly_score"]),
        "predicted_anomaly": result["predicted_anomaly"],
        "latency_ms": result["latency_ms"],
        "split_counts": {k: len(splits[k]) for k in ("train", "val", "test")},
        "status": "single-image-inference-passed",
        "note": "Raw anomaly score is not a probability; heatmap is visual-only.",
    }
    save_json(folder / "smoke_report.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/kaggle/input/datasets/ipythonx/mvtec-ad"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/kaggle/working/ihsanvision_runs"))
    parser.add_argument("--category", choices=CATEGORIES, default="bottle")
    parser.add_argument("--model", choices=MODELS, default="padim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="cpu")
    args = parser.parse_args(argv)
    summary = verify_checkpoint(
        args.dataset_root, args.output_dir, args.category,
        args.model, args.seed, args.accelerator,
    )
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
