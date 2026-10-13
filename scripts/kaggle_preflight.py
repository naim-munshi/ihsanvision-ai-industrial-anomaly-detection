"""Find a local MVTec AD dataset and validate it before training.

Usage:
    python -m scripts.kaggle_preflight
    python -m scripts.kaggle_preflight --input-dir /kaggle/input
    python -m scripts.kaggle_preflight --dataset-root /path/to/MVTecAD

This script never downloads training data or starts training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan_dataset(root: Path, category: str = "bottle") -> dict:
    root = Path(root).resolve()
    train = root / category / "train" / "good"
    test = root / category / "test"
    if not train.is_dir() or not test.is_dir():
        raise ValueError(f"Missing MVTec AD {category}/train/good or {category}/test in {root}")
    normal = sorted(p for p in train.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    test_images = sorted(p for p in test.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
    test_normal = [p for p in test_images if p.parent.name == "good"]
    test_abnormal = [p for p in test_images if p.parent.name != "good"]
    if not normal or not test_normal or not test_abnormal:
        raise ValueError("Train normal, test normal, and test anomaly images are all required")
    return {
        "dataset_root": str(root),
        "category": category,
        "train_good_images": len(normal),
        "test_good_images": len(test_normal),
        "test_anomaly_images": len(test_abnormal),
        "test_total_images": len(test_images),
        "ground_truth_directory": (root / category / "ground_truth").is_dir(),
    }


def find_dataset_roots(input_dir: Path, category: str = "bottle") -> list[Path]:
    """Find real MVTec directory structures, not just folders called 'mvtec'."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    roots = []
    # The user may supply the dataset directory itself.
    if (input_dir / category / "train" / "good").is_dir():
        roots.append(input_dir.resolve())
    else:
        # Kaggle often places datasets at input/datasets/<owner>/<slug>.
        for train in input_dir.rglob(f"{category}/train/good"):
            root = train.parents[2].resolve()
            if (root / category / "test").is_dir() and root not in roots:
                roots.append(root)
    return sorted(roots)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--category", choices=CATEGORIES, default="bottle")
    args = parser.parse_args(argv)
    if args.dataset_root:
        candidates = [args.dataset_root]
    else:
        candidates = find_dataset_roots(args.input_dir, args.category)
    if not candidates:
        parser.error("No MVTec AD dataset found. Check Kaggle Add Input and dataset extraction.")
    for candidate in candidates:
        try:
            print(json.dumps(scan_dataset(candidate, args.category), indent=2))
        except ValueError as exc:
            print(f"Invalid dataset at {candidate}: {exc}")
    print("Preflight only: no model was trained and no dataset was downloaded.")


if __name__ == "__main__":
    main()
