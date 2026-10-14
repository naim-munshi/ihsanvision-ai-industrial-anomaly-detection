"""Sequential PaDiM-vs-PatchCore evaluation with split-equivalence checks.

All models in a category must use identical train, validation and held-out test
paths. This ensures that differing metrics reflect model behavior rather than
different evaluation subsets. No artificial benchmark results are generated.
"""
import argparse
import csv
import json
from pathlib import Path

from .config import CATEGORIES, TrainConfig, save_json
from .train import run


def read_split_manifest(path: Path) -> dict[str, list[str]]:
    """Read saved training splits, rejecting malformed or internally overlapping sets."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid split manifest: {path}")
    split_names = ("train", "val", "test")
    for name in split_names:
        values = manifest.get(name)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"Invalid {name} image paths in {path}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} paths in {path}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(manifest[a]) & set(manifest[b]):
            raise ValueError(f"Split leakage ({a}/{b}) in {path}")
    return {name: sorted(manifest[name]) for name in split_names}


def require_matching_splits(category: str, manifests: dict[str, dict]) -> dict[str, int]:
    """Fail closed if different model runs contain different images in any split."""
    if not {"padim", "patchcore"}.issubset(manifests):
        raise ValueError(f"Both PaDiM and PatchCore are required for {category}")
    first, second = manifests["padim"], manifests["patchcore"]
    for name in ("train", "val", "test"):
        if first[name] != second[name]:
            raise RuntimeError(
                f"Unfair benchmark for {category}: PaDiM/PatchCore {name} image paths differ"
            )
    return {name: len(first[name]) for name in ("train", "val", "test")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=["bottle"])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    rows = []
    split_counts = {}
    for category in args.categories:
        manifests = {}
        category_rows = []
        for model in ("padim", "patchcore"):
            report = run(TrainConfig(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                category=category, model=model,
                seed=args.seed, accelerator=args.accelerator,
            ))
            manifest_path = args.output_dir / f"{model}-{category}-seed{args.seed}" / "splits.json"
            manifests[model] = read_split_manifest(manifest_path)
            category_rows.append({
                "category": category, "model": model,
                "train_seconds": report["train_seconds"],
                **report["evaluation"]["metrics"],
            })
        # Only include category results after both manifests have passed.
        split_counts[category] = require_matching_splits(category, manifests)
        rows.extend(category_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["category", "model", "train_seconds"] + sorted({
        k for row in rows for k in row if k not in ("category", "model", "train_seconds")
    })
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    save_json(args.output_dir / "comparison.json", {
        "rows": rows,
        "seed": args.seed,
        "split_counts": split_counts,
        "split_equivalence": "verified_path_by_path",
        "note": (
            "Same seed and independently verified train/val/test images per category. "
            "Validation withheld from MVTec test; do not compare directly with "
            "full-test leaderboard metrics. Timing includes model-specific training work."
        ),
    })
    print(f"Saved {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
