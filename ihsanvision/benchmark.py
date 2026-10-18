"""Sequential PaDiM-vs-PatchCore evaluation with split-equivalence checks.

Default: train both models for every selected category.
--reuse-existing: require completed PaDiM artifacts, reuse any completed
PatchCore artifacts, and only fit PatchCore when absent. Never silently
retrain PaDiM. Before the new PatchCore fit, enforce the PaDiM split manifest.
"""
import argparse
import csv
import json
import math
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


def run_folder(output_dir: Path, model: str, category: str, seed: int) -> Path:
    return output_dir / f"{model}-{category}-seed{seed}"


def read_completed_run(folder: Path, model: str, category: str, seed: int) -> tuple[dict, dict]:
    """Verify a reusable experiment report without deserializing the checkpoint.

    Serialized Lightning checkpoints are potentially unsafe: this preflight
    only checks the artifact's existence. Subsequent model loading must always
    be restricted to user-trusted checkpoints, never random online files.
    """
    required = ("model.ckpt", "report.json", "splits.json")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot reuse {model}/{category}: missing {', '.join(missing)} in {folder}. "
            "Preserve/upload all three artifacts; no automatic PaDiM retraining."
        )
    if (folder / "model.ckpt").stat().st_size == 0:
        raise ValueError(f"Empty checkpoint in {folder}")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("model") != model or report.get("category") != category:
        raise ValueError(f"Report does not match {model}/{category}: {folder}")
    config = report.get("config")
    if not isinstance(config, dict) or config.get("seed") != seed:
        raise ValueError(f"Report seed does not match {seed}: {folder}")

    manifest = read_split_manifest(folder / "splits.json")
    counts = {key: len(paths) for key, paths in manifest.items()}
    if report.get("split_counts") != counts:
        raise ValueError(f"Report split counts disagree with splits.json: {folder}")

    evaluation = report.get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing or invalid evaluation metrics in {folder}")
    for name in ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"):
        value = metrics.get(name)
        # bool is a subclass of int: accepting "true" from a damaged JSON
        # report as 1.0 would fabricate an apparently perfect AUROC/F1.
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing or invalid {name} in {folder}")
    duration = report.get("train_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Missing or invalid train_seconds in {folder}")
    return report, manifest


def preflight_existing_padim(output_dir: Path, categories: list[str], seed: int) -> dict[str, tuple[dict, dict]]:
    """Check *all* PaDiM artifacts before any compute starts."""
    found = {}
    for category in categories:
        folder = run_folder(output_dir, "padim", category, seed)
        found[category] = read_completed_run(folder, "padim", category, seed)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=["bottle"])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu", "mps", "cuda"))
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Require existing PaDiM report/splits/checkpoint. Train only missing PatchCore runs.",
    )
    args = parser.parse_args(argv)

    existing_padim = (
        preflight_existing_padim(args.output_dir, args.categories, args.seed)
        if args.reuse_existing else {}
    )

    rows = []
    split_counts = {}
    model_sources = {}
    for category in args.categories:
        manifests = {}
        category_rows = []
        model_sources[category] = {}
        for model in ("padim", "patchcore"):
            folder = run_folder(args.output_dir, model, category, args.seed)
            if args.reuse_existing and model == "padim":
                report, _ = existing_padim[category]
                source = "reused"
            elif args.reuse_existing and any((folder / f).exists() for f in ("report.json", "splits.json", "model.ckpt")):
                report, _ = read_completed_run(folder, model, category, args.seed)
                source = "reused"
            else:
                # For new PatchCore work, compare dataset paths *before* fitting
                # against the already completed PaDiM split manifest.
                reference = (
                    run_folder(args.output_dir, "padim", category, args.seed) / "splits.json"
                    if args.reuse_existing and model == "patchcore" else None
                )
                report = run(TrainConfig(
                    dataset_root=args.dataset_root,
                    output_dir=args.output_dir,
                    category=category, model=model,
                    seed=args.seed, accelerator=args.accelerator,
                    reference_splits=reference,
                ))
                source = "trained"
            manifests[model] = read_split_manifest(folder / "splits.json")
            category_rows.append({
                "category": category, "model": model,
                "train_seconds": report["train_seconds"],
                **report["evaluation"]["metrics"],
            })
            model_sources[category][model] = source
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
        "model_sources": model_sources,
        "note": (
            "Exact train/val/test images checked per category. "
            "Validation held out from the original MVTec test; do not directly compare "
            "with full-test leaderboard scores. Reused training times were recorded "
            "in earlier runs, potentially using different hardware."
        ),
    })
    print(f"Saved {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
