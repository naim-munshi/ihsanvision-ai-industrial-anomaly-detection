"""Plan and explicitly resume MVTec AD PaDiM/PatchCore multi-category experiments.

Dry-run by default. Never overwrites partial artifacts or trains a previously
completed run. Before any compute, validates every selected category and checks
existing split manifests and saved evaluation reports. Requires an operator's
local dataset and Anomalib environment to actually train.

  python -m scripts.resume_mvtec_training \
    --dataset-root /private/mvtec_ad --runs-dir /private/ihsanvision_runs
  python -m scripts.resume_mvtec_training \
    --dataset-root /private/mvtec_ad --runs-dir /private/ihsanvision_runs \
    --categories bottle cable --execute --max-jobs 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ihsanvision.benchmark import (
    read_completed_run, read_split_manifest, require_matching_splits, run_folder,
)
from ihsanvision.config import CATEGORIES, TrainConfig
from ihsanvision.train import run
from scripts.kaggle_preflight import scan_dataset

ARTIFACTS = ("model.ckpt", "report.json", "splits.json")

def require_external_runs(dataset_root: Path, runs_dir: Path) -> None:
    """Reject output directories inside the source tree, including symlink aliases."""
    source = Path(dataset_root).expanduser().resolve()
    destination = Path(runs_dir).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Training runs directory must be outside the dataset root")



def plan(dataset_root: Path, runs_dir: Path, categories: list[str],
         seed: int) -> dict:
    """Check *all* inputs and saved experiments before permitting any training."""
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Specify at least one nonduplicate category")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("Unsupported MVTec AD category")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    dataset_root = Path(dataset_root).expanduser()
    runs_dir = Path(runs_dir).expanduser()
    require_external_runs(dataset_root, runs_dir)
    jobs: list[dict] = []
    existing_pairs: list[dict] = []
    input_counts: list[dict] = []
    for category in categories:
        # Require all three semantic source splits to be available even for
        # completed-run reuse. Never silently claim readiness on incomplete data.
        dataset = scan_dataset(dataset_root, category)
        input_counts.append({
            "category": category,
            "train_good": dataset["train_good_images"],
            "test_good": dataset["test_good_images"],
            "test_anomaly": dataset["test_anomaly_images"],
        })
        present: dict[str, bool] = {}
        manifests: dict[str, dict] = {}
        for model in ("padim", "patchcore"):
            folder = run_folder(runs_dir, model, category, seed)
            # Direct training now exclusively creates run directories.
            # Reject even an empty/engine-only interrupted directory before
            # claiming that a new fit can be safely resumed. Symlinked run
            # directories must not be treated as trusted completed artifacts.
            if folder.is_symlink():
                raise RuntimeError(
                    f"Symlinked {category}/{model} run directory found; "
                    "preserve and inspect it before resuming."
                )
            artifacts = [name for name in ARTIFACTS if (folder / name).exists()]
            if folder.exists() and not artifacts:
                raise RuntimeError(
                    f"Existing {category}/{model} run directory has no completed "
                    "artifact trio. It may be an interrupted run; inspect or "
                    "archive it before resuming."
                )
            if 0 < len(artifacts) < len(ARTIFACTS):
                raise RuntimeError(
                    f"Partial {category}/{model} artifacts found. Do not overwrite; "
                    "inspect or archive them before resuming."
                )
            if len(artifacts) == len(ARTIFACTS):
                # Reject malformed saved reports, overlapping or missing
                # split manifests, wrong seeds and unusable metrics; this does
                # not cryptographically verify checkpoint authenticity.
                _, manifest = read_completed_run(folder, model, category, seed)
                manifests[model] = manifest
                present[model] = True
            else:
                present[model] = False
                jobs.append({"category": category, "model": model})
        if len(manifests) == 2:
            require_matching_splits(category, manifests)
        existing_pairs.append({
            "category": category,
            "completed_models": [model for model in ("padim", "patchcore") if present[model]],
            "missing_models": [model for model in ("padim", "patchcore") if not present[model]],
        })
    return {
        "schema_version": 1,
        "type": "mvtec_ad_resumable_training_plan",
        "status": "all_completed" if not jobs else "jobs_missing",
        "seed": seed,
        "categories": categories,
        "existing": existing_pairs,
        "jobs": jobs,
        "training_jobs_pending": len(jobs),
        "dataset_counts": input_counts,
        "notes": [
            "Dry-run unless --execute is supplied. No training is performed during planning.",
            "Completed model reports and train/val/test split manifests are validated and reused.",
            "When the second model of a category is fitted, its dataset split must match the first.",
            "Existing run directories with partial model artifacts are refused to avoid data loss.",
            "Only original MVTec AD layouts are supported; MVTec AD 2 requires its own adapter.",
            "No accuracy, hardware runtime or new checkpoints are claimed by this plan.",
        ],
    }


def execute(plan_data: dict, dataset_root: Path, runs_dir: Path, *,
            seed: int, accelerator: str, max_jobs: int = 1,
            trainer=run) -> list[dict]:
    """Run a *bounded* number of missing jobs, never overwriting completed runs."""
    require_external_runs(dataset_root, runs_dir)
    if type(max_jobs) is not int or not 1 <= max_jobs <= 4:
        raise ValueError("max_jobs must be between 1 and 4")
    if not isinstance(plan_data, dict) or plan_data.get("schema_version") != 1:
        raise ValueError("First provide a validated training plan")
    if accelerator not in {"auto", "cpu", "gpu", "mps", "cuda"}:
        raise ValueError("Unsupported accelerator")
    completed: list[dict] = []
    for task in plan_data["jobs"][:max_jobs]:
        category, model = task["category"], task["model"]
        folder = run_folder(runs_dir, model, category, seed)
        if folder.exists() or folder.is_symlink():
            raise RuntimeError(
                f"{category}/{model} run directory appeared since planning; refusing overwrite"
            )
        other_model = "patchcore" if model == "padim" else "padim"
        reference = run_folder(runs_dir, other_model, category, seed) / "splits.json"
        reference_splits = reference if reference.is_file() else None
        report = trainer(TrainConfig(
            dataset_root=Path(dataset_root),
            category=category,
            model=model,
            output_dir=Path(runs_dir),
            seed=seed,
            accelerator=accelerator,
            reference_splits=reference_splits,
        ))
        # Verify the completed artifact contract before proceeding to the next
        # training job. A failed run remains for operator investigation.
        read_completed_run(folder, model, category, seed)
        completed.append({"category": category, "model": model, "status": "trained_and_verified"})
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps", "cuda"), default="auto")
    parser.add_argument("--execute", action="store_true", help="Explicitly fit missing models")
    parser.add_argument("--max-jobs", type=int, default=1, help="Bounded new fits (1-4) per invocation")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_jobs <= 4:
            raise ValueError("max-jobs must be between 1 and 4")
        result = plan(args.dataset_root, args.runs_dir, args.categories, args.seed)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.execute and result["jobs"]:
            outcomes = execute(
                result, args.dataset_root, args.runs_dir, seed=args.seed,
                accelerator=args.accelerator, max_jobs=args.max_jobs,
            )
            print(json.dumps({"executed": outcomes}, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Resumable training validation failed: {exc}") from None


if __name__ == "__main__":
    main()
