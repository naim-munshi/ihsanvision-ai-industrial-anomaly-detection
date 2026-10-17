"""MVTec AD 2 PaDiM/PatchCore training with PUBLIC-ONLY local evaluation.

Unlike classic MVTec AD, MVTec AD 2 supplies normal-only train/validation
and test_public with labeled anomaly masks. This workflow NEVER evaluates
test_private or test_private_mixed (official leaderboard evaluation only).

Dry-run by default, no downloads. Actual fitting requires a locally
obtained, legally usable dataset, the Anomalib runtime, an explicit
--execute and a separate --acknowledge-noncommercial flag.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import time

from .ad2 import preflight_ad2, verify_ad2_splits
from .config import MODELS, audit_splits, make_model, save_json, scalar_metrics, seed_everything
from .registry import AD2_CATEGORIES


@dataclass(frozen=True)
class AD2TrainConfig:
    dataset_root: Path
    output_dir: Path
    category: str
    model: str
    seed: int = 42
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    accelerator: str = "auto"

    def validate(self) -> dict:
        if self.category not in AD2_CATEGORIES or self.model not in MODELS:
            raise ValueError("Unsupported AD2 category/model")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if self.train_batch_size < 1 or self.eval_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid batch size or worker count")
        if self.accelerator not in {"auto", "cpu", "gpu", "mps", "cuda"}:
            raise ValueError("Unsupported accelerator")
        # Never write checkpoints or reports into the source dataset tree.
        source = self.dataset_root.resolve()
        destination = self.run_folder().resolve()
        if destination == source or source in destination.parents:
            raise ValueError("AD2 output must be outside the dataset root")
        return preflight_ad2(self.dataset_root, self.category)

    def run_folder(self) -> Path:
        return Path(self.output_dir) / f"ad2-{self.model}-{self.category}-seed{self.seed}"


def run(config: AD2TrainConfig) -> dict:
    """Execute a SINGLE explicit run after preflight; never overwrite artifacts."""
    counts = config.validate()
    folder = config.run_folder()
    if folder.exists():
        raise FileExistsError(
            f"Existing AD2 run directory: {folder}. "
            "Inspect and preserve previous artifacts before retrying."
        )
    # Imports/possible pretrained backbone download occur only during
    # explicitly requested execution, after data & destination preflight.
    from anomalib import __version__ as anomalib_version
    from anomalib.data import MVTecAD2
    from anomalib.engine import Engine
    import torch

    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    # AD2's separate normal validation set MUST NOT be built from public test
    # or the private test sets. This datamodule uses the public test by default
    # and we pin it explicitly for compatibility clarity.
    data = MVTecAD2(
        root=config.dataset_root,
        category=config.category,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        test_type="public",
        seed=config.seed,
    )
    data.setup()
    manifest = audit_splits(data)
    verify_ad2_splits(manifest, config.dataset_root, config.category)
    save_json(folder / "splits.json", manifest)
    model = make_model(config.model)
    engine = Engine(
        default_root_dir=str(folder / "engine"),
        accelerator=config.accelerator,
        devices=1,
        max_epochs=1,
        logger=False,
    )
    start = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    elapsed = time.perf_counter() - start
    checkpoint = folder / "model.ckpt"
    engine.trainer.save_checkpoint(str(checkpoint))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("No nonempty AD2 checkpoint was created")
    evaluated = engine.test(
        model=model, datamodule=data, ckpt_path=str(checkpoint)
    )
    report = {
        "dataset": "mvtec_ad_2",
        "model": config.model,
        "category": config.category,
        "seed": config.seed,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "anomalib": anomalib_version,
        },
        "training": {
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "num_workers": config.num_workers,
            "accelerator": config.accelerator,
            "elapsed_seconds": round(elapsed, 3),
        },
        "splits": {name: len(paths) for name, paths in manifest.items()},
        "source_counts": counts,
        "evaluation": {
            "protocol": "dedicated_normal_validation_then_public_test_only",
            "test_type": "public",
            "metrics": scalar_metrics(evaluated),
            "warning": (
                "Private AD2 leaderboard labels are unavailable. "
                "Public-test results are not official private benchmark results."
            ),
        },
        "checkpoint": str(checkpoint.resolve()),
    }
    save_json(folder / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", choices=AD2_CATEGORIES, required=True)
    parser.add_argument("--model", choices=MODELS, default="padim")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-noncommercial", action="store_true",
                        help="Operator confirms dataset non-commercial license conditions")
    args = parser.parse_args(argv)
    config = AD2TrainConfig(
        dataset_root=args.dataset_root, output_dir=args.output_dir,
        category=args.category, model=args.model, seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        accelerator=args.accelerator,
    )
    try:
        counts = config.validate()
        print(
            "MVTec AD 2 public-only preflight: "
            f"category={args.category}, model={args.model}, "
            f"train_normal={counts['train_normal']}, "
            f"validation_normal={counts['validation_normal']}, "
            f"public_good={counts['public_test_normal']}, "
            f"public_bad={counts['public_test_anomalous']}."
        )
        if config.run_folder().exists():
            raise FileExistsError(
                "Existing AD2 run folder; inspect before retrying, no overwrite"
            )
        if not args.execute:
            print("DRY RUN: dataset layout checked; no training, downloads or model loading.")
            return
        if not args.acknowledge_noncommercial:
            raise ValueError(
                "--execute requires --acknowledge-noncommercial "
                "after independently reviewing MVTec AD 2 terms"
            )
        report = run(config)
        print(f"Saved checkpoint: {report['checkpoint']}")
        print(f"PUBLIC test metrics (not official private benchmark): {report['evaluation']['metrics']}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"MVTec AD 2 preflight/training refused: {exc}") from None


if __name__ == "__main__":
    main()
