"""Fit, audit, checkpoint and evaluate an industrial anomaly detection model."""
import argparse
from pathlib import Path
import platform
import time

from .config import (CATEGORIES, MODELS, TrainConfig, audit_splits,
                     make_model, save_json, scalar_metrics, seed_everything)

def run(config: TrainConfig) -> dict:
    config.validate()
    from anomalib import __version__ as anomalib_version
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    import torch

    run_dir = config.output_dir / f"{config.model}-{config.category}-seed{config.seed}"
    if (run_dir / "report.json").exists():
        raise FileExistsError(f"Existing report: {run_dir / 'report.json'}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)

    # Validation uses a reserved fraction of MVTec's test directory.
    # The remaining images become the untouched final test set.
    data = MVTecAD(
        root=config.dataset_root, category=config.category,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        val_split_mode="from_test",
        val_split_ratio=config.validation_ratio,
        test_split_mode="from_dir", seed=config.seed,
    )
    data.setup()
    manifest = audit_splits(data)
    save_json(run_dir / "splits.json", manifest)
    model = make_model(config.model)
    engine = Engine(default_root_dir=str(run_dir / "engine"),
                    accelerator=config.accelerator, devices=1,
                    max_epochs=1, logger=False)
    start = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    duration = time.perf_counter() - start
    checkpoint = run_dir / "model.ckpt"
    engine.trainer.save_checkpoint(str(checkpoint))
    if not checkpoint.is_file():
        raise RuntimeError("Training did not create checkpoint")
    results = engine.test(model=model, datamodule=data, ckpt_path=str(checkpoint))
    report = {
        "model": config.model, "category": config.category,
        "config": config.serializable(),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__, "anomalib": anomalib_version},
        "train_seconds": round(duration, 3),
        "split_counts": {key: len(paths) for key, paths in manifest.items()},
        "evaluation": {
            "protocol": "Disjoint validation holdout from MVTec test; evaluate on remainder",
            "metrics": scalar_metrics(results),
            "note": "F1 depends on calibrated threshold; AUROC is threshold-independent.",
        },
        "checkpoint": str(checkpoint.resolve()),
    }
    save_json(run_dir / "report.json", report)
    return report

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--category", choices=CATEGORIES, default="bottle")
    parser.add_argument("--model", choices=MODELS, default="padim")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--accelerator", default="auto",
                        choices=("auto", "cpu", "gpu", "mps", "cuda"))
    args = parser.parse_args(argv)
    report = run(TrainConfig(
        dataset_root=args.dataset_root, category=args.category,
        model=args.model, output_dir=args.output_dir,
        seed=args.seed, train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, validation_ratio=args.validation_ratio,
        accelerator=args.accelerator))
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"Test metrics: {report['evaluation']['metrics']}")

if __name__ == "__main__":
    main()
