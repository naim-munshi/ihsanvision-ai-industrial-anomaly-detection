"""Sequential, reproducible PaDiM-vs-PatchCore benchmark."""
import argparse
import csv
from pathlib import Path

from .config import CATEGORIES, TrainConfig, save_json
from .train import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=["bottle"])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args(argv)
    rows = []
    for category in args.categories:
        for model in ("padim", "patchcore"):
            report = run(TrainConfig(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                category=category, model=model,
                seed=args.seed, accelerator=args.accelerator,
            ))
            rows.append({
                "category": category, "model": model,
                "train_seconds": report["train_seconds"],
                **report["evaluation"]["metrics"],
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["category", "model", "train_seconds"] + sorted({
        k for row in rows for k in row if k not in ("category", "model", "train_seconds")
    })
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    save_json(args.output_dir / "comparison.json", {"rows": rows,
        "note": "Same seed/split protocol; model architectures differ. Do not compare to historical notebook metrics."})
    print(f"Saved {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
