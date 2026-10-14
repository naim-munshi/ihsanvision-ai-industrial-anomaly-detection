"""Stage a trusted existing PaDiM experiment for an incremental benchmark.

Example (Kaggle; after attaching the user's saved artifacts as a private input):
    python -m scripts.stage_reference_artifacts \
        --source-dir /kaggle/input/ihsanvision-padim/padim-bottle-seed42 \
        --output-dir /kaggle/working/ihsanvision_runs

Copies only report.json, splits.json and model.ckpt into the expected benchmark
folder. Does not train, load/deserialise model weights or overwrite existing runs.
The operator must trust the checkpoint's provenance before any later inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from src.benchmark import read_completed_run
from src.config import CATEGORIES


ESSENTIAL_ARTIFACTS = ("model.ckpt", "report.json", "splits.json")


def stage_reference(
    source_dir: Path, output_dir: Path, category: str = "bottle", seed: int = 42
) -> Path:
    if category not in CATEGORIES:
        raise ValueError("Unsupported MVTec category")
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    folder_name = f"padim-{category}-seed{seed}"
    destination = output_dir / folder_name
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark artifact directory: {destination}"
        )

    # Validate source before copying 168 MB+ of weights. No torch.load is used.
    read_completed_run(source_dir, "padim", category, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-padim-", dir=output_dir) as scratch:
        staged = Path(scratch) / folder_name
        staged.mkdir()
        for name in ESSENTIAL_ARTIFACTS:
            shutil.copy2(source_dir / name, staged / name)
        read_completed_run(staged, "padim", category, seed)
        # On the same volume, a rename makes all three files visible together.
        # Do not overwrite an existing destination even if another process raced.
        if destination.exists():
            raise FileExistsError(f"Destination was created during staging: {destination}")
        staged.rename(destination)
    return destination


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, default="bottle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    path = stage_reference(args.source_dir, args.output_dir, args.category, args.seed)
    print(f"Staged reference PaDiM artifacts: {path}")
    print("No weights deserialized, no model trained; ensure checkpoint source is trusted.")


if __name__ == "__main__":
    main()
