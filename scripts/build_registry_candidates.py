"""Build a PRIVATE candidate model registry from validated classic MVTec runs.

Never loads checkpoints, auto-approves a model, or publishes artifacts.
Only explicit categories are scanned; any partial/invalid run fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ihsanvision.benchmark import read_completed_run, require_matching_splits, run_folder
from ihsanvision.config import CATEGORIES


def fingerprint(path: Path) -> str:
    """Hash trusted checkpoint bytes without deserializing them."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidates(runs_dir: Path, categories: list[str], seed: int) -> dict:
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Select distinct categories")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("Unsupported classic MVTec category")
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid seed")
    models = []
    for category in categories:
        manifests = {}
        for model in ("padim", "patchcore"):
            folder = run_folder(Path(runs_dir), model, category, seed)
            report, manifest = read_completed_run(folder, model, category, seed)
            manifests[model] = manifest
            checkpoint = folder / "model.ckpt"
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise ValueError("Checkpoint must be a regular non-symlink file")
            resolved = checkpoint.resolve()
            stat_before = checkpoint.stat()
            digest = fingerprint(checkpoint)
            stat_after = checkpoint.stat()
            if (stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                    stat_before.st_mtime_ns, stat_before.st_ctime_ns) != (
                    stat_after.st_dev, stat_after.st_ino, stat_after.st_size,
                    stat_after.st_mtime_ns, stat_after.st_ctime_ns):
                raise RuntimeError("Checkpoint changed while hashing")
            models.append({
                "category": category, "model": model,
                "checkpoint": str(resolved), "sha256": digest,
                "seed": seed, "protocol": report["evaluation"].get("protocol"),
            })
        # A candidate pair is scientifically invalid when its train/val/test
        # samples differ, even if both individual reports are well-formed.
        require_matching_splits(category, manifests)
    return {"version": 1, "status": "candidates_only_not_approved", "models": models}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = args.output.expanduser().resolve()
        source = args.runs_dir.expanduser().resolve()
        if destination == source or source in destination.parents:
            raise ValueError("Candidate output must be outside experiment runs")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Candidate output exists; refusing overwrite")
        candidates = build_candidates(args.runs_dir, args.categories, args.seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(candidates, stream, indent=2, allow_nan=False)
            stream.write("\n")
        destination.chmod(0o600)
        print("Saved private candidate registry; NOT approved for serving.")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"Candidate registry refused: {exc}") from None


if __name__ == "__main__":
    main()
