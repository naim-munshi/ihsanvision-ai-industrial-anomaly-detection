"""Inspect Mac/local checkpoint artifacts and dependency installation without inference.

Usage:
    python -m scripts.local_preflight
    python -m scripts.local_preflight --checkpoint /path/to/model.ckpt
No checkpoint is deserialized here; no dataset or network is required.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform

REQUIRED_PACKAGES = ("torch", "anomalib", "numpy", "fastapi", "uvicorn")


def check_artifacts(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint).expanduser()
    output = {
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
    }
    folder = checkpoint.parent
    for name in ("report.json", "splits.json", "smoke_report.json", "smoke_overlay.png"):
        output[name] = (folder / name).is_file()

    if output["splits.json"]:
        splits = json.loads((folder / "splits.json").read_text(encoding="utf-8"))
        if not isinstance(splits, dict) or not all(
            isinstance(splits.get(s), list) for s in ("train", "val", "test")
        ):
            raise ValueError("splits.json must contain train, val and test lists")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(splits[a]).intersection(splits[b]):
                raise ValueError(f"Dataset leakage: {a}/{b} paths overlap")
        output["split_counts"] = {key: len(splits[key]) for key in ("train", "val", "test")}
        output["splits_disjoint"] = True

    if output["report.json"]:
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        output["model"] = report.get("model")
        output["category"] = report.get("category")
        output["metric_names"] = sorted(report.get("evaluation", {}).get("metrics", {}))
        if "split_counts" in output and report.get("split_counts") != output["split_counts"]:
            raise ValueError("report.json and splits.json disagree on sample counts")
    if output["smoke_report.json"]:
        smoke = json.loads((folder / "smoke_report.json").read_text(encoding="utf-8"))
        output["smoke_status"] = smoke.get("status")
        if "split_counts" in output and smoke.get("split_counts") != output["split_counts"]:
            raise ValueError("smoke_report.json and splits.json disagree on sample counts")
    output["artifacts_complete"] = bool(
        output["checkpoint_exists"] and output["checkpoint_bytes"] > 0
        and all(output[x] for x in ("report.json", "splits.json", "smoke_report.json", "smoke_overlay.png"))
    )
    return output


def environment_summary() -> dict:
    versions = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "operating_system": platform.system(),
        "packages": versions,
        "dependencies_present": all(v is not None for v in versions.values()),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(os.getenv(
            "IHSANVISION_CHECKPOINT",
            str(Path.home() / "ihsanvision-artifacts/padim-bottle-seed42/model.ckpt"),
        )),
    )
    args = parser.parse_args(argv)
    result = {"environment": environment_summary(), "artifacts": check_artifacts(args.checkpoint)}
    print(json.dumps(result, indent=2))
    if not result["environment"]["dependencies_present"] or not result["artifacts"]["artifacts_complete"]:
        raise SystemExit("Local preflight failed: missing dependencies or experiment artifacts")


if __name__ == "__main__":
    main()
