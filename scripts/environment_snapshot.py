"""Capture a minimal, non-secret experiment environment snapshot without training.

Usage:
    python -m scripts.environment_snapshot --repo-dir .
    python -m scripts.environment_snapshot --repo-dir . --runs-dir runs --output runs/environment.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess


PACKAGES = ("python", "torch", "torchvision", "anomalib", "lightning", "numpy",
            "imagecodecs", "timm", "pandas", "scikit-learn")


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGES:
        if package == "python":
            result[package] = platform.python_version()
        else:
            try:
                result[package] = version(package)
            except PackageNotFoundError:
                result[package] = None
    return result


def git_value(repo_dir: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return completed.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def accelerator_info() -> dict:
    """Probe runtime availability without assuming torch or GPU support is installed."""
    try:
        import torch
    except ImportError:
        return {"torch_import": "missing", "cuda_available": None, "mps_available": None}
    except Exception as exc:
        return {"torch_import": f"failed: {type(exc).__name__}",
                "cuda_available": None, "mps_available": None}
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
    info = {"torch_import": "ok", "cuda_available": cuda, "mps_available": mps}
    if cuda:
        info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    return info


def collect(repo_dir: Path, runs_dir: Path | None = None) -> dict:
    repo_dir = Path(repo_dir).resolve()
    commit = git_value(repo_dir, "rev-parse", "HEAD")
    status = git_value(repo_dir, "status", "--porcelain")
    record = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
        "accelerator": accelerator_info(),
        "git": {
            "commit": commit,
            "branch": git_value(repo_dir, "branch", "--show-current"),
            "dirty": None if commit is None else bool(status),
        },
        "caveat": (
            "Runtime snapshot only; hardware availability does not establish which device "
            "actually executed a past experiment. Verify per-run accelerator and logs."
        ),
    }
    if runs_dir is not None:
        root = Path(runs_dir).resolve()
        record["runs"] = {
            name: {
                "checkpoint_present": (root / name / "model.ckpt").is_file(),
                "report_present": (root / name / "report.json").is_file(),
                "splits_present": (root / name / "splits.json").is_file(),
            }
            for name in ("padim-bottle-seed42", "patchcore-bottle-seed42")
        }
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file only; never overwrite")
    args = parser.parse_args(argv)
    serialized = json.dumps(collect(args.repo_dir, args.runs_dir), indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(serialized)
        print(f"Saved experiment environment: {args.output}")


if __name__ == "__main__":
    main()
