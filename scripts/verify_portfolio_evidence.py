"""Read-only verification of saved benchmark and PatchCore smoke artifacts.

No training, checkpoint deserialization, model publishing or remote calls.
Verifies original per-file SHA-256 manifest, cross-report benchmark consistency,
saved Markdown summary and real smoke ZIP contents. This is an *artifact evidence*
gate only: passing it NEVER means a public or commercial release is authorized.

Usage:
 python -m scripts.verify_portfolio_evidence \
   --runs-dir "$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15" \
   --smoke-zip "$HOME/Downloads/ihsanvision_patchcore_smoke_evidence.zip" \
   --output "$HOME/ihsanvision-artifacts/portfolio_evidence_audit.json"
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

from PIL import Image

from .summarize_comparison import summarize


REQUIRED = frozenset({
    "padim-bottle-seed42/model.ckpt",
    "padim-bottle-seed42/report.json",
    "padim-bottle-seed42/splits.json",
    "patchcore-bottle-seed42/model.ckpt",
    "patchcore-bottle-seed42/report.json",
    "patchcore-bottle-seed42/splits.json",
    "comparison.json",
    "comparison.csv",
})
SMOKE_NAMES = frozenset({"smoke_report.json", "smoke_overlay.png"})
HASH_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


def _stream_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(runs_dir: Path) -> dict:
    """Restrict to expected relative files and verify all recorded hashes."""
    root = Path(runs_dir).resolve()
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError("SHA256SUMS.txt is missing")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise ValueError("Malformed SHA256 manifest line")
        checksum, name = fields
        if not HASH_PATTERN.fullmatch(checksum):
            raise ValueError("Malformed SHA256 checksum")
        if name not in REQUIRED or name in listed:
            raise ValueError("Unexpected/duplicate manifest member")
        listed[name] = checksum.lower()
    if set(listed) != REQUIRED:
        raise ValueError("Required evidence missing from SHA256 manifest")
    total_bytes = 0
    for name, expected in sorted(listed.items()):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Evidence missing: {name}")
        if not path.stat().st_size:
            raise ValueError(f"Evidence is empty: {name}")
        if _stream_sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        total_bytes += path.stat().st_size
    return {"files_verified": len(REQUIRED), "total_size_bytes": total_bytes}


def verify_summary(runs_dir: Path) -> dict:
    """Validate exact metrics/manifests via existing read-only benchmark auditor."""
    path = Path(runs_dir) / "benchmark_summary.md"
    if not path.is_file():
        raise FileNotFoundError("Missing benchmark_summary.md")
    regenerated = summarize(runs_dir)
    if path.read_text(encoding="utf-8") != regenerated:
        raise ValueError("Saved benchmark summary differs from source reports")
    return {"report_consistency": "verified", "saved_summary_matches": True}


def verify_smoke_zip(path: Path) -> dict:
    """Verify an operator-saved *two-file* smoke archive without extraction."""
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(SMOKE_NAMES) or set(names) != SMOKE_NAMES:
            raise ValueError("Expected exactly two PatchCore smoke-evidence files")
        for entry in archive.infolist():
            if entry.file_size > 20 * 1024 * 1024:
                raise ValueError("Smoke ZIP entry exceeds 20 MiB safety bound")
        if archive.testzip() is not None:
            raise ValueError("Smoke ZIP is corrupt")
        record = json.loads(archive.read("smoke_report.json"))
        if not isinstance(record, dict):
            raise ValueError("Invalid smoke-report JSON")
        if record.get("model") != "patchcore" or record.get("status") != "single-image-inference-passed":
            raise ValueError("Unexpected model/smoke status")
        if record.get("predicted_anomaly") is not True:
            raise ValueError("Expected observed defective-image anomaly")
        png = archive.read("smoke_overlay.png")
        with Image.open(BytesIO(png)) as image:
            if image.format != "PNG" or image.width < 1 or image.height < 1:
                raise ValueError("Smoke overlay is not a valid PNG")
            dimensions = [image.width, image.height]
            image.verify()
    return {"zip_integrity": "verified", "smoke_model": "patchcore",
            "status": "single-image-inference-passed", "overlay_dimensions": dimensions}


def verify(runs_dir: Path, smoke_zip: Path) -> dict:
    result = {
        "schema_version": 1,
        "status": "artifact_evidence_verified",
        "scientific_replication": False,
        "public_release_authorized": False,
        "source": "user_owned_local_saved_artifacts",
        "benchmark_manifest": verify_manifest(runs_dir),
        "benchmark": verify_summary(runs_dir),
        "patchcore_smoke": verify_smoke_zip(smoke_zip),
        "limitations": [
            "Checks original included checksums; cannot establish independent provenance.",
            "Benchmark consistency does not replicate training or prove generalized model quality.",
            "A single defective-image smoke result does not measure defect recall.",
            "No source image/mask/checkpoint is uploaded or deserialized.",
            "Licensing, security, actual deployment, end-to-end real UI and video remain separate release gates.",
        ],
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--smoke-zip", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        report = verify(args.runs_dir, args.smoke_zip)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Portfolio artifact evidence gate FAILED: {exc}") from None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2, allow_nan=False)
            writer.write("\n")
        print(f"Saved audit record: {args.output}")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
