"""Static Git source-release hygiene audit (no model runtime, no network).

Examine exactly the git-tracked source tree for accidental datasets, model
weights, unreviewed media and high-confidence secret patterns. This checks
public SOURCE HYGIENE only: it cannot certify model science, Docker inference,
rights ownership, dependency vulnerability status or portfolio completion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

MAX_SOURCE_BYTES = 1_000_000
BLOCKED_EXTENSIONS = {
    ".ckpt", ".pt", ".pth", ".onnx", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".sqlite", ".sqlite3", ".db", ".pem", ".key",
    ".zip", ".tar", ".gz", ".7z", ".parquet", ".h5", ".hdf5",
}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
                    ".mp4", ".mov", ".webm"}
PRIVATE_TOP_LEVEL = {"data", "datasets", "runs", "outputs", "results", "private",
                     "checkpoints", "weights"}
APPROVED_EXAMPLES = {".env.example", ".env.docker.example"}
MEDIA_RIGHTS_FILE = "docs/release-media-rights.md"
TOKEN_PATTERNS = {
    "github_classic_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}"),
    "github_fine_grained_token": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "aws_access_key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "private_key_material": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
}
MEDIA_PREFIXES = ("docs/media/", "demo/media/")


def tracked_paths(root: Path) -> list[str]:
    """Read Git's tracked file inventory; never traverse ignored private data."""
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached"], timeout=15
    )
    names = [name.decode("utf-8") for name in output.split(b"\x00") if name]
    if not names:
        raise ValueError("Not a checkout with tracked files")
    return names


def audit_source(root: Path, names: list[str]) -> dict:
    root = Path(root).resolve()
    findings: list[str] = []
    media: list[str] = []
    seen: set[str] = set()
    for name in names:
        posix = PurePosixPath(name)
        if (posix.is_absolute() or ".." in posix.parts or name in seen
                or not name or "\\" in name):
            findings.append(f"Invalid or duplicate tracked path: {name}")
            continue
        seen.add(name)
        path = root.joinpath(*posix.parts)
        if path.is_symlink() or not path.is_file():
            findings.append(f"Tracked file is absent or a symlink: {name}")
            continue
        if posix.parts[0] in PRIVATE_TOP_LEVEL:
            findings.append(f"Private dataset/artifact directory is tracked: {name}")
            continue
        if (posix.name.startswith(".env")
                and posix.name not in APPROVED_EXAMPLES):
            findings.append(f"Real environment configuration is tracked: {name}")
            continue
        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            findings.append(f"Binary, model or private artifact is tracked: {name}")
            continue
        if path.suffix.lower() in MEDIA_EXTENSIONS:
            media.append(name)
            if not name.startswith(MEDIA_PREFIXES):
                findings.append(f"Media outside designated release folders: {name}")
            continue
        if path.stat().st_size > MAX_SOURCE_BYTES:
            findings.append(f"Oversized source file requires manual review: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append(f"Unexpected non-text tracked source: {name}")
            continue
        for token_name, expression in TOKEN_PATTERNS.items():
            if expression.search(content):
                # Never expose matched secret bytes in CI logs.
                findings.append(f"Potential {token_name} in tracked source: {name}")
    if media and MEDIA_RIGHTS_FILE not in seen:
        findings.append(
            "Media exists but the owner-approved rights register is missing: "
            + MEDIA_RIGHTS_FILE
        )
    return {
        "status": "blocked" if findings else "source_hygiene_passed",
        "tracked_files_checked": len(names),
        "release_media_count": len(media),
        "license_selected": (root / "LICENSE").is_file() and "LICENSE" in seen,
        "source_hygiene_findings": findings,
        "pending_external_release_gates": [
            "Evidence of completed real Bottle/Cable/Carpet paired experiments",
            "Latest local real-model end-to-end and Docker acceptance",
            "Approved/rights-cleared real screenshots and demo recording",
            "Owner legal review of responsible-use source license and source-media rights",
            "Reviewed versioned GitHub release",
        ],
        "v1_portfolio_release_verified": False,
        "meaning": (
            "Static source-only audit; never confirms experimental metrics, "
            "licenses, real Docker inference, human approvals or a v1.0 release."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--ci", action="store_true",
                        help="Exit nonzero only for public-source hygiene failures")
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    try:
        report = audit_source(root, tracked_paths(root))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            UnicodeError, ValueError) as exc:
        parser.error(f"Could not audit Git-tracked source: {exc}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.ci and report["status"] != "source_hygiene_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
