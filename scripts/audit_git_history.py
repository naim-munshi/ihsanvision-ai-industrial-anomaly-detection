"""Read-only audit of *reachable Git history* before a public v1 release.

Includes all fetched branches and tags. Detects sensitive historical paths
and high-confidence token patterns in bounded Git blobs, even if the file was
deleted from HEAD. Prints no historical secret bytes or private file paths.

This cannot attest copyright, license ownership, unreachable/garbage-
collected objects, external forks, or secrets hidden in encoded/binary content.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from scripts.check_release_hygiene import (
    APPROVED_EXAMPLES, BLOCKED_EXTENSIONS, MEDIA_EXTENSIONS,
    PRIVATE_TOP_LEVEL, TOKEN_PATTERNS,
)

MAX_HISTORY_PATHS = 100_000
MAX_BLOBS = 25_000
MAX_TEXT_BLOB_BYTES = 1_000_000
MAX_TEXT_BYTES_TOTAL = 150 * 1024 * 1024


def git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            input=input_bytes, stderr=subprocess.DEVNULL, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot inspect complete local Git history") from exc


def _redacted_path_id(path: str) -> str:
    """Anonymous stable reference so artifacts do not expose customer filenames."""
    return sha256(path.encode("utf-8", errors="replace")).hexdigest()[:16]


def _classify_path(name: str) -> str | None:
    posix = PurePosixPath(name)
    if posix.name.startswith(".env") and posix.name not in APPROVED_EXAMPLES:
        return "private_environment_file"
    if posix.parts and posix.parts[0] in PRIVATE_TOP_LEVEL:
        return "private_dataset_or_artifact_folder"
    if posix.suffix.lower() in BLOCKED_EXTENSIONS:
        return "model_weight_database_or_archive"
    if posix.suffix.lower() in MEDIA_EXTENSIONS:
        return "historical_media_requires_rights_review"
    return None


def audit(root: Path) -> dict:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Require a local Git checkout, not a source archive")
    # This checks EVERY FETCHED historical filename, including paths to
    # deleted files. A head-only scan would completely miss those.
    raw_paths = git(root, "log", "--all", "--name-only", "--format=", "-z")
    seen_paths = {
        item.decode("utf-8", errors="replace").strip()
        for item in raw_paths.split(b"\x00") if item.strip()
    }
    if len(seen_paths) > MAX_HISTORY_PATHS:
        raise RuntimeError("History path limit exceeded; manual audit required")
    findings = []
    for name in sorted(seen_paths):
        category = _classify_path(name)
        if category:
            findings.append({
                "type": category,
                "path_reference": _redacted_path_id(name),
            })

    # Read all reachable unique Git objects, NOT only current working files.
    raw_objects = git(root, "rev-list", "--objects", "--all")
    identities = {
        line.split(b" ", 1)[0].decode("ascii")
        for line in raw_objects.splitlines() if line
    }
    if len(identities) > MAX_BLOBS * 4:
        raise RuntimeError("History object limit exceeded; manual audit required")
    metadata = git(
        root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_bytes=("\n".join(sorted(identities)) + "\n").encode("ascii"),
    )
    blobs = []
    for line in metadata.splitlines():
        sha, kind, raw_size = line.decode("ascii").split()
        if kind == "blob":
            blobs.append((sha, int(raw_size)))
    if len(blobs) > MAX_BLOBS:
        raise RuntimeError("History blob limit exceeded; manual audit required")

    inspected, skipped_large, scanned_bytes = 0, 0, 0
    for identity, size in blobs:
        if size > MAX_TEXT_BLOB_BYTES:
            skipped_large += 1
            continue
        if scanned_bytes + size > MAX_TEXT_BYTES_TOTAL:
            raise RuntimeError("History content audit byte limit exceeded")
        payload = git(root, "cat-file", "blob", identity)
        scanned_bytes += len(payload)
        inspected += 1
        # Never emit matched secrets, nor raw historical filenames or lines.
        body = payload.decode("utf-8", errors="replace")
        for key, expression in TOKEN_PATTERNS.items():
            if expression.search(body):
                findings.append({
                    "type": "potential_" + key,
                    "blob_reference": identity[:16],
                })

    # Some relevant histories may have been omitted by shallow clone,
    # so explicitly report the shallow-clone condition rather than certify.
    shallow = git(root, "rev-parse", "--is-shallow-repository").strip() == b"true"
    return {
        "schema_version": 1,
        "status": ("incomplete_shallow_history" if shallow else
                   "findings_require_review" if findings else "no_known_patterns_found"),
        "audited_ref_scope": "all locally fetched branches/tags",
        "shallow_repository": shallow,
        "unique_historical_paths": len(seen_paths),
        "reachable_blob_count": len(blobs),
        "text_blobs_scanned": inspected,
        "large_blobs_not_content_scanned": skipped_large,
        "historical_text_bytes_scanned": scanned_bytes,
        "findings": findings,
        "full_rights_or_secret_clearance_verified": False,
        "v1_portfolio_release_verified": False,
        "limitations": [
            "Checks only reachable locally fetched Git refs; missing remotes, forks and garbage-collected objects are outside scope.",
            "Secret matching recognizes limited high-confidence patterns; unknown, encoded, encrypted or large binary payloads may contain hidden content.",
            "Historical media flags require a human license/consent assessment, not automatic copyright judgment.",
            "A clean scan does not revoke or erase credentials previously distributed; rotate any exposed secret.",
            "Real weights, dataset evidence, owner approvals and model/Docker acceptance are independent v1.0 release gates.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, help="New JSON evidence file; do not overwrite")
    args = parser.parse_args(argv)
    try:
        report = audit(args.repo)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Git history audit INCOMPLETE: {exc}") from None
    if args.output:
        output = args.output.expanduser()
        if output.exists() or output.is_symlink():
            parser.error("Refusing to overwrite an existing history report")
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
