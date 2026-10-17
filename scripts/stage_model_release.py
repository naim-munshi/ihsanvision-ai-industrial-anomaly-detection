"""Stage immutable, explicitly reviewed model registry/approval releases OFFLINE.

Never loads a model, changes an active server, or approves candidates implicitly.
Activate or roll back by selecting a verified release and restarting the server.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from ihsanvision.config import MODELS
from ihsanvision.provenance import CheckpointFingerprinter
from ihsanvision.registry import SUPPORTED_CATEGORIES

HASH = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON = 256_000
RELEASE_FILES = ("model-registry.json", "approved-models.json", "release.json")


def _private_json(path: Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Private input must be an absolute regular non-symlink file")
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Private manifest too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Private manifest must be an object")
    return data


def _pair(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or parts[0] not in SUPPORTED_CATEGORIES or parts[1] not in MODELS:
        raise ValueError("Model pair must be supported category:model")
    return parts[0], parts[1]


def _verified_entries(candidates: dict, selected: list[str]) -> list[dict]:
    if set(candidates) != {"version", "status", "models"} or (
        type(candidates["version"]) is not int or candidates["version"] != 1
        or candidates["status"] != "candidates_only_not_approved"
    ):
        raise ValueError("Expected an unapproved candidate inventory v1")
    values = candidates["models"]
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("Candidate inventory must contain 1-64 models")
    requested = [_pair(value) for value in selected]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Select one or more distinct model pairs")
    inventory = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {
            "category", "model", "checkpoint", "sha256", "seed", "protocol"
        }:
            raise ValueError("Unexpected candidate fields")
        key = _pair(f"{item['category']}:{item['model']}")
        if key in inventory:
            raise ValueError("Duplicate candidate model pair")
        if not isinstance(item["sha256"], str) or not HASH.fullmatch(item["sha256"]):
            raise ValueError("Malformed candidate checksum")
        if type(item["seed"]) is not int or item["seed"] < 0:
            raise ValueError("Invalid candidate seed")
        path = item["checkpoint"]
        if not isinstance(path, str) or not path:
            raise ValueError("Missing checkpoint")
        candidate = Path(path)
        if not candidate.is_absolute() or candidate.is_symlink() or (
            not candidate.is_file() or candidate.suffix.lower() != ".ckpt"
        ):
            raise ValueError("Candidate checkpoint must be a trusted absolute regular .ckpt file")
        inventory[key] = item

    fingerprinter = CheckpointFingerprinter()
    verified = []
    for key in requested:
        if key not in inventory:
            raise ValueError("Requested model pair absent from candidate inventory")
        item = inventory[key]
        checkpoint = Path(item["checkpoint"])
        identity = fingerprinter.identify(checkpoint)
        if identity.digest != item["sha256"]:
            raise ValueError("Candidate checkpoint SHA-256 mismatch")
        fingerprinter.verify_unchanged(checkpoint, identity)
        verified.append(item)
    return verified


def _write_private(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def stage_release(candidates_path: Path, destination: Path, selected: list[str],
                  *, reviewed: bool) -> Path:
    """Produce private immutable-looking manifests, never mutate the live service."""
    if reviewed is not True:
        raise ValueError("Manual scientific, licensing and checkpoint-safety review required")
    destination = Path(destination).expanduser()
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError("Release output must be a new absolute directory")
    entries = _verified_entries(_private_json(candidates_path), selected)
    model_registry = {"version": 1, "models": [
        {"category": e["category"], "model": e["model"], "checkpoint": e["checkpoint"]}
        for e in entries
    ]}
    approvals = {"version": 1, "approvals": [
        {"category": e["category"], "model": e["model"], "checkpoint_sha256": e["sha256"]}
        for e in entries
    ]}
    release = {
        "version": 1,
        "status": "staged_manual_activation_required",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved_pairs": selected,
        "notes": (
            "Operator affirmed external review; not a signed attestation, "
            "certification or automatic model quality check. Restart service to activate."
        ),
    }
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        for name, payload in zip(RELEASE_FILES, (model_registry, approvals, release)):
            _write_private(destination / name, payload)
        verify_release(destination)
    except Exception:
        # Only remove artifacts made in our newly created exclusive directory.
        for name in RELEASE_FILES:
            (destination / name).unlink(missing_ok=True)
        destination.rmdir()
        raise
    return destination


def verify_release(folder: Path) -> int:
    """Revalidate all staged checksum pins before switching service config."""
    folder = Path(folder).expanduser()
    if not folder.is_absolute() or folder.is_symlink() or not folder.is_dir():
        raise ValueError("Release must be an absolute, non-symlink directory")
    registry = _private_json(folder / RELEASE_FILES[0])
    approved = _private_json(folder / RELEASE_FILES[1])
    metadata = _private_json(folder / RELEASE_FILES[2])
    if (set(registry) != {"version", "models"} or registry["version"] != 1
            or set(approved) != {"version", "approvals"} or approved["version"] != 1
            or metadata.get("status") != "staged_manual_activation_required"):
        raise ValueError("Invalid staged release schema")
    models, approvals = registry["models"], approved["approvals"]
    if (not isinstance(models, list) or not isinstance(approvals, list)
            or not models or len(models) != len(approvals) or len(models) > 64):
        raise ValueError("Staged registry/approval pair count mismatch")
    pins = {}
    for item in approvals:
        if not isinstance(item, dict) or set(item) != {
            "category", "model", "checkpoint_sha256"
        }:
            raise ValueError("Invalid approval entry")
        pair = _pair(f"{item['category']}:{item['model']}")
        if pair in pins or not isinstance(item["checkpoint_sha256"], str) or (
                not HASH.fullmatch(item["checkpoint_sha256"])):
            raise ValueError("Duplicate or malformed approval")
        pins[pair] = item["checkpoint_sha256"]

    fingerprinter = CheckpointFingerprinter()
    seen = set()
    for item in models:
        if not isinstance(item, dict) or set(item) != {"category", "model", "checkpoint"}:
            raise ValueError("Invalid registry entry")
        pair = _pair(f"{item['category']}:{item['model']}")
        if pair in seen or pair not in pins:
            raise ValueError("Model/approval disagreement")
        seen.add(pair)
        path = Path(item["checkpoint"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked trusted checkpoint")
        identity = fingerprinter.identify(path)
        if identity.digest != pins[pair]:
            raise ValueError("Staged checkpoint integrity mismatch")
        fingerprinter.verify_unchanged(path, identity)
    if set(pins) != seen:
        raise ValueError("Unexpected approval pair")
    return len(seen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--candidates", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)
    stage.add_argument("--approve", nargs="+", required=True, metavar="CATEGORY:MODEL")
    stage.add_argument("--operator-reviewed", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            dest = stage_release(args.candidates, args.output_dir, args.approve,
                                 reviewed=args.operator_reviewed)
            print(f"Staged private release: {dest}. NOT activated. Restart required.")
        else:
            count = verify_release(args.release_dir)
            print(f"Verified {count} staged model pin(s). Not a model-quality test.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Model release refused: {exc}") from None


if __name__ == "__main__":
    main()
