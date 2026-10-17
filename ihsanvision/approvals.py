"""Operator-controlled, opt-in checkpoint approval pins.

This manifest is a *local allowlist* of expected checkpoint SHA-256 bytes
for specific category/model pairs. It is NOT evidence of model accuracy,
dataset rights, signatures, security review or regulatory certification.

File format:
{
  "version": 1,
  "approvals": [
    {"category": "bottle", "model": "padim",
     "checkpoint_sha256": "<exact 64 lowercase hex chars>"}
  ]
}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .config import MODELS
from .registry import SUPPORTED_CATEGORIES

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_BYTES = 64_000


def configured_approvals() -> dict[tuple[str, str], str] | None:
    """None: legacy/unpinned mode. Dict: enforce exactly these expected hashes.

    Approvals are snapshotted once on app startup. A modified manifest
    requires server restart; inference rechecks checkpoint identity per request.
    """
    raw_setting = os.environ.get("IHSANVISION_APPROVED_MODELS")
    if raw_setting is None:
        return None
    raw = raw_setting.strip()
    if not raw:
        raise ValueError(
            "IHSANVISION_APPROVED_MODELS was set but empty: "
            "refusing to silently disable approval enforcement"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("IHSANVISION_APPROVED_MODELS requires an absolute local file")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Approval manifest must be a regular non-symlink file")
    try:
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("Approval manifest exceeds size limit")
        with path.open("r", encoding="utf-8") as file:
            body = file.read(_MAX_BYTES + 1)
    except (OSError, UnicodeError) as exc:
        raise ValueError("Unable to read local approval manifest") from exc
    if len(body.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("Approval manifest exceeds size limit")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Approval manifest is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {"version", "approvals"}:
        raise ValueError("Approval manifest requires exactly version and approvals")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported approval manifest version")
    values = data["approvals"]
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("Approval list must have 1-64 entries")
    approvals: dict[tuple[str, str], str] = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {
            "category", "model", "checkpoint_sha256"
        }:
            raise ValueError("Malformed approval entry")
        category, model, digest = (
            item["category"], item["model"], item["checkpoint_sha256"]
        )
        if category not in SUPPORTED_CATEGORIES or model not in MODELS:
            raise ValueError("Unsupported approved category/model")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("Approval SHA-256 must be 64 lowercase hex characters")
        pair = (category, model)
        if pair in approvals:
            raise ValueError("Duplicate category/model approval")
        approvals[pair] = digest
    return approvals


def restrict_catalog(
    catalog: dict[str, dict[str, Path]],
    approvals: dict[tuple[str, str], str] | None,
) -> dict[str, dict[str, Path]]:
    """Do not advertise unapproved pairings when opt-in pins are enabled.

    Catalog discovery is metadata-only. Hash enforcement happens during
    the serialized inference step, immediately before model execution.
    """
    if approvals is None:
        return catalog
    return {
        category: {
            model: checkpoint for model, checkpoint in models.items()
            if (category, model) in approvals
        }
        for category, models in catalog.items()
        if any((category, model) in approvals for model in models)
    }
