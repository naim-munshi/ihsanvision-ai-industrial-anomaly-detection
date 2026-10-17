"""Trusted server-side category/model checkpoint registry.

Opt-in via IHSANVISION_MODEL_REGISTRY=/absolute/private/model-registry.json.
Never allow clients to provide checkpoint paths. A registry *lists available
weights*; it does not claim the other MVTec categories have been trained.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import CATEGORIES, MODELS

# AD2 classes are supported as registry labels only when the operator
# independently trains and evaluates the corresponding weights.
# Actual dataset directory names from Anomalib's MVTecAD2Dataset.
# The formerly documented "wall_plugs" spelling is not a dataset category.
AD2_CATEGORIES = (
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
)
SUPPORTED_CATEGORIES = CATEGORIES + AD2_CATEGORIES


def configured_registry() -> dict[str, dict[str, Path]]:
    """Read a private, trusted JSON registry; fail closed if misconfigured."""
    raw = os.getenv("IHSANVISION_MODEL_REGISTRY", "").strip()
    if not raw:
        return {}
    registry_path = Path(raw).expanduser()
    if not registry_path.is_absolute():
        raise ValueError("IHSANVISION_MODEL_REGISTRY must be an absolute local path")
    try:
        source = registry_path.read_text(encoding="utf-8")
        if len(source) > 256_000:
            raise ValueError("Model registry JSON is too large")
        data = json.loads(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read valid private model registry JSON") from exc

    if not isinstance(data, dict) or set(data) != {"version", "models"}:
        raise ValueError("Registry must contain only version and models")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported model registry version")
    if not isinstance(data["models"], list) or len(data["models"]) > 64:
        raise ValueError("Registry models must be a list of at most 64 entries")
    registry: dict[str, dict[str, Path]] = {}
    for entry in data["models"]:
        if not isinstance(entry, dict) or set(entry) != {"category", "model", "checkpoint"}:
            raise ValueError("Every registry entry requires category, model and checkpoint")
        category, model, raw_path = (
            entry["category"], entry["model"], entry["checkpoint"]
        )
        if category not in SUPPORTED_CATEGORIES or model not in MODELS:
            raise ValueError("Unsupported category/model in local registry")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Checkpoint must be a non-empty absolute path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or path.suffix.lower() != ".ckpt":
            raise ValueError("Registry checkpoints must have absolute .ckpt paths")
        if category in registry and model in registry[category]:
            raise ValueError("Duplicate category/model checkpoint in model registry")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Trusted registry checkpoint missing or empty: {category}/{model}")
        registry.setdefault(category, {})[model] = path.resolve()
    return registry


def model_catalog(legacy_bottle: dict[str, Path]) -> dict[str, dict[str, Path]]:
    """Preserve existing bottle env configuration, extend with registered weights."""
    catalog = {"bottle": dict(legacy_bottle)} if legacy_bottle else {}
    for category, models in configured_registry().items():
        catalog.setdefault(category, {}).update(models)
    return catalog


def public_catalog(catalog: dict[str, dict[str, Path]], preferred_model: str) -> dict:
    """Serialize only category and model identifiers, never checkpoint paths."""
    by_category = {
        category: [model for model in MODELS if model in catalog[category]]
        for category in SUPPORTED_CATEGORIES if catalog.get(category)
    }
    default_category = "bottle" if "bottle" in by_category else next(iter(by_category), None)
    available = by_category.get(default_category, [])
    default_model = preferred_model if preferred_model in available else (available[0] if available else None)
    return {
        "available_models": available,
        "default_model": default_model,
        "available_categories": list(by_category),
        "models_by_category": by_category,
        "default_category": default_category,
    }
