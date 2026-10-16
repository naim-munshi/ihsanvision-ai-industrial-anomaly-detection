"""Compare two saved sequential localhost serving profiles without rerunning inference.

Validates shared image hash/dimensions, experiment setup, cache reuse, and
statistic shapes before emitting a truthful, read-only Markdown summary.

Example:
 python -m scripts.compare_serving_profiles \
   --padim "$HOME/ihsanvision-artifacts/padim_serving_profile_2026-10-16.json" \
   --patchcore "$HOME/ihsanvision-artifacts/patchcore_serving_profile_2026-10-16.json" \
   --output "$HOME/ihsanvision-artifacts/paired_serving_profile_2026-10-16.md"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIELDS = ("mean_ms", "p50_nearest_rank_ms", "p95_nearest_rank_ms")
GROUPS = ("client_round_trip", "server_reported_prediction")


def _number(value: object, label: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid numeric field: {label}")
    return float(value)


def _parse(path: Path, model: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{model} profile must be an object")
    if data.get("model") != model or data.get("status") != "sequential_profile_complete":
        raise ValueError(f"{model} profile has wrong model/status")
    if data.get("schema_version") != 1:
        raise ValueError(f"{model} profile schema mismatch")
    for key in ("warmup_requests", "measured_requests"):
        value = data.get(key)
        if type(value) is not int or value < (1 if key == "measured_requests" else 0):
            raise ValueError(f"{model} invalid {key}")
    if data.get("sample_type") not in ("synthetic", "user_supplied"):
        raise ValueError(f"{model} invalid sample type")
    digest = data.get("sample_sha256_of_transmitted_png")
    if (not isinstance(digest, str) or len(digest) != 64
            or not all(c in "0123456789abcdef" for c in digest)):
        raise ValueError(f"{model} invalid SHA-256")
    dims = data.get("input_image_dimensions")
    if (not isinstance(dims, list) or len(dims) != 2
            or any(type(x) is not int or x <= 0 for x in dims)):
        raise ValueError(f"{model} invalid dimensions")
    cache = data.get("model_cache_reuse_counts")
    if (not isinstance(cache, dict)
            or any(type(cache.get(k)) is not int or cache[k] < 0
                   for k in ("true", "false", "unknown"))
            or sum(cache[k] for k in ("true", "false", "unknown")) != data["measured_requests"]):
        raise ValueError(f"{model} invalid cache reuse counts")
    for group in GROUPS:
        metrics = data.get(group)
        if not isinstance(metrics, dict) or metrics.get("samples") != data["measured_requests"]:
            raise ValueError(f"{model} missing {group} measurements")
        for key in FIELDS:
            _number(metrics.get(key), f"{model}.{group}.{key}")
    return data


def compare(padim_path: Path, patchcore_path: Path) -> str:
    padim = _parse(padim_path, "padim")
    patchcore = _parse(patchcore_path, "patchcore")
    keys = ("sample_type", "sample_sha256_of_transmitted_png",
            "input_image_dimensions", "warmup_requests", "measured_requests")
    for key in keys:
        if padim[key] != patchcore[key]:
            raise ValueError(f"Cannot compare profiles with differing {key}")
    if not isinstance(padim.get("system"), dict) or not isinstance(patchcore.get("system"), dict):
        raise ValueError("Missing runtime provenance")
    for key in ("os", "machine", "python_version"):
        a, b = padim["system"].get(key), patchcore["system"].get(key)
        if not isinstance(a, str) or not a or a != b:
            raise ValueError(f"Cannot compare profiles with differing system.{key}")
    for model, data in (("PaDiM", padim), ("PatchCore", patchcore)):
        cache = data["model_cache_reuse_counts"]
        if cache["true"] != data["measured_requests"]:
            raise ValueError(f"{model} has cold/unknown measured predictions: not a warm-only comparison")

    num = padim["measured_requests"]
    warmup = padim["warmup_requests"]
    dims = padim["input_image_dimensions"]
    lines = [
        "# PaDiM vs PatchCore — local cached sequential HTTP profile",
        "",
        f"- Both saved profiles completed, **{num} measured requests per model** after "
        f"**{warmup} warmup requests**.",
        f"- Identical transmitted PNG SHA-256: `{padim['sample_sha256_of_transmitted_png']}`.",
        f"- Identical sample dimensions: **{dims[0]} × {dims[1]}**; type: "
        f"`{padim['sample_type']}`.",
        f"- Environment: {padim['system']['os']}, {padim['system']['machine']}, "
        f"Python {padim['system']['python_version']}. "
        "No independent hardware/accelerator audit was performed.",
        "- **100% reported cache reuse** for both measured sets.",
    ]
    for name, group in (("Client HTTP round-trip", GROUPS[0]),
                         ("Server-reported prediction span", GROUPS[1])):
        lines.extend([
            "",
            f"## {name} (ms)",
            "",
            "| Metric | PaDiM | PatchCore |",
            "| --- | ---: | ---: |",
        ])
        for label, field in (("Mean", "mean_ms"),
                             ("P50 (nearest rank)", "p50_nearest_rank_ms"),
                             ("P95 (nearest rank)", "p95_nearest_rank_ms")):
            a = _number(padim[group][field], field)
            b = _number(patchcore[group][field], field)
            lines.append(f"| {label} | {a:.3f} | {b:.3f} |")
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "Results are **observed single-client sequential cached-serving measurements**, "
        "not a concurrent throughput or production SLA claim. A comparison of 20 calls per "
        "model is noisy and sensitive to request ordering, system load and temperature.",
        "",
        "PaDiM may have a lower mean/p50 in this particular run while PatchCore has "
        "a lower observed p95; there is **no demonstrated generally faster or more "
        "stable model**. The script validates profile consistency, not original inference "
        "log authenticity, sample/test-split membership, score calibration or model accuracy.",
        "",
        "Further gates: repeated alternating controlled sessions, cold-start and memory "
        "profiling, finite concurrent admission probe, dataset split provenance, secure "
        "container testing and public-release review.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padim", type=Path, required=True)
    parser.add_argument("--patchcore", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New .md file; never overwrite")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        result = compare(args.padim.expanduser(), args.patchcore.expanduser())
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Paired serving profile validation failed: {exc}") from None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(result)
        print(f"Saved paired profile: {args.output}")
    print(result)


if __name__ == "__main__":
    main()
