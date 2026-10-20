"""Run private normal/defect demo images through trusted localhost checkpoints.

Checks real API routing/response/PNG overlay; records threshold-label agreement
only descriptively. Test images may come from MVTec's *original test directory*
and are NOT an independent accuracy study. No training or checkpoint writes.

Expected private sample layout:
    samples/bottle/normal.png
    samples/bottle/defective.png
    ... same for cable, carpet, metal_nut, capsule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from urllib import error, parse

from .dual_model_acceptance import image_sample, local_origin
from .local_api_smoke import multipart_png
from .multicategory_smoke import (
    CATEGORIES, MODELS, cleanup_inspection, request_json,
    validate_response, verify_catalog,
)

KINDS = ("normal", "defective")


def sample_pair(samples_dir: Path, category: str) -> dict:
    directory = samples_dir / category
    originals = {
        kind: directory / f"{kind}.png"
        for kind in KINDS
    }
    for kind, path in originals.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing private sample for {category}/{kind}")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Private sample too large: {category}/{kind}")
    samples = {kind: image_sample(path) for kind, path in originals.items()}
    if samples["normal"]["sha256"] == samples["defective"]["sha256"]:
        raise ValueError(f"Normal and defective samples share bytes: {category}")
    return samples


def verify_real_samples(origin: str, categories: list[str],
                        samples_dir: Path, key: str, *,
                        timeout: float = 300) -> dict:
    origin = local_origin(origin)
    if not isinstance(key, str) or len(key) < 16 or any(c in key for c in "\r\n"):
        raise ValueError("Missing or invalid private operator key")
    if not 1 <= timeout <= 600:
        raise ValueError("Timeout must be 1–600 seconds")
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Expected distinct, nonempty category selection")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("Unsupported product category")

    # Validate every input locally before sending the first HTTP prediction.
    samples = {category: sample_pair(samples_dir, category)
               for category in categories}
    verify_catalog(origin, categories)
    rows = []

    for category in categories:
        for model in MODELS:
            for kind in KINDS:
                sample = samples[category][kind]
                body, content_type = multipart_png(sample["png_bytes"])
                query = parse.urlencode({"category": category, "model": model})
                data = request_json(
                    origin + "/predict?" + query,
                    body=body, method="POST", timeout=timeout,
                    headers={
                        "Content-Type": content_type,
                        "X-IhsanVision-Key": key,
                    },
                )
                # The response may have added a persistent private history
                # row. Delete only the row returned for *this request*.
                inspection_id = data.get("inspection_id")
                try:
                    evidence = validate_response(
                        data, category=category, model=model,
                        dimensions=sample["dimensions"])
                finally:
                    if inspection_id is not None:
                        if not isinstance(inspection_id, str):
                            raise ValueError("Invalid returned history record identifier")
                        cleanup_inspection(origin, key, inspection_id)

                predicted = data.get("predicted_anomaly")
                expected = kind == "defective"
                agreement = (predicted == expected
                             if type(predicted) is bool else None)
                rows.append({
                    "category": category,
                    "model": model,
                    "sample": kind,
                    "predicted_anomaly": predicted,
                    "label_agrees_with_sample_kind": agreement,
                    "anomaly_score_raw": float(data["anomaly_score"]),
                    "latency_ms_including_possible_load": evidence["latency_ms"],
                    "overlay_png_valid": evidence["overlay_png_valid"],
                    "model_reused": evidence["model_reused"],
                })
                marker = ("MATCH" if agreement is True
                          else "MISMATCH" if agreement is False
                          else "UNKNOWN")
                print(f"API PASSED: {category}/{model}/{kind} "
                      f"(threshold label: {marker}; overlay valid)", flush=True)

    matched = sum(row["label_agrees_with_sample_kind"] is True for row in rows)
    mismatched = sum(row["label_agrees_with_sample_kind"] is False for row in rows)
    unknown = sum(row["label_agrees_with_sample_kind"] is None for row in rows)
    return {
        "status": "real_sample_api_integration_passed",
        "tested_requests": len(rows),
        "threshold_label_matches": matched,
        "threshold_label_mismatches": mismatched,
        "threshold_label_unknown": unknown,
        "rows": rows,
        "limitations": [
            "These manually selected samples may come from the original MVTec test folder; "
            "they are not an independent held-out accuracy experiment.",
            "A matching predicted label is not a validated sensitivity/specificity estimate. "
            "Mismatches require manual failure-case review.",
            "Model scores are raw, uncalibrated and not comparable across categories/models.",
            "Valid overlays do not establish correct defect localization.",
            "This exercises the HTTP API, not manual dashboard, batch/history UX or Docker.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES,
                        default=list(CATEGORIES))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--key-file", type=Path,
                        help="Private mode-0600 key file; otherwise use client key env")
    parser.add_argument("--output", type=Path,
                        help="Optional new private 0600 JSON report (never overwrites)")
    args = parser.parse_args(argv)
    if args.key_file:
        key_file = args.key_file.expanduser()
        if (not key_file.is_file() or key_file.is_symlink()
                or key_file.stat().st_mode & 0o077):
            parser.error("Key file must be regular, non-symlink and mode 0600")
        key = key_file.read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get("IHSANVISION_CLIENT_API_KEY", "")
    if args.output is not None:
        target = args.output.expanduser()
        if not target.is_absolute() or target.exists() or target.is_symlink():
            parser.error("Report output must be a new private absolute path")
    try:
        result = verify_real_samples(
            args.url, args.categories,
            args.samples_dir.expanduser(), key, timeout=args.timeout)
        if args.output is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, allow_nan=False)
                file.write("\n")
            print("Saved private API acceptance metadata:", target)
    except (OSError, ValueError, RuntimeError, error.URLError,
            json.JSONDecodeError) as exc:
        parser.exit(1, f"Real-sample localhost acceptance FAILED: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
