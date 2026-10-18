"""Read-only, no-model-runtime error-case audit for real annotated test predictions.

Input is a PRIVATE, manually audited CSV of per-image results. The utility
never fabricates predictions, loads checkpoint binaries or reads image bytes.
It prints aggregate performance and a limited list of anonymous case IDs.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from io import StringIO
import math
import os
from pathlib import Path
import re

from ihsanvision.config import CATEGORIES, MODELS

COLUMNS = ("category", "model", "sample_id", "split", "ground_truth",
           "predicted_anomaly", "anomaly_score")
VALID_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}\Z")
MAX_CSV_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 25000


def read_cases(source: Path, *, categories: tuple[str, ...] = (
    "bottle", "cable", "carpet",
)) -> dict[tuple[str, str], dict[str, tuple[bool, bool, float]]]:
    """Require complete paired per-image test records and reject ambiguity."""
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Require distinct requested categories")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("Unsupported category")
    source = Path(source).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError("Input must be an existing absolute private regular CSV file")
    if source.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Private input exceeds the maximum size")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("CSV column order/schema differs from documented input")
        observations: dict[tuple[str, str], dict[str, tuple[bool, bool, float]]] = (
            defaultdict(dict)
        )
        count = 0
        for row in reader:
            count += 1
            if count > MAX_RECORDS:
                raise ValueError("CSV record limit exceeded")
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed or extended CSV row")
            category, model = row["category"], row["model"]
            if category not in categories or model not in MODELS:
                raise ValueError("Unexpected portfolio category or model")
            identifier = row["sample_id"]
            if not VALID_ID.fullmatch(identifier):
                raise ValueError("sample_id must be opaque and filename/path-free")
            if row["split"] != "test":
                raise ValueError("Only independently labeled held-out test rows are accepted")
            if row["ground_truth"] not in ("normal", "anomaly"):
                raise ValueError("ground_truth must be normal/anomaly")
            if row["predicted_anomaly"] not in ("true", "false"):
                raise ValueError("predicted_anomaly must be true/false")
            try:
                score = float(row["anomaly_score"])
            except ValueError as exc:
                raise ValueError("anomaly_score must be numeric") from exc
            if not math.isfinite(score):
                raise ValueError("anomaly_score must be finite")
            key = (category, model)
            if identifier in observations[key]:
                raise ValueError("Duplicate image/model observation")
            observations[key][identifier] = (
                row["ground_truth"] == "anomaly",
                row["predicted_anomaly"] == "true",
                score,
            )
    if count == 0:
        raise ValueError("No test cases supplied")
    for category in categories:
        left = observations.get((category, "padim"))
        right = observations.get((category, "patchcore"))
        if not left or not right:
            raise ValueError(f"Both paired models required for {category}")
        if left.keys() != right.keys():
            raise ValueError(f"Unfair sample IDs in {category} PaDiM/PatchCore predictions")
        if any(left[i][0] != right[i][0] for i in left):
            raise ValueError(f"Inconsistent ground-truth labels in {category}")
        if not any(case[0] for case in left.values()) or not any(
            not case[0] for case in left.values()
        ):
            raise ValueError(f"{category} needs both normal and anomalous test cases")
    return observations


def summarize_cases(
    observations: dict[tuple[str, str], dict[str, tuple[bool, bool, float]]],
    categories: tuple[str, ...] = ("bottle", "cable", "carpet"),
) -> str:
    """Aggregate actual supplied labels, never recalculate benchmark AUROC."""
    lines = [
        "# Portfolio test-set classification error audit",
        "",
        "**Source:** Operator-supplied, manually labeled per-image CSV. This "
        "utility did not check image bytes, checkpoint identity, source split "
        "provenance or the reliability of the labels.",
        "",
        "| Category | Model | N | TP | TN | FP | FN | Precision | Recall | Specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    mistakes: list[tuple[str, str, list[str], list[str]]] = []
    for category in categories:
        for model in MODELS:
            cases = observations[(category, model)]
            tp = sum(actual and predicted for actual, predicted, _ in cases.values())
            tn = sum(not actual and not predicted for actual, predicted, _ in cases.values())
            fp = sum(not actual and predicted for actual, predicted, _ in cases.values())
            fn = sum(actual and not predicted for actual, predicted, _ in cases.values())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            specificity = tn / (tn + fp) if tn + fp else None

            def pct(value: float | None) -> str:
                return f"{value:.3f}" if value is not None else "N/A"

            lines.append(
                f"| {category} | {model} | {len(cases)} | {tp} | {tn} | {fp} | "
                f"{fn} | {pct(precision)} | {pct(recall)} | {pct(specificity)} |"
            )
            false_positives = sorted(
                i for i, (actual, predicted, _) in cases.items()
                if not actual and predicted
            )
            false_negatives = sorted(
                i for i, (actual, predicted, _) in cases.items()
                if actual and not predicted
            )
            mistakes.append((category, model, false_positives, false_negatives))

    lines.extend(["", "## Observed error cases (opaque sample IDs)", ""])
    for category, model, fp, fn in mistakes:
        lines.append(
            f"- **{category} / {model}:** "
            f"{len(fp)} false positive(s), {len(fn)} false negative(s). "
            f"First five FP IDs: {', '.join(fp[:5]) or 'none observed'}; "
            f"first five FN IDs: {', '.join(fn[:5]) or 'none observed'}."
        )
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- A zero observed error count means **none were observed in the supplied "
        "samples**, not that the model is flawless. An independently reviewed "
        "overlay and ground-truth audit are still required.",
        "- This report calculates confusion statistics only; image/pixel AUROC "
        "and F1 from Anomalib belong in the separate verified benchmark summary.",
        "- Thresholding was performed upstream by the trained model, not "
        "calibrated here. Raw anomaly scores must not be compared across models.",
        "- IDs must be non-sensitive randomized references; do not paste real "
        "dataset paths, filenames, customer information or private case images.",
        "- The per-image test predictions and original model/dataset evidence "
        "must remain privately preserved outside GitHub.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES,
                        default=["bottle", "cable", "carpet"])
    parser.add_argument("--output", type=Path,
                        help="New private Markdown destination; never overwrite")
    args = parser.parse_args(argv)
    observations = read_cases(args.input, categories=tuple(args.categories))
    report = summarize_cases(observations, tuple(args.categories))
    if args.output is None:
        print(report, end="")
    else:
        destination = args.output.expanduser()
        if not destination.is_absolute() or destination.is_symlink():
            parser.error("Output must be an absolute private non-symlink path")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report)
        print("Saved private, per-image error analysis without image data.")


if __name__ == "__main__":
    main()
