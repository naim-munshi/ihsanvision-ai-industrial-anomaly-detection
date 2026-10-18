"""Produce a verified, read-only Markdown summary of a completed two-model benchmark.

Usage:
    python -m scripts.summarize_comparison --runs-dir /kaggle/working/ihsanvision_runs
    python -m scripts.summarize_comparison --runs-dir runs --output runs/summary.md

Nothing is trained or deserialized. A summary is refused unless saved model
reports, metrics, and exact train/val/test manifests agree with comparison.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ihsanvision.benchmark import read_completed_run, require_matching_splits, run_folder
from ihsanvision.config import CATEGORIES


METRICS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
MODELS = ("padim", "patchcore")


def summarize(runs_dir: Path) -> str:
    """Validate saved benchmark evidence before formatting a Markdown report."""
    runs_dir = Path(runs_dir)
    comparison = json.loads((runs_dir / "comparison.json").read_text(encoding="utf-8"))
    if not isinstance(comparison, dict):
        raise ValueError("comparison.json must contain a JSON object")
    if comparison.get("split_equivalence") != "verified_path_by_path":
        raise ValueError("Benchmark lacks verified split equivalence")
    seed = comparison.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Benchmark seed missing or invalid")
    counts_by_category = comparison.get("split_counts")
    sources_by_category = comparison.get("model_sources")
    rows = comparison.get("rows")
    if not isinstance(counts_by_category, dict) or not counts_by_category:
        raise ValueError("Benchmark has no category split counts")
    if not isinstance(sources_by_category, dict) or not isinstance(rows, list):
        raise ValueError("Benchmark provenance or result rows missing")
    if set(sources_by_category) != set(counts_by_category):
        raise ValueError("Benchmark model-source categories disagree with split counts")

    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid benchmark result row")
        category, model = row.get("category"), row.get("model")
        if category not in counts_by_category or model not in MODELS:
            raise ValueError("Unexpected category or model in benchmark results")
        key = (category, model)
        if key in indexed:
            raise ValueError(f"Duplicate benchmark result: {category}/{model}")
        indexed[key] = row

    if len(indexed) != len(counts_by_category) * len(MODELS):
        raise ValueError("Both PaDiM and PatchCore result rows are required per category")

    lines = [
        "# PaDiM vs PatchCore — Verified Benchmark Summary",
        "",
        f"Seed: `{seed}`. Split equivalence: **verified path by path**.",
        "",
        "| Category | Model | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Training (s) | Origin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for category in sorted(counts_by_category):
        sources = sources_by_category[category]
        if not isinstance(sources, dict) or set(sources) != set(MODELS):
            raise ValueError(f"Incomplete model-source provenance for {category}")
        if any(source not in ("trained", "reused") for source in sources.values()):
            raise ValueError(f"Invalid model-source provenance for {category}")

        manifests = {}
        for model in MODELS:
            folder = run_folder(runs_dir, model, category, seed)
            report, manifest = read_completed_run(folder, model, category, seed)
            expected = {
                "category": category,
                "model": model,
                "train_seconds": report["train_seconds"],
                **report["evaluation"]["metrics"],
            }
            if indexed[(category, model)] != expected:
                raise ValueError(f"Comparison row differs from saved report: {category}/{model}")
            manifests[model] = manifest
            metrics = report["evaluation"]["metrics"]
            lines.append(
                f"| {category} | {model} | "
                + " | ".join(f"{metrics[name]:.4f}" for name in METRICS)
                + f" | {report['train_seconds']:.1f} | {sources[model]} |"
            )
        observed_counts = require_matching_splits(category, manifests)
        if counts_by_category[category] != observed_counts:
            raise ValueError(f"Comparison split counts differ from saved runs: {category}")

    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- The two model runs use identical recorded train, validation and test **paths**; "
        "this does not establish matching dataset file hashes.",
        "- Validation was held out from the original MVTec AD test directory. "
        "The resulting test metrics are **not** directly comparable with full-test leaderboard scores.",
        "- Training durations are recorded for provenance, **not** a controlled speed comparison: "
        "reused runs may have been measured in different sessions, hardware or environments.",
        "- AUROC is threshold-independent; F1 depends on the evaluation threshold. "
        "Neither a raw anomaly score nor a heatmap is a calibrated defect probability.",
        "- This report is generated from saved artifacts without loading model checkpoints. "
        "Successful inference, production latency and broader category performance are not implied.",
        "",
    ])
    return "\n".join(lines)


def summarize_saved_runs(runs_dir: Path, categories: list[str], seed: int) -> str:
    """Verified offline comparison from saved run triples, without comparison.json.

    The resumable training runner produces individual completed model runs but
    does not create a fresh aggregate comparison.json. This reporting path
    therefore reconstructs a *read-only* portfolio table directly from the
    six private model/report/split triples after the requested runs actually
    finish. It does not invoke Anomalib, deserialize a checkpoint, or train.
    """
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Require nonempty, unique portfolio categories")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("Unsupported MVTec AD category")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    lines = [
        "# PaDiM vs PatchCore — Validated Saved-Run Portfolio Comparison",
        "",
        f"Recorded seed: `{seed}`; all metrics originate from the saved per-run reports.",
        "",
        "| Category | Model | Train | Val | Test | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Train (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in categories:
        manifests = {}
        category_rows = []
        for model in MODELS:
            folder = run_folder(Path(runs_dir), model, category, seed)
            report, manifest = read_completed_run(folder, model, category, seed)
            manifests[model] = manifest
            category_rows.append((model, report))
        counts = require_matching_splits(category, manifests)
        for model, report in category_rows:
            metrics = report["evaluation"]["metrics"]
            lines.append(
                f"| {category} | {model} | {counts['train']} | {counts['val']} | "
                f"{counts['test']} | "
                + " | ".join(f"{metrics[name]:.4f}" for name in METRICS)
                + f" | {report['train_seconds']:.1f} |"
            )
    lines.extend([
        "",
        "## Evidence scope and limitations",
        "",
        "- All requested category/model saved checkpoint, report and split triples "
        "were present and structurally validated **without model deserialization**.",
        "- For each category, PaDiM and PatchCore train/validation/test **paths** match "
        "exactly. Underlying image-byte identity was not independently verified.",
        "- Validation examples were taken from the original MVTec AD test folder; "
        "these results are not directly comparable to official full-test leaderboard scores.",
        "- Training times come from saved runs and are not a controlled hardware comparison.",
        "- AUROC and F1 reflect one recorded seed and split; do not infer industrial "
        "generalization, calibrated scores, or per-pixel segmentation quality.",
        "- Report contents and checkpoint existence are audited, not retrained or "
        "cryptographically proven to come from an independent run.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument(
        "--from-saved-runs", action="store_true",
        help="Build an offline multi-category comparison from completed model/report/split trios",
    )
    parser.add_argument(
        "--categories", nargs="+", choices=CATEGORIES,
        default=["bottle", "cable", "carpet"],
        help="Ordered portfolio categories (used with --from-saved-runs)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Recorded experiment seed (used with --from-saved-runs)")
    parser.add_argument("--output", type=Path, help="Optional new Markdown file; existing files are never overwritten")
    args = parser.parse_args(argv)
    summary = (summarize_saved_runs(args.runs_dir, args.categories, args.seed)
               if args.from_saved_runs else summarize(args.runs_dir))
    if args.output is None:
        print(summary, end="")
    else:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(summary)
        print(f"Saved verified benchmark summary: {args.output}")


if __name__ == "__main__":
    main()
