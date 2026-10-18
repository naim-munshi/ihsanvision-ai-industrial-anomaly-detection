# Verified PaDiM–PatchCore benchmark summary

The read-only `scripts.summarize_comparison` command converts a **completed**
`ihsanvision.benchmark` experiment into a Markdown table suitable for review.
This is a reporting tool, **not** a training command. No model checkpoints are
deserialized, and no additional scores are estimated.

## Verify the completed Bottle comparison or future cable/carpet runs

Run in a notebook code cell (do **not** paste bare shell commands into Python):

~~~python
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/ihsanvision-ai-industrial-anomaly-detection")
runs = Path("/kaggle/working/ihsanvision_runs")
subprocess.run([
    sys.executable, "-m", "scripts.summarize_comparison",
    "--runs-dir", str(runs),
    "--output", str(runs / "benchmark_summary.md"),
], cwd=repo, check=True)
~~~

This command requires a checkout **containing the report script**; a Kaggle
clone made before the feature is merged will not have it until updated.
The output path must not already exist. Delete/replace neither the original
comparison artifacts nor a prior summary without manual inspection.

For an output-only preview (no file changes), omit `--output`:

~~~bash
python -m scripts.summarize_comparison --runs-dir /kaggle/working/ihsanvision_runs
~~~

## Portfolio 3-category saved runs (without new training)

The normal `scripts.resume_mvtec_training` pipeline writes **individual**
completed `model.ckpt`, `report.json`, and `splits.json` triples. It does
**not** automatically regenerate a multi-category `comparison.json`. When
Bottle, Cable and Carpet have genuinely completed their two model runs and
all six run directories are under the same private `--runs-dir`, generate a
read-only portfolio table directly from **those saved reports**:

```bash
python -m scripts.summarize_comparison \
  --runs-dir /absolute/private/ihsanvision_runs \
  --from-saved-runs --categories bottle cable carpet --seed 42
```

To save a **new** private Markdown report, append
`--output /absolute/private/portfolio-three-category.md`; existing files
are not overwritten. The command checks all six checkpoint/report/split
triples, validates metric ranges, checks each pair's exact train/val/test
paths and prints actual reported image/pixel AUROC/F1 and split counts.
It **does not deserialize** checkpoints or estimate missing scores. Missing
Cable/Carpet experiments make the command **fail**, not insert blanks or
fake performance. For these saved-run tables no aggregate
`comparison.json` is required; use the default comparison.json path below
when checking an already created `ihsanvision.benchmark` aggregate.

## Validation contract

The generator refuses to report results unless all of these hold:

- `comparison.json` exists, declares seed and `verified_path_by_path` split equivalence,
  and contains exactly one PaDiM and one PatchCore result per category.
- Each model has nonempty `model.ckpt`, `report.json`, and `splits.json`.
- Saved reports identify the correct model, category and seed, contain valid
  core AUROC/F1 metrics and nonnegative elapsed training time.
- Individual comparison rows exactly agree with the underlying saved reports.
- Both split manifests are internally disjoint and have matching train, validation
  and test image **paths**, with counts equal to the reported counts.
- Provenance records which models were `trained` or `reused`.

The output always cautions that the experimental validation holdout differs
from MVTec's published full-test protocol; image/path agreement does **not**
prove dataset-content hash equivalence; and training-time values should not be
compared as controlled performance measurements across different hardware or
sessions.

## Saving evidence

After a successful comparison, preserve these **outside Git**:

- `comparison.json`, `comparison.csv` and optional `benchmark_summary.md`.
- Both models' `report.json` and `splits.json` and their trusted `model.ckpt`.
- Hardware/runtime information, dependency versions, dataset revision and
  qualitative smoke-test output, when available.

Do not upload checkpoint binaries or MVTec images to a public GitHub repository.
The summary does not demonstrate operational reliability, inference speed or
successful deployment. Record those with separate verified experiments.

**Current evidence status:** Bottle PaDiM and PatchCore **have both completed real Kaggle runs**; documented observed metrics and a Mac checksum-verified backup are in [the PatchCore evidence report](../reports/2026-10-15-kaggle-patchcore-bottle.md). An owner-side successful comparison summary audit has been reported, but full independent provenance and new cable/carpet runs are not claimed. Use this verifier on each preserved original artifact set rather than hand-entering or inventing metrics.
