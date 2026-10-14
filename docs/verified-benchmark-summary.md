# Verified PaDiM–PatchCore benchmark summary

The read-only `scripts.summarize_comparison` command converts a **completed**
`ihsanvision.benchmark` experiment into a Markdown table suitable for review.
This is a reporting tool, **not** a training command. No model checkpoints are
deserialized, and no additional scores are estimated.

## After the real Kaggle comparison completes

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

**Current evidence status:** At the time this tool was authored, PatchCore training
was running in a Kaggle notebook. No completed PatchCore scores are asserted here.
