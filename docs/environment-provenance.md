# Capturing Kaggle Experiment Environment

Use `scripts.environment_snapshot` to preserve the software/hardware/runtime
context next to a completed benchmark. **This script does not train models,
load checkpoints, re-run inference, or modify any saved experiment artifact.**

## Capture in Kaggle after training

Run from the source checkout after you have updated it to a revision containing
this utility. If a PatchCore training command is still running, **do not switch
branches or run a Git pull in that active checkout**; wait for the training
process to finish and preserve its artifacts first.

~~~python
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/ihsanvision-ai-industrial-anomaly-detection")
runs = Path("/kaggle/working/ihsanvision_runs")

subprocess.run([
    sys.executable, "-m", "scripts.environment_snapshot",
    "--repo-dir", str(repo),
    "--runs-dir", str(runs),
    "--output", str(runs / "environment.json"),
], cwd=repo, check=True)
~~~

The command records:
- Capture time (UTC), operating system, machine architecture and logical CPU count.
- Python, PyTorch, TorchVision, Anomalib, Lightning, NumPy, Imagecodecs, Timm,
  pandas and scikit-learn versions if installed.
- Whether CUDA or Apple MPS are available **at capture time**, with CUDA device
  name when available.
- Current Git revision/branch and whether tracked/untracked changes are present.
- Presence (not validity) of each model's checkpoint, report and split manifest.

**Important interpretation:** A snapshot taken *after* training shows the
environment **at capture time**; it cannot prove what hardware ran a historical
PaDiM training job. Treat model-specific `report.json` and original training
logs as the primary record of the training configuration. Kaggle may reset
`/kaggle/working` on a new session; keep this JSON alongside the comparison
files and checkpoints in private persistent storage.

No environment variables, tokens, service credentials, or file contents are
collected. Saving the output requires a **new filename**; existing files are
never overwritten. With no `--output`, the JSON is printed only.

This is useful in conjunction with the [verified benchmark summary](verified-benchmark-summary.md),
which validates completed metric and split artifacts before publishing a table.
