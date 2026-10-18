# MVTec AD multi-category training: validate, plan, explicitly resume

The repository already includes normal-only PaDiM and PatchCore training for
**all 15 MVTec AD categories** and a benchmark comparing their **matching
per-category train/validation/test splits**. This new runner addresses the
operational problem of interrupted runs, partial private artifacts and limited
Kaggle/GPU sessions. **It does not supply dataset bytes, GPU compute or new
experiment results.**


**Interrupted-run guard:** Even an existing *empty* per-model run directory,
a directory containing only `engine/` or stray notes, or a symlinked run
directory now causes the **dry-run planner** to fail before it schedules
training. Direct training reserves a completely new run directory, so such
leftovers must be inspected or privately archived first; never delete a
checkpoint just to make the plan pass. A new run directory appearing between
dry-run and execution is also refused.


## Preflight before all GPU work

Mount or extract a properly licensed, operator-owned MVTec AD dataset locally
or into Kaggle. Never redistribute its images to this source repository. The
expected root contains `bottle/train/good`, `bottle/test/good`, and
`bottle/test/<defect>` (and corresponding folders for selected categories).
Start with a dry-run:

~~~bash
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable carpet
~~~

The tool prints a **read-only, path-scrubbed** JSON plan: category image counts,
saved/completed models, missing models and order of missing fits. **No new
checkpoints, split files or training jobs are created by this invocation.**
Supply `--categories` to keep experimental scope within available dataset
classes; omitting it explicitly checks all 15 categories before any training.

Planning refuses **any** partial checkpoint/report/split triple (for example,
a checkpoint without a final report), malformed prior metrics, inconsistent
seed, or disagreeing PaDiM/PatchCore split manifests. These artifacts are
left untouched. Investigate failures and independently preserve the current
experiment directory; do not delete partial work to force the trainer to run.

## Controlled execution — never automatic

After verifying the plan, explicitly request **at most one new job at first**:

~~~bash
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable capsule \
  --seed 42 --accelerator cpu --execute --max-jobs 1
~~~

On supported GPU environments, select `--accelerator cuda` or a compatible
accelerator only after confirming backend availability. Anomalib may download
pretrained feature-extractor weights for *new* training. Each completed job
writes the existing `model.ckpt`, `splits.json`, `report.json` artifacts.
The job is then **revalidated** against the expected saved-report contract.
Only missing models are fitted; a completed run is not retrained. The number of
new fits per invocation is strictly capped at 1–4 with `--max-jobs`.
Repeated invocation safely plans what is **still** missing.

When a model of a category already exists, its saved exact train/val/test
manifest is supplied as the reference for the other model and checked by the
training code **before fitting**. When both are completed, the planner
refuses any mismatch in corresponding split file paths. This minimizes
accidental unfair comparison; it does not replace independent evaluation.

## Scientific and deployment caveats

- The existing evaluation protocol holds validation images out of MVTec AD's
  **original test directory**, leaving the rest for final test: do not compare
  these scores with official full-test benchmark rankings.
- Models trained here are for **single category** inference; register only a
  trusted, verified result in the application's private multi-category model
  registry after reviewing splits, performance, heatmaps and checkpoint SHA.
- This script works with the classic **MVTec AD** directory layout, **not
  MVTec AD 2**. AD2 needs its own adapter and rights review.
- Planning validates run artifacts' structure and saved metrics, **not**
  cryptographic authenticity or Lightning checkpoint code safety.
- A dry-run is safe on any machine with the dataset paths; an actual
  `--execute` invocation requires a compatible Anomalib/PyTorch installation,
  compute budget, and operator monitoring. Do not schedule jobs silently or
  publish outputs without dataset/model rights review.

See also [training configuration](../ihsanvision/train.py),
[original two-model benchmark](../ihsanvision/benchmark.py) and
[trained category registry](multicategory-model-registry.md).
