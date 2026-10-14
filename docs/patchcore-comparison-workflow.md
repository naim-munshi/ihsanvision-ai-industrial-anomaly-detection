# Incremental PatchCore benchmark — reuse the existing PaDiM experiment

## Why this workflow exists

The original PaDiM/bottle run is completed. Repeating it just to add PatchCore would waste compute, and a casual new split can invalidate comparison. Use `src.benchmark --reuse-existing` to **reuse the previous PaDiM checkpoint/report/split manifest and train only PatchCore**, provided that its run artifacts are available in Kaggle.

**Current status:** The orchestration and offline tests are implemented. A real PatchCore training job has **not** yet been run, so there are **no PatchCore metrics to report**. Do not merge into the released branch on that basis.

## Prerequisites

- A Kaggle notebook with the same MVTec AD dataset attached.
- `feature/portfolio-production-pipeline` checkout updated to the latest commit, with `requirements.txt` installed.
- Saved PaDiM experiment artifacts `model.ckpt`, `report.json`, and `splits.json`, originally produced by `src.train --model padim --seed 42`.
- Dataset mounted at the **same resolved paths** used to generate the reference manifest. A Kaggle path change can cause an intentional safety stop even if image contents are identical.
- Compatible Anomalib 2.5.0 and PyTorch environment; PatchCore default backbone may require a pretrained-weight download during **training**.

### Step 1 — Find dataset and preserved PaDiM experiment

In Kaggle, within the repo directory:

~~~bash
python -m scripts.kaggle_preflight
~~~

Check whether `/kaggle/working/ihsanvision_runs/padim-bottle-seed42` still contains the three saved artifacts. Kaggle `/working` may be reset after a session, so **do not assume previous artifacts persist**.

If artifacts are already in the expected directory, skip Step 2.

### Step 2 — Restore PaDiM artifacts when the previous Kaggle session is gone

The user previously saved the checkpoint outside Git, on a Mac under:

~~~text
~/ihsanvision-artifacts/padim-bottle-seed42/
  model.ckpt
  report.json
  splits.json
~~~

If the Kaggle run directory is missing, place those **three files** into a private user-controlled Kaggle dataset (or notebook input). Its mount location is account-dependent: find the folder containing the three files, rather than assuming a guessed dataset slug.

Within the repo directory in Kaggle, stage them non-destructively:

~~~bash
python -m scripts.stage_reference_artifacts \
  --source-dir /kaggle/input/YOUR-PRIVATE-DATASET \
  --output-dir /kaggle/working/ihsanvision_runs \
  --category bottle --seed 42
~~~

Replace `YOUR-PRIVATE-DATASET` with the actual attached folder containing **model.ckpt, report.json and splits.json**. If they are nested inside `padim-bottle-seed42`, append that folder to `--source-dir`.

The staging helper validates the report, matching seed/category, metric fields, disjoint split counts and checkpoint presence, copies only these three files, and **refuses to overwrite any existing PaDiM artifact directory**. It never unpickles the checkpoint, trains a model or connects to external services. Only use checkpoints you trust. Keep the private input private; do not commit weights or MVTec images to Git.

### Step 3 — Run only missing PatchCore training (the first new real experiment)

~~~bash
python -m src.benchmark \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --categories bottle \
  --seed 42 \
  --output-dir /kaggle/working/ihsanvision_runs \
  --accelerator cpu \
  --reuse-existing
~~~

**Replace the dataset root with the path shown by `kaggle_preflight`.** The example uses CPU because the historical PaDiM/bottle run used CPU; if GPU is used instead, document the hardware difference and do not imply directly comparable training timings.

This command:

1. **Requires** complete existing PaDiM artifacts. If missing, it aborts before starting any training rather than retraining PaDiM.
2. Checks each saved PaDiM report, checkpoint existence, seed, category, split counts and core metric sanity.
3. If PatchCore has already completed in the same output folder, reuses its result; a **partial** PatchCore run is refused, not overwritten.
4. Otherwise initializes the Anomalib dataset and checks that the **actual train/validation/test image paths are identical to the saved PaDiM manifest before starting PatchCore fitting**. If not, it aborts before fitting.
5. Trains and evaluates PatchCore, then requires exact PaDiM/PatchCore train/validation/test path equivalence before writing summary files.

After a completed compatible run, inspect:

~~~text
/kaggle/working/ihsanvision_runs/
  padim-bottle-seed42/              # preserved; never retrained
    model.ckpt
    report.json
    splits.json
  patchcore-bottle-seed42/          # newly trained if absent
    model.ckpt
    report.json
    splits.json
  comparison.csv
  comparison.json
~~~

`comparison.json` includes **`model_sources`** such as `{"bottle":{"padim":"reused","patchcore":"trained"}}` and `split_equivalence: "verified_path_by_path"`. Those provenance fields make the comparison auditable.

### Step 4 — Verify PatchCore's saved checkpoint

A real held-out defective-image smoke test is optional but strongly recommended after training:

~~~bash
python -m scripts.kaggle_smoke_test \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --output-dir /kaggle/working/ihsanvision_runs \
  --model patchcore --category bottle --seed 42 --accelerator cpu
~~~

This script now samples only a defective image from the **saved held-out test split**, never the validation holdout. The displayed anomaly score and overlay are qualitative smoke evidence, not standalone accuracy proof.

### Step 5 — Preserve the new evidence

Download/store `comparison.json`, `comparison.csv`, the two model-specific `report.json` and `splits.json` files, and a trusted PatchCore `model.ckpt` outside the repository. Preserve environment versions, hardware, and any network/runtime limitations. **Do not push the checkpoint or MVTec AD image data to GitHub.**

## Benchmark interpretation and risks

- **Valid comparison:** same dataset/category, same exact train/validation/test paths, recorded model and seed. The held-out set still differs from the published full-test MVTec protocol.
- **Training time is not necessarily comparable** when reused PaDiM and newly trained PatchCore ran in different Kaggle sessions/hardware.
- PatchCore's default `wide_resnet50_2` backbone and coreset search can require more compute/memory than PaDiM; the repository has not measured it yet.
- No fabricated PatchCore metrics, latency values, images, or performance claims should be published.
- A missing or corrupt reference run stops the workflow. Inspect artifact integrity and source paths; **do not delete or overwrite anything automatically**.
- The comparator verifies dataset *paths*, not file-content hashes. Dataset version/provenance should still be recorded.
