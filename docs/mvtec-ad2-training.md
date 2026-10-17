# MVTec AD 2 — Non-commercial, public-test-only training

IhsanVision.AI's **classic MVTec AD** training pipeline is **not** appropriate
for MVTec AD 2. AD 2 has an independent normal-only `validation/good` folder
and separate public/official private test sets. The AD2-specific training
entrypoint builds the Anomalib `MVTecAD2` datamodule with `test_type="public"`.
**No user dataset was downloaded and no real AD2 model has been trained by
this change.**

## Data rights

Official source: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>.
MVTec states the dataset is for **non-commercial use only**; Anomalib
documents its license as **CC BY-NC-SA 4.0**. Check the current official
license and any applicable distribution conditions before using the data.
Neither dataset images nor trained checkpoints are bundled with this
repository. A commercial factory deployment would require its own
appropriately licensed data and legal review.

Anomalib's documented MVTec AD 2 class and source:
<https://anomalib.readthedocs.io/en/lib-v2.4.0/markdown/guides/reference/data/datamodules/image/mvtecad2.html>.
This implementation was designed against that public API and the current
open-source dataset parser, **not validated against user hardware**.

## Dataset layout

The operator must obtain the dataset from its authorized source and
place it locally outside Git. For an example category, the required folders
are:

~~~text
/private/MVTec_AD_2/sheet_metal/
  train/good/*.png
  validation/good/*.png
  test_public/good/*.png
  test_public/bad/*.png
  test_public/ground_truth/bad/<anomaly_filename_stem>_mask.png
~~~

The optional official `test_private` and `test_private_mixed` directories
are *not* part of local metric evaluation. Their ground truth is not
publicly available; testing on them cannot produce a valid local AUROC/F1.

All eight canonical AD2 category labels supported by the Anomalib
source: `can`, `fabric`, `fruit_jelly`, `rice`,
`sheet_metal`, `vial`, `wallplugs`, `walnuts`. Note that
**`wallplugs` (without underscore)** is the actual directory spelling;
the earlier experimental `wall_plugs` registry label was inaccurate.

## Dry-run — no training, downloads, model loading or file creation

~~~bash
python -m ihsanvision.train_ad2 \
  --dataset-root /private/MVTec_AD_2 \
  --output-dir /private/ihsanvision-ad2-runs \
  --category sheet_metal \
  --model padim
~~~

The preflight requires **nonempty normal train and normal validation** images,
both public normal and public anomalous examples, and one matching mask
for every public anomaly. It rejects symlinks to image files, invalid
categories, empty masks, missing splits and a previous/partial run directory.
The dry-run prints only counts. It does not scan or publish private-test
filenames, read image pixels or touch the requested output directory.

## Explicit non-commercial research fitting

Only after verifying license compatibility and selecting a locally available
dataset and compatible Anomalib/PyTorch environment:

~~~bash
python -m ihsanvision.train_ad2 \
  --dataset-root /private/MVTec_AD_2 \
  --output-dir /private/ihsanvision-ad2-runs \
  --category sheet_metal --model padim \
  --accelerator cpu \
  --execute --acknowledge-noncommercial
~~~

The `--acknowledge-noncommercial` flag is a conscious operator action;
**it is not a grant of commercial rights**. `--execute` alone refuses to
train. A fresh private run folder is created using
`ad2-<model>-<category>-seed<seed>`; existing folders (including
partial experiments) are refused without overwriting evidence.
The trainer uses `validation/good` separately and **only
`test_public` for measured local test metrics**, saving
`splits.json`, `model.ckpt`, `report.json` under the private
run directory.

Saved manifest paths are validated before fitting to ensure train
images come from the normal training folder, val images come from
the dedicated normal validation folder, and evaluated test images
come exclusively from `test_public`. The private test directories
cannot be passed into the experiment split.

Be aware Anomalib might download **backbone pretrained weights** on
explicit training execution if they are not already cached. The
AD2 *dataset* itself is not downloaded by this script because the
local root/category must already be present. Operator should review
runtime versions and artifact provenance.

## Evaluation caveats

- Local metrics refer only to the public test subset, never the official
  unseen private test leaderboard. Validation is **normal-only**, so
  thresholds and some supervised metrics may be undefined or unsuitable
  without additional independent calibration; inspect each report
  before relying on any F1 score.
- PaDiM and PatchCore scores cannot be compared directly across
  different models or categories. A new AD2 category must have its own
  verified checkpoint and metrics before registering it in the server.
- Dataset structure checks and fake-run CI tests do **not** establish
  scientific reproducibility, measured quality, hardware compatibility,
  correct masks in pixels, or commercial rights.
- Actual AD2 model training/evaluation, production authorization,
  domain-shift and lighting scenario evaluation remain open.
