# Portfolio failure-case audit — no invented predictions

For portfolio gate **P6**, a table of image/pixel AUROC/F1 is not enough. A
recruiter should also see where the models actually make wrong decisions:
**false positives** (normal rejected) and **false negatives** (defect missed).

`scripts.audit_error_cases` is a **read-only, lightweight** reporter for
genuinely collected per-image, independently labeled **held-out test**
predictions. It **does not** load PaDiM/PatchCore, fetch MVTec images, train,
infer labels, classify samples by filename, calibrate thresholds, or create
any example data itself. A dummy CSV used in CI must never be presented as
experimental evidence.

## Input: an owner-owned, PRIVATE CSV

After real evaluation, collect each model's actual final prediction per
sample in a private CSV with exactly these columns and order:

```text
category,model,sample_id,split,ground_truth,predicted_anomaly,anomaly_score
```

- `category`: `bottle`, `cable`, or `carpet` only for v1.0.
- `model`: `padim` or `patchcore`.
- `sample_id`: **opaque, consistent across paired models**, 4–64 characters
  of letters, digits, underscores or hyphens, starting with a letter/digit.
  Create and privately preserve a mapping to the original files; **never use
  full paths, actual filenames, sensitive item IDs or MVTec image bytes here**.
- `split`: literally `test`, and the owner must ensure the image really
  belongs to the independently held-out test split.
- `ground_truth`: `normal` or `anomaly` from the dataset/operator's
  independent ground truth, never inferred from the prediction.
- `predicted_anomaly`: `true` or `false` from actual checkpoint-backed
  model output.
- `anomaly_score`: finite **raw** float reported by that model; score units
  and decision thresholds differ across models.

**Every category must have the two models run on precisely the same opaque
image IDs and ground-truth labels**, including both normal and anomalous
samples. Missing runs, unfair paired samples, duplicates, malformed labels,
or unsupported splits all cause failure.

Run on the machine holding the real private CSV:

```bash
python -m scripts.audit_error_cases \
  --input /absolute/private/ihsanvision-test-predictions.csv \
  --categories bottle cable carpet \
  --output /absolute/private/portfolio-error-audit.md
```

Output is a **new** private Markdown file; pre-existing files are never
overwritten. The report includes TP/TN/FP/FN and precision/recall/specificity
per model/category and the first five opaque false-positive/false-negative IDs.
A zero count is reported as **none observed**, never “zero possible errors.”
The report cannot authenticate the input's content or calculate pixel AUROC;
the separately verified Anomalib [saved-run benchmark summary](verified-benchmark-summary.md)
provides the actual pixel/image AUROC and F1.

## Review examples with actual images

For two to six *actual observed* FP/FN cases, privately examine the original
input, trusted ground truth, and actual overlay. Record observed possible
causes (camera pose, lighting, subtle defect or thresholding) **only where
supported by the images**. An empty error list is not grounds to invent
mistakes. For GitHub, describe observations in words or use **only
rights-cleared original media**; MVTec dataset images and masks must not be
redistributed without permission. Never put the private CSV, sample mappings,
customer data or model weights into Git.

A report from this script alone does **not** satisfy P6; it needs independently
reviewed input provenance, the official saved-run scorecard, and an honest
qualitative assessment.
