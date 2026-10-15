# 2026-10-15 — macOS browser verification of PaDiM/PatchCore selector

## Evidence and scope

**Evidence source:** Two user-provided Safari screenshots of the locally served IhsanVision.AI inspection dashboard on macOS, plus a user-provided Terminal command/response for `GET http://127.0.0.1:8003/models`. Both screenshots show the selected local file name `normal_bottle.png`, predicted-normal labels, raw scores, inference timing and rendered localization overlays. This file records **directly observed UI behavior**; it does not bundle source images, masks or binary model checkpoints.

The dashboard was served from commit `234e918` (merged PR #9). The Uvicorn server printed `Application startup complete` and `Uvicorn running on http://127.0.0.1:8003`. The user was running a local worktree and configured both model checkpoints from their checksum-verified macOS artifact archive.

Model discovery response:

~~~json
{
  "available_models": ["padim", "patchcore"],
  "default_model": "padim"
}
~~~

## Observed normal-image UI outcomes

| Item | PaDiM | PatchCore |
| --- | --- | --- |
| Browser selection | PaDiM | PatchCore |
| Uploaded name shown | `normal_bottle.png` | `normal_bottle.png` |
| Raw score displayed | `0.3637` | `0.1275` |
| UI label | Predicted normal | Predicted normal |
| UI latency | `611.68 ms` | `694.3 ms` |
| Relative anomaly localization | Overlay displayed | Overlay displayed |

**What is established:** Both registered model families are available to the same dashboard process, an image upload resulted in an inference response for each selected model, the reported model matched the selected one and the page rendered a relative heatmap. The `/models` response exposed only model names/default, not local checkpoint paths.

**What is not established:** The uploaded file's membership of the saved held-out test manifest has not been independently confirmed. One normal example per model does **not** establish accuracy, specificity, latency percentiles, segmentation quality or real-production behavior. The displayed raw scores are uncalibrated and **not directly comparable between PaDiM and PatchCore**. The two timings are one-off UI-reported prediction spans and should not be interpreted as a controlled speed benchmark.

## Additional Safari observation — PaDiM defective bottle (screenshot dated 2026-10-15)

A subsequent user-provided local Safari screenshot shows the same dashboard's **PaDiM** option selected with file `defective_bottle.png`. The observed values are:

| Field | Observed |
| --- | --- |
| Selected model | PaDiM |
| Uploaded filename shown | `defective_bottle.png` |
| Raw anomaly score | `0.8633` |
| Prediction | `Flagged anomaly` |
| UI-reported prediction span | `674.09 ms` |
| Overlay | Rendered; localized red highlighting near the visually damaged rim |

This establishes **one observed positive classification** on a user-labeled defective bottle in the browser. The screenshot does not establish that the sample is in the saved held-out test manifest; pixel-level localization has not been compared to a ground-truth mask. This is not an accuracy, recall, cold-start or p95 benchmark. The corresponding **PatchCore result on the exact same defective file is still pending**. The underlying MVTec image is not committed.

## Next acceptance test

Use the **same defective bottle image** in the browser for both model choices; record the predicted labels, raw scores, overlay behavior and the actual source image's membership of the saved held-out test manifest. Repeat a model switch to detect stale results or cache-routing issues. Do not reuse test samples for model selection or threshold tuning. Any future performance comparison requires a controlled warm/cold protocol and enough repeated trials.

Avoid committing MVTec AD test images or proprietary checkpoint binaries to the repository.
