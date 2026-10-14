# Browser UI validation — normal and defective bottle examples

**Observation:** 2026-10-14 JST. Evidence: two screenshots supplied by the project user from the local IhsanVision.AI browser dashboard. Screenshots themselves were **not committed to GitHub**. This report records visible behavior only; it is not a new dataset-wide evaluation.

## Purpose and environment

Validate the implemented web demo in a real browser with two bottle images, one labeled normal and one labeled defective. The interface visibly reported `Model configured`, rendered a selected image, and displayed the result of a PaDiM inference including a prediction label and relative anomaly-localization overlay. The server was run locally on macOS with the project's saved trained PaDiM/bottle checkpoint.

| UI field | Normal example | Defective example |
| --- | --- | --- |
| Uploaded name shown | `normal_bottle.png` | `defective_bottle.png` |
| Raw anomaly score | **0.3637** | **0.8633** |
| Displayed classification | `Predicted normal` | `Flagged anomaly` |
| Model | `padim` | `padim` |
| Displayed prediction latency | **853.49 ms** | **409.43 ms** |
| Localization preview | Rendered | Rendered |
| Original image preview | Rendered | Rendered |

The normal screenshot shows a bottle viewed from above and a diffuse reddish relative overlay. The defective screenshot shows a visibly damaged/chipped region on the bottle rim and a localized reddish response overlapping parts of the damaged region.

## What passed

1. The responsive page loaded and displayed the configured-model status.
2. Each uploaded bottle image displayed in the inspection panel.
3. The `Analyze image` workflow produced a model result for both examples.
4. The UI displayed a numeric score, model family, predicted label, model latency, and a corresponding localization image.
5. The example labeled normal was predicted normal; the example labeled defective was flagged as anomalous.

This is **real-browser qualitative functional evidence**, not merely a synthetic local API smoke test.

## Limits and research integrity

- Two manually selected images are **not sufficient to infer accuracy, recall, F1, pixel-IoU, model calibration, or operational robustness**.
- The images were downloaded for UI testing. Their membership in the previously recorded **67-image final held-out set is not established by the screenshots**. The simple Kaggle sample-download snippet selects from `test/good` and `test/*`; that does not distinguish the validation holdout. Do not add these two results to held-out evaluation statistics until their paths are checked against the persisted `splits.json`.
- The score is a **raw model score**, not a defect probability, and the heatmap is a **relative color visualization**, not a ground-truth defect mask. Without the segmentation mask, the visual overlap cannot establish localization accuracy.
- Displayed times are two individual request measurements, not p50/p95 or an SLA. A separate 10-request synthetic-input latency sample is recorded in [the performance report](2026-10-14-macos-10-request-inference-benchmark.md).
- The captured browser screenshots are available in the conversation only; they have not been uploaded to the repository. Avoid redistributing MVTec AD examples without reviewing dataset licensing and permission.

## Next acceptance gates

- For reproducibility, use the saved split manifest to select an actual held-out normal and defective bottle image, and record the source image paths/labels without publishing restricted data.
- Add optional manual visual QA for upload validation, empty file, unsupported format, and API service unavailable.
- Execute and record an actual PatchCore comparison using an exactly matching evaluation partition.
- For a public service, address authorization, request limits, logging/privacy, model lifecycle and deployment isolation.

**Milestone outcome:** The local browser image-upload, model-classification and heatmap display loop has been observed working for both requested visual classes. This is a **prototype UI functional milestone**, not the completion of industrial product qualification.
