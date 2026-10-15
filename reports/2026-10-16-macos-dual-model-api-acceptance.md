# 2026-10-16 — real macOS dual-model API acceptance

## Evidence

**Source:** User-shared Terminal output from a Mac running a localhost IhsanVision FastAPI service, executing `python -m scripts.dual_model_acceptance` from an updated GitHub `main` worktree at commit `ae778e2`. This report describes the observed CLI result, **not a rerun performed by GitHub Actions**.

The output printed `Saved acceptance evidence: .../dual_model_acceptance_2026-10-16.json` and `"status": "passed"`. Two manually selected 900 × 900 bottle PNGs were used from the user's local `bottle-ui-test` folder. The report file, source images, checkpoints and overlays remain **private on the user's Mac**.

## Four real checkpoint-backed localhost predictions

| Model | User-selected sample | Raw anomaly score | Predicted anomaly | Expected label matched | UI/API reported span (ms) | Model reused | Overlay PNG valid |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| PaDiM | `normal_bottle.png` | 0.3637291193 | false | true | 769.34 | false | true |
| PaDiM | `defective_bottle.png` | 0.8632831573 | true | true | 238.82 | true | true |
| PatchCore | `normal_bottle.png` | 0.1275397539 | false | true | 1132.07 | false | true |
| PatchCore | `defective_bottle.png` | 0.9287785292 | true | true | 397.65 | true | true |

The acceptance report recorded SHA-256 hashes of the **original user-supplied images**:

| Sample | SHA-256 |
| --- | --- |
| Normal | `9e7bfe658efe9b6db2607a96405b4ac6880936eaf93b89c85bcd311122ceb57b` |
| Defective | `11a691a06a33dd78c5fcab6a822caa48c0d72f27f660e19f5abd9b5aa392b0eb` |

The acceptance script tests two model selections × two labeled samples over HTTP and validates reported model identity, prediction type, finite score and decoded PNG overlay dimensions. Both samples were classified according to their supplied labels. A cache miss followed by cache reuse was observed for each model family in sequence.

## What this establishes

The live localhost API loaded the two trusted model families, routed user-selected requests correctly, returned plausible scores/boolean labels, produced valid same-size PNG localization overlays and reused the saved model on repeated requests. The outcome strengthens the four prior Safari browser observations, which are separately documented in [macOS UI results](2026-10-15-macos-dual-model-dashboard-normal-ui.md).

## What this does **not** establish

- **No statistical performance claim:** four manually selected outcomes do not provide estimates of defect recall, specificity, F1 or model quality across production data.
- **No ground-truth provenance check:** the original file SHA-256 hashes are recorded, but their membership in the saved Kaggle held-out `test` manifest has **not** been verified. The normal and defective paths are locally selected.
- **No pixel localization score:** validating overlay bytes, PNG format and dimensions does not prove correct segmentation, pixel AUROC or pixel IoU for these inputs.
- **No controlled speed comparison:** request times include possible checkpoint restoration and sequential/caching effects. The raw PaDiM and PatchCore scores are not calibrated or comparable across model types. The four durations are not p50/p95 latency or throughput.
- **No production security acceptance:** no TLS gateway/hosted deployment, multiworker throughput, admission/load or comprehensive UI error-state QA has been demonstrated.

## Follow-up gates

1. Use the saved `dual_model_acceptance_2026-10-16.json` only as private run evidence; do not commit images or model weights.
2. Record source image provenance against the saved train/val/test manifests without changing the held-out protocol.
3. Execute M7 profiler with controlled input across both models, archived p50/p95 and subsequent concurrency/admission checks.
4. Test API/UI error cases, deployment container execution and the portfolio release gate before a public launch.
