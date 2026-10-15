# Evaluation Methodology and Results

## Scope

This document records what has **actually been executed** in Kaggle and separates measured results from planned benchmark work. These values are not marketing claims or standardized cross-paper comparisons.

## Dataset and experimental protocol

**Dataset:** MVTec AD, category `bottle`.

**Model:** PaDiM, Anomalib 2.5.0; the new pipeline was invoked with `--accelerator cpu` and random seed 42.

**Split construction:** Training samples come from the dataset's `train/good` directory. MVTec AD's test directory is divided into two non-overlapping partitions via Anomalib's `val_split_mode="from_test"` and `val_split_ratio=0.2`. The validation partition is used by the Anomalib evaluation/postprocessing workflow; the remaining images are used as final test samples. A path-level manifest and overlap checks are part of `src/config.py`.

| Split | Observed count |
| --- | ---: |
| Train (normal) | 209 |
| Validation | 16 |
| Remaining held-out test | 67 |

**Important:** This differs from the published full-test MVTec AD protocol. Therefore do not compare the reported test values directly with those of external papers or the previous notebook's default evaluation.

## Recorded PaDiM/bottle results

The user-executed Kaggle notebook showed a completed training run, checkpoint saving and reloading for test evaluation:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| `image_AUROC` | **0.9963235855** | Image-level ranking performance |
| `image_F1Score` | **0.9591836929** | Image-level threshold-dependent F1 |
| `pixel_AUROC` | **0.9767602682** | Pixel-wise anomaly ranking |
| `pixel_F1Score` | **0.6497339010** | Pixel-wise threshold-dependent F1 |

The user-uploaded executed notebook and [run report](../reports/2026-10-13-kaggle-padim-bottle.md) are the provenance for this table. The entire dataset, checkpoint binary, all masks and `report.json` were **not** independently collected into this repository.

### What the metrics mean

- **AUROC** evaluates score discrimination over different decision thresholds; 1.0 is perfect.
- **F1** is the harmonic mean of precision and recall at the evaluated threshold.
- **Image metrics** operate on whole-image normal vs defective predictions.
- **Pixel metrics** evaluate localization with per-pixel masks.

The **pixel F1 (~0.65)** is materially below pixel AUROC (~0.98). Good anomaly ranking therefore does not automatically imply accurate binary segmentation. Further threshold analysis and visual inspection are required.

## Single-image inference verification

A separate user-run `scripts.kaggle_smoke_test` produced the following visible output:

| Observation | Value |
| --- | --- |
| Model/category | PaDiM / bottle |
| Result status | `single-image-inference-passed` |
| Prediction | `true` (flagged as anomalous) |
| Raw anomaly score | Approximately **0.8633** |
| Duration | Approximately **1396.49 ms** |
| Saved overlay | `smoke_overlay.png` |

The screenshot shows a valid heatmap overlay on a visibly damaged bottle. The smoke script checks for a checkpoint, report, split manifest, image inference and a valid PNG overlay. **For future executions, the script now restricts candidate defective images to paths appearing in the saved held-out `test` split**, rejecting a run where no such defective test image remains; it does not fall back to a validation image. This is a protocol safeguard, not a new measured model result.

**What this does not establish:** high recall on other defect types, pixel-level segmentation validity, a calibrated defect probability, model fairness/robustness, or live API throughput. Because this prediction loads the checkpoint, the reported duration is not steady-state latency.

## Browser demonstration (qualitative only)

The local browser dashboard was tested with `normal_bottle.png` and `defective_bottle.png`. The screenshots show raw PaDiM scores **0.3637 (predicted normal)** and **0.8633 (flagged anomaly)** with image previews and colored anomaly overlays. See the [UI inspection record](../reports/2026-10-14-browser-bottle-inspection.md).

**Do not count these as two new independent held-out evaluation samples** until their original dataset paths are compared with the saved `splits.json`. This manual demonstration checks UI behavior and plausible qualitative prediction, not model-level validation metrics.

## How to reproduce

Follow [development.md](development.md#kaggle-experiments), using the same dataset path, Anomalib version, model, seed and split configuration. Compare `report.json`, `splits.json` and generated checkpoint only within the same protocol.

For each *new* experiment log:

1. Dataset version, class, split counts and ground-truth presence.
2. Model and backbone, seed, library versions, CPU/GPU and available memory.
3. Train duration, image/pixel AUROC and F1, and sample counts by normal/anomaly.
4. Inference latency p50/p95 under an explicitly defined warm/cold setup.
5. A visual comparison of original image, ground-truth mask and predicted heatmap.

## Observed PatchCore/bottle comparison (2026-10-15)

The user-provided Kaggle console output shows completed **PatchCore** training, saved-checkpoint reload, held-out evaluation, and `comparison.csv` creation; the PaDiM checkpoint hash equality check also passed. The source repository does **not** yet contain the new run's JSON records or checkpoint; see the [observed evidence report](../reports/2026-10-15-kaggle-patchcore-bottle.md) for provenance and limits.

| Metric | Earlier PaDiM | Newly observed PatchCore |
| --- | ---: | ---: |
| Image AUROC | 0.9963235855 | **1.0000000000** |
| Image F1 | 0.9591836929 | **0.9900990129** |
| Pixel AUROC | 0.9767602682 | **0.9856935740** |
| Pixel F1 | 0.6497339010 | **0.7102153301** |

The completed benchmark orchestrator requires identical recorded train/validation/test paths before publishing the comparison. This supports protocol consistency, though source reports and split manifests still require independent **content** verification. The user subsequently saved an archive on macOS and passed SHA-256 verification for all eight model/comparison artifacts against its bundled checksum manifest. Results reflect a single category and seed with validation images held out from the original test directory. They are **not** directly comparable to published full-test MVTec leaderboard scores. Coreset selection took 49 min 8 sec on CPU; this is not total train time and does not establish a controlled PaDiM/PatchCore speed comparison. A later saved PatchCore checkpoint smoke test on a held-out `broken_large/000.png` predicted anomaly `true`, with raw score `0.9287775754928589` and one-shot duration `4174.69 ms`, and produced `smoke_report.json` and `smoke_overlay.png`. This is qualitative evidence only; the duration includes checkpoint loading. The smoke files were generated **after** the verified archive and still require backup. Future work: JSON content-level audit, smoke artifact backup, multiple seeds and categories.

## Research integrity

- Do not tune to or repeatedly select a best model on the final test subset.
- Keep thresholds and calibration choices based on validation data, not final test outcomes.
- Keep the exploratory notebook's **historical** metrics separate from the new pipeline.
- Never fabricate missing categories, latencies or confusion matrices.
