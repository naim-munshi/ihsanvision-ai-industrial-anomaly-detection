# 2026-10-15 — PatchCore bottle: completed Kaggle benchmark observed

## Evidence status

**Source:** User-provided Kaggle notebook console output on 2026-10-15 (JST). It shows a real Anomalib 2.5.0 PatchCore fit, held-out evaluation, and checkpoint-backed smoke inference. Later the user downloaded a Kaggle ZIP and extracted it on their Mac; a macOS `shasum -a 256 -c SHA256SUMS.txt` check returned `OK` for all eight backed-up model/comparison files. The checkpoint binaries and raw JSON **remain outside the public source repository**, and their report contents have not yet been independently cross-checked by the offline summary utility. The original metrics are **console-observed**; the archive is checksum-verified against its bundled manifest, not independently reproduced.

The notebook ended with the following messages:

~~~text
Selecting Coreset Indices.: 100%|██████████| 21400/21400 [49:08<00:00,  7.26it/s]
Trainer.fit stopped: max_epochs=1 reached.
Restoring states from the checkpoint path at /kaggle/working/ihsanvision_runs/patchcore-bottle-seed42/model.ckpt
Loaded model weights from the checkpoint at /kaggle/working/ihsanvision_runs/patchcore-bottle-seed42/model.ckpt
Saved /kaggle/working/ihsanvision_runs/comparison.csv
PaDiM checkpoint preserved.
BENCHMARK COMPLETED SUCCESSFULLY
~~~

The checkpoint preservation message follows the caller's SHA-256 equality assertion; the previously saved PaDiM checkpoint's SHA-256 was independently displayed by the user as `990aee31d88375dfdd7d6cf46ead38a61348f86bfe73a3026be6c071d569d29b`.

## Experimental protocol

- Dataset: MVTec AD, category `bottle`.
- Random seed: 42.
- Recorded reference splits: 209 normal training images, 16 validation images held out from the original test directory, 67 remaining test images.
- Models: an earlier saved PaDiM experiment reused via `--reuse-existing`; PatchCore trained in this session.
- The benchmark implementation checks that the PatchCore train/validation/test paths match the PaDiM split manifest *before* PatchCore fitting and once more before publishing comparison files. Observing benchmark completion supports that those checks did not raise errors. Independent artifact revalidation is pending.
- Environment reported in the notebook: Python 3.12; Anomalib 2.5.0, NumPy 2.0.2, Imagecodecs 2025.8.2 and PyTorch 2.10.0+cpu; no GPU used.

## Observed held-out test metrics

| Metric | Earlier PaDiM reported result | Newly observed PatchCore result |
| --- | ---: | ---: |
| `image_AUROC` | 0.9963235855 | **1.0000000000** |
| `image_F1Score` | 0.9591836929 | **0.9900990129** |
| `pixel_AUROC` | 0.9767602682 | **0.9856935740** |
| `pixel_F1Score` | 0.6497339010 | **0.7102153301** |

PatchCore's numerical advantage in this *particular held-out split* is a measured result, not evidence of general superiority over PaDiM or a guarantee of real-world defect detection. A perfect observed image AUROC is **not** a guarantee of 100% detection performance.

## Observed training/evaluation detail

- PatchCore default model was logged at 24.9 million parameters (not a measure of parameter *updates*; PatchCore trains its reference memory bank rather than optimizing a typical classification loss).
- CPU coreset selection: 21,400 of 21,400 indices, elapsed approximately 49 min 8 sec.
- Lightning log: fit stopped on `max_epochs=1`; displayed 27/27 training batches and 9/9 test batches.
- Checkpoint was reloaded for testing, and `comparison.csv` was reported saved.
- The coreset-selection duration is **not** the complete training time. The two models' training times came from separate sessions and are not directly comparable as a controlled speed benchmark.

## 2026-10-15 follow-up: Mac archive verification and real-image inference

A saved Kaggle output archive, `ihsanvision_benchmark_backup.zip`, was reported as **389.04 MB**, containing eight data/model files plus `SHA256SUMS.txt`. `unzip -t` reported no compressed-data errors. After extracting it into `~/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/`, the user ran `shasum -a 256 -c SHA256SUMS.txt` and all eight entries returned `OK`:

- PaDiM `model.ckpt`, `report.json`, `splits.json`.
- PatchCore `model.ckpt`, `report.json`, `splits.json`.
- `comparison.json` and `comparison.csv`.

The bundled manifest is an internal integrity reference; this check alone does **not** independently validate scientific correctness, model security, provenance or the numeric contents of the JSON reports.

The user also executed `scripts.kaggle_smoke_test` for `patchcore` on the saved held-out test manifest. The Kaggle output reports:

| Single-image smoke field | Observed value |
| --- | --- |
| Defective held-out input | `/kaggle/input/datasets/ipythonx/mvtec-ad/bottle/test/broken_large/000.png` |
| Checkpoint | `/kaggle/working/ihsanvision_runs/patchcore-bottle-seed42/model.ckpt` |
| Predicted anomaly | `true` |
| Raw anomaly score | `0.9287775754928589` |
| One-shot inference duration | `4174.69 ms` |
| Split counts | Train 209 / validation 16 / test 67 |
| Status | `single-image-inference-passed` |
| Saved artifacts | `smoke_report.json`, `smoke_overlay.png` |

The user-provided output shows that Lightning loaded the saved model weights and that a valid overlay PNG was generated. The timing includes checkpoint loading overhead and is **not a warm inference speed benchmark**. The score is not a calibrated probability. The newly produced smoke evidence was not part of the first ZIP and still needs a durable private backup.

## Required follow-up before portfolio publication

1. Keep the Mac ZIP **and extracted checksum-verified files** in private durable storage. Preserve the independent original PaDiM backup too; do not upload checkpoints or MVTec images to public GitHub.
2. Revalidate `comparison.json` against both saved model reports and exact split manifests with `python -m scripts.summarize_comparison --runs-dir ...` from an updated `main` checkout; this is still pending.
3. Save the new PatchCore `smoke_report.json` and `smoke_overlay.png` privately before the Kaggle session expires (these files were generated **after** the 389 MB ZIP).
4. Record hardware/runtime provenance using `scripts.environment_snapshot`; a current snapshot cannot prove the previous PaDiM environment.
5. Run further independent-sample and deployment tests, including multi-category and multiple-seed analysis. One positive smoke prediction is **qualitative evidence**, not recall or robustness evidence.

## Evaluation cautions

The validation holdout is drawn from the original MVTec test images; consequently these numbers are **not directly comparable** to published full-test MVTec leaderboard metrics. F1 is threshold-dependent; AUROC is a ranking metric. Relative heatmaps and uncalibrated anomaly scores are not defect probabilities. Neither a single category nor one seed establishes production reliability.
