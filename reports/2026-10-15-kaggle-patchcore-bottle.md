# 2026-10-15 — PatchCore bottle: completed Kaggle benchmark observed

## Evidence status

**Source:** User-provided Kaggle notebook console output on 2026-10-15 (JST). It shows the completion of a real Anomalib 2.5.0 PatchCore fit and held-out evaluation. The raw `comparison.json`, PatchCore `report.json`, `splits.json`, and `model.ckpt` have **not yet been independently copied into or inspected from this source repository**. This report therefore records **observed console evidence**, not an independently archived or replicated result.

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

## Required follow-up before portfolio publication

1. Preserve **both** checkpoints, reports and split manifests, plus `comparison.json` and `comparison.csv`, outside ephemeral Kaggle `/working` (prefer private storage and a local backup).
2. Revalidate `comparison.json` and original reports with `python -m scripts.summarize_comparison --runs-dir ...` after updating a local checkout to contain PR #5.
3. Record hardware/environment provenance via `scripts.environment_snapshot` (PR #6), making clear when it was captured.
4. Run a separate held-out defective-image PatchCore checkpoint smoke test and archive the overlay. This is **not** equivalent to metric verification.
5. Avoid uploading MVTec images or Lightning checkpoint binaries into this public source repository.

## Evaluation cautions

The validation holdout is drawn from the original MVTec test images; consequently these numbers are **not directly comparable** to published full-test MVTec leaderboard metrics. F1 is threshold-dependent; AUROC is a ranking metric. Relative heatmaps and uncalibrated anomaly scores are not defect probabilities. Neither a single category nor one seed establishes production reliability.
