# Model card — IhsanVision.AI Bottle PaDiM and PatchCore

**Scope:** two actual saved Anomalib model families evaluated on **MVTec AD / bottle**, seed **42**. This is an applied ML engineering **portfolio experiment**, not a universal industrial anomaly detector, calibrated classifier, or published leaderboard entry.

| Field | PaDiM | PatchCore |
| --- | --- | --- |
| Framework | Anomalib 2.5.0 / PyTorch Lightning | Anomalib 2.5.0 / PyTorch Lightning |
| Task | Normal-only anomaly detection and localization | Normal-only anomaly detection and localization |
| Dataset | MVTec AD, bottle | MVTec AD, bottle |
| Source | [Observed Kaggle run](../reports/2026-10-13-kaggle-padim-bottle.md) | [Observed Kaggle run](../reports/2026-10-15-kaggle-patchcore-bottle.md) |
| Private artifact | `padim-bottle-seed42/model.ckpt` | `patchcore-bottle-seed42/model.ckpt` |
| Local inference | [Real macOS/API acceptance](../reports/2026-10-16-macos-dual-model-api-acceptance.md) | [Real macOS/API acceptance](../reports/2026-10-16-macos-dual-model-api-acceptance.md) |
| Intended use | Unsupervised (normal-only) research/demo defect inspection | Unsupervised (normal-only) research/demo defect inspection |

## Data and evaluation methodology

Both runs recorded the same category/seed and matching **train, validation and test image paths**. The documented split uses **209 normal training images**, **16 validation images** held out from the original MVTec test directory, and **67 remaining held-out test images**.

The validation subset is not an independent official validation set. The resulting metrics **cannot be fairly compared to official full-test MVTec AD leaderboard numbers**. The shared path manifest is strong evidence of pairwise protocol agreement but does **not prove underlying image-byte equivalence** without additional content hashes.

| Held-out metric | PaDiM | PatchCore |
| --- | ---: | ---: |
| Image AUROC | 0.9963235855 | 1.0000000000 |
| Image F1 | 0.9591836929 | 0.9900990129 |
| Pixel AUROC | 0.9767602682 | 0.9856935740 |
| Pixel F1 | 0.6497339010 | 0.7102153301 |

These are saved/observed **single-seed, single-category** results; sample sizes are modest. A PatchCore image AUROC of 1.0000 here does **not** imply perfect defect detection in manufacturing. F1 values depend on the model's evaluation threshold. The pixel F1 values (especially PaDiM) indicate localization mistakes even when AUROC is high.

Model decision thresholds, weights and original raw JSON reports should be validated against the user's private artifact archive before any new release claim. The checkpoint files are intentionally **not stored in Git**.

## Inference outputs and bounded usage

The serving API returns a model/category identifier, **raw uncalibrated anomaly score**, optional anomaly decision and a **relative normalized heatmap**. These heatmaps are not globally calibrated segmentation masks. PaDiM and PatchCore raw scores are not comparable without calibration.

Real, manually selected Mac Bottle images yielded correct normal/defective labels and valid same-dimension overlay PNGs for both model families; four examples **do not provide a statistically valid recall estimate** or guarantee membership in the saved held-out split.

## Hardware and latency (observed warm Mac HTTP profiles)

| Sequential HTTP round trip | PaDiM | PatchCore |
| --- | ---: | ---: |
| p50 (ms) | 388.87 | 493.05 |
| p95 (ms) | 766.46 | 514.82 |
| Cached measured calls | 20/20 | 20/20 |

One user-provided 900×900 image was used with three warmups and 20 sequential measured calls per model. This benchmark does not measure sustained throughput, model accuracy, peak memory, cold starts or full production tail-latency distributions. The original per-request JSON remains on the user's Mac. See [source report](../reports/2026-10-16-macos-paired-serving-profile.md).

## Limitations and foreseeable failure modes

- Lighting, material, camera resolution, alignment, lens artifacts, novel defect types and factory domain shift are **not** systematically evaluated.
- Rare high-cost false negatives may occur; do not auto-reject/accept physical products without independent human review and pilot validation.
- Per-image normalized heatmaps cannot be interpreted as calibrated defect boundaries or confidence maps.
- Small held-out samples and single seed cannot establish robust cross-factory generalization; multiple categories and explicit failure-case review are portfolio v1.0 acceptance gates.
- Lightning checkpoints may invoke unsafe deserialization when loaded from untrusted sources. Use only operator-trusted privately stored artifacts.
- No real new cable/carpet results exist. Those are [separate pending experiments](portfolio-v1-plan.md), not entries in this model card.

## Provenance, rights and reproducibility

Record `report.json`, `splits.json`, the exact checkpoint SHA-256 and hardware/library versions privately for each trained model. Run [read-only comparison validation](verified-benchmark-summary.md) and [portfolio evidence verification](portfolio-release-readiness.md) on the owner's preserved artifacts.

MVTec AD and pretrained backbones have their own usage terms. Dataset images, masks and checkpoint binaries are not redistributed through GitHub. This project is for research/portfolio demonstrations; any commercial use requires separate rights, robustness and safety review.
