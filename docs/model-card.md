# Model Card — IhsanVision.AI (PaDiM / MVTec AD Bottle)

## Model details

| Field | Value |
| --- | --- |
| Model family | PaDiM (Patch Distribution Modeling) |
| Framework | Anomalib 2.5.0 / PyTorch Lightning |
| Dataset | MVTec Anomaly Detection, **bottle** category |
| Training | One-class/normal appearance modelling |
| Trained weights | `model.ckpt`, held outside Git repository |
| Random seed | 42 |
| Stage | Research portfolio prototype |
| Model owner | IhsanVision.AI project maintainer |
| Provenance | [Executed experiment report](../reports/2026-10-13-kaggle-padim-bottle.md) |

**Note:** This document describes **one specific trained PaDiM checkpoint**, not all models selectable by the software. PatchCore has a model factory and benchmark runner but no verified Kaggle result yet.

## Intended use

Research, coursework, portfolio demonstration and human-reviewed industrial quality-inspection experiments using representative bottle images.

The model predicts a **raw anomaly score**, a binary anomaly label if its thresholds are available, and a **relative image heatmap** to help interpret possible defects. Use model output alongside manual inspection.

## Training and evaluation

Training uses 209 normal bottle images. The experiment creates a disjoint 16-image validation subset from the original test folder and evaluates against the remaining 67-image held-out test subset, with explicit split audits.

| Held-out metric | Observed value |
| --- | ---: |
| Image AUROC | 0.9963 |
| Image F1 | 0.9592 |
| Pixel AUROC | 0.9768 |
| Pixel F1 | 0.6497 |

See [evaluation protocol](evaluation.md) for threshold-dependence, data leakage checks and the reason these metrics **cannot be compared directly to published full-test MVTec scores**.

A Kaggle-based saved-checkpoint smoke test on one defective image produced a predicted anomaly and PNG overlay. This was not a batch production test.

## Limitations

- Bottle images only: no measured generalization to other MVTec categories or manufacturing environments.
- Pixel F1 is materially below pixel AUROC; do not assume accurate binary defect boundaries.
- Raw anomaly scores are **not calibrated probabilities** and should not be compared across model families as probabilities.
- Per-image normalized heatmaps are **not calibrated segmentation masks**.
- Domain shift, lighting changes, camera differences and unseen defect types have not been systematically evaluated.
- One observed checkpoint-backed inference latency includes model loading, and is not warm-serving throughput.
- No production safety certification, reliability guarantee, fairness/robustness study or deployment SLA.

## Security, provenance and distribution

The checkpoint is a **trusted local experiment artifact**. PyTorch/Lightning serialized checkpoint files must not be loaded from unknown sources. The training data and model weights are not redistributed through this repository.

MVTec AD has its own licensing restrictions. Review the [official dataset information](https://www.mvtec.com/company/research/datasets/mvtec-ad) before use, especially commercially.

## Reproduction

The repository's [Kaggle setup](development.md#kaggle-experiments) describes the training command, dataset discovery, and checkpoint verification. Keep `model.ckpt`, `report.json`, `splits.json`, `smoke_report.json` and `smoke_overlay.png` together under the experiment artifact directory.
