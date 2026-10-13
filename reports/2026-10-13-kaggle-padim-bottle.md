# Kaggle verification: PaDiM / MVTec AD bottle

**Date:** 2026-10-13 (JST)
**Evidence:** User-supplied executed Kaggle notebook `ihsanvision-ai-industrial-anomaly-detection.ipynb`, code cell with `python -m src.train` output.
**Status:** Training + saved-checkpoint test successfully completed, as shown in the supplied notebook. This report is a transcription of that execution evidence, **not** an independently rerun benchmark.

## Command captured

```bash
python -m src.train \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --category bottle \
  --model padim \
  --output-dir /kaggle/working/ihsanvision_runs \
  --accelerator cpu
```

The branch used by the captured notebook was `feature/portfolio-production-pipeline`. The output includes the PyTorch 2.10.0+cpu and Anomalib 2.5.0 environment; earlier cells report NumPy 2.0.2.

## Results captured from the pipeline output

| Metric | Raw value | Percentage |
| --- | ---: | ---: |
| image_AUROC | 0.9963235855102539 | 99.63% |
| image_F1Score | 0.9591836929321289 | 95.92% |
| pixel_AUROC | 0.9767602682113647 | 97.68% |
| pixel_F1Score | 0.6497339010238647 | 64.97% |

Other run facts observed:

- **Train good images:** 209 (separately counted in another executed Kaggle cell).
- **Training:** 27/27 batches, one epoch, CPU, reported 00:29.
- **Test:** 9/9 batches, reported 00:19.
- **Checkpoint:** `/kaggle/working/ihsanvision_runs/padim-bottle-seed42/model.ckpt`.
- The output confirms: `Restoring states from the checkpoint path` and `Loaded model weights from the checkpoint`; the final command prints the checkpoint path and all four test metrics.
- The full `report.json` and `splits.json` **were not uploaded** with the notebook, so their content and exact sample counts have **not** been independently inspected.

## Interpretation

The pipeline uses `val_split_mode="from_test"`, `val_split_ratio=0.2`, and `test_split_mode="from_dir"`. These are **held-out remainder-of-test** results, not the complete official MVTec AD test protocol. Therefore, comparison to third-party MVTec benchmark tables is not valid without matching evaluation splits and threshold procedures.

AUROC measures ranking/discrimination. The pixel-level F1 (64.97%) is substantially lower than pixel AUROC (97.68%); visual localization and threshold calibration require further investigation.

## Next verification

1. Inspect `report.json` and `splits.json` to confirm split counts and threshold/calibration protocol.
2. Use the saved checkpoint for real single-image inference via `python -m src.predict` (not yet demonstrated by the uploaded notebook).
3. Inspect the overlay for a known defective image.
4. Execute equivalent held-out tests with PatchCore on the same seed, category and dataset.
5. Collect a deployable demo result and document hardware-dependent per-image latency.

The original notebook's separate `engine.test()` cell returned different values and uses its own data-module defaults. Do **not** combine its results with this pipeline run as if they came from one evaluation protocol.
