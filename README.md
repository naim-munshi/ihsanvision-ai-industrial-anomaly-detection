# IhsanVision.AI — Industrial Anomaly Detection

A reproducible computer-vision prototype for **unsupervised industrial defect detection and localization**. It trains **PaDiM** and **PatchCore** on the [MVTec AD benchmark](https://www.mvtec.com/company/research/datasets/mvtec-ad), produces auditable training/test splits, saves checkpoints and metrics, and provides a FastAPI service with a responsive browser demo.

> **Status:** Software implementation and lightweight automated tests are included. Full MVTec training, real-checkpoint reload/inference, latency, and end-to-end deployment **must still be executed and verified** on an environment with the dataset and Anomalib 2.5.0 installed. No production reliability claim is made.

## Architecture

```text
MVTec AD dataset
  ├── train/good images ──────> PaDiM / PatchCore fit
  └── test/ ──> disjoint validation holdout ──> model calibration
                    └────────> remaining final test ──> metrics report
                              |
                  Lightning model.ckpt
                              |
Image upload ──> FastAPI ──> Anomalib Engine.predict ──> score + overlay
                              |
                        Browser demo
```

### Technology stack

| Layer | Tools |
| --- | --- |
| Vision models | Anomalib 2.5.0, PyTorch, PaDiM, PatchCore |
| Experiment data | MVTec AD, Pandas, deterministic split seed |
| Inference | Lightning checkpoints, Anomalib Engine |
| Backend | FastAPI, Python multipart |
| Interface | HTML/CSS/JavaScript (no build step) |
| Testing/CI | pytest, GitHub Actions, Python 3.11/3.12 |

## Getting started

Python 3.11 or 3.12 is recommended. Create an environment:

```bash
git clone https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
cd ihsanvision-ai-industrial-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU installs, follow [PyTorch's platform selector](https://pytorch.org/get-started/locally/) to pick the correct CPU/CUDA/MPS-compatible wheels before installing the other requirements. The first model run may download pretrained backbone weights.

**Dataset directory:** Supply a legal local copy of MVTec AD; this project does **not** redistribute images or masks.

```text
/path/to/mvtec_ad/
  bottle/
    train/good/
    test/good/
    test/broken_large/
    ground_truth/
  cable/
  ...
```

MVTec AD uses **CC BY-NC-SA 4.0**; check its license and conditions before any commercial use.

## Kaggle preflight (no download or training)

After attaching the MVTec AD input in Kaggle and cloning this feature branch, run:

```bash
python -m scripts.kaggle_preflight
```

The script searches the common Kaggle nested paths, prints the usable `dataset_root`, and verifies normal training images plus normal/anomalous test images exist. It **does not** start training or download data. Copy the exact `dataset_root` field into the training command below. For a custom location, use `--input-dir` or `--dataset-root`.

A shared Kaggle notebook can use:

```python
!python -m scripts.kaggle_preflight
```

If the repository was cloned before this utility was added, run `git pull` **on the feature branch** first.

## Train a model

```bash
python -m src.train \
  --dataset-root /path/to/mvtec_ad \
  --category bottle \
  --model padim \
  --output-dir runs \
  --seed 42 \
  --accelerator cpu
```

For PatchCore, use `--model patchcore`. On compatible hardware, `--accelerator auto` can select an accelerator automatically. To change worker and batch settings, use `--num-workers`, `--train-batch-size`, and `--eval-batch-size`.

**Outputs:** `runs/padim-bottle-seed42/model.ckpt`, `report.json`, `splits.json`, and engine artifacts. Model weights and datasets are ignored by Git. Reusing the same output directory with an existing `report.json` is deliberately rejected; keep each experiment identifiable.

### Evaluation protocol and data leakage

MVTec AD has an official training folder and test folder, but no independent validation directory. The pipeline configures `val_split_mode="from_test"` and `val_split_ratio=0.2`; validation samples are reserved from the official test folder, while the remaining samples are evaluated once as held-out test data. Paths are checked for duplicates/overlap and written to `splits.json`.

This is **not the same as** evaluating on the full official MVTec test split. Consequently, these test metrics should **not** be compared directly with full-test or different-threshold published benchmark results. The method is suited for an auditable portfolio experiment but not a standardized leaderboard submission. With small categories, check each split's class coverage before interpreting AUROC.

The saved `report.json` contains the actual numeric metrics returned by Anomalib and runtime versions. It does not invent or backfill results when training has not run.

## Benchmark PaDiM against PatchCore

```bash
python -m src.benchmark \
  --dataset-root /path/to/mvtec_ad \
  --categories bottle cable \
  --seed 42 \
  --output-dir runs/benchmark
```

The script runs the two models sequentially per category, then writes `comparison.csv` and `comparison.json`. Runtime depends heavily on image count and available hardware. Inspect actual metric names in each generated report; they depend on the installed Anomalib evaluator. Add hardware specs, peak memory, per-image inference latency and samples/sec before making performance comparisons.

## Run single-image inference

```bash
python -m src.predict \
  --image /path/to/mvtec_ad/bottle/test/broken_large/000.png \
  --checkpoint runs/padim-bottle-seed42/model.ckpt \
  --model padim \
  --output runs/example-overlay.png
```

The CLI prints a **raw anomaly score**, a prediction label if Anomalib provides one, timing including checkpoint loading, and writes a visualization PNG. The score is **not a probability**; the overlay uses **per-image contrast scaling** and is **not a calibrated segmentation mask**.

**Security:** Lightning `.ckpt` files are pickle-based. Load only files you personally created or otherwise trust. Never let remote users upload model checkpoints.

## Serve the API and web demo

```bash
export IHSANVISION_CHECKPOINT="$PWD/runs/padim-bottle-seed42/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for the browser demo or **http://127.0.0.1:8000/docs** for OpenAPI/Swagger.

| Method | Path | Use |
| --- | --- | --- |
| GET | `/` | Browser interface |
| GET | `/health` | Health and checkpoint configuration |
| POST | `/predict` | Multipart image upload, returns JSON score, label, overlay as base64 PNG |

The API rejects empty, invalid, overlarge, or unsupported image types (max 10 MiB / 25 megapixels). It never accepts arbitrary checkpoint paths from clients. **Run behind authentication and a reverse proxy before any external deployment**; currently there is no authentication, rate limit, tenant isolation, persistent prediction history, or streaming inference. The current inference implementation reloads its checkpoint for each prediction; measured response times include this overhead. Do not claim production throughput or low latency.

## Automated tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

CI covers static compilation and **lightweight tests** for config validation, split leakage checks, image overlay output, API health, image upload validation, and mocked inference. It intentionally does **not** download MVTec data or pretrained model weights. A **real model smoke test** must be performed separately:

1. Train one MVTec category with `python -m src.train`.
2. Verify `splits.json` contains three non-overlapping groups.
3. Verify `model.ckpt` and `report.json` were generated.
4. Run `python -m src.predict` with that exact checkpoint.
5. Run `uvicorn` and upload an image in the browser.
6. Run the PaDiM/PatchCore benchmark and include a reproducible results table.

## Repository map

```text
src/
  config.py       model selection, validation, split auditing
  train.py        training, checkpoint, evaluation
  predict.py      single-image inference and overlay
  benchmark.py    sequential model comparison
  api.py          FastAPI inference endpoints
demo/index.html   responsive inspection demo
tests/            offline unit and API tests
.github/workflows/ci.yml
notebooks/ihsanvision_mvtec_padim.ipynb
requirements.txt
requirements-dev.txt
```

## Historical exploratory notebook

The original Kaggle experiment reported the following **historical, user-reported** PaDiM results for `bottle`: image AUROC 1.000, image F1 0.992, pixel AUROC 0.9822, pixel F1 0.7164. These values are **not reproduced or independently verified** by the new pipeline, and its separate validation/test protocol prevents direct numerical comparison. Keep the notebook as a provenance record, **not** a production benchmark.

## Known limitations and engineering backlog

- Run and document the real Anomalib 2.5.0 end-to-end smoke test on the provided MVTec images.
- Test model reloading and threshold persistence with both PaDiM and PatchCore checkpoints.
- Record model-level image/pixel AUROC, F1, class support, inference p50/p95, hardware and dataset version.
- Consider validation-only threshold calibration, stronger stratified holdouts, external domain-shift testing and repeat seeds.
- Add service authentication, backpressure, observability, batching and ONNX/OpenVINO export before calling it production-ready.

## Data attribution

Dataset: Bergmann et al., **The MVTec Anomaly Detection Dataset**, International Journal of Computer Vision (2021). See the [official project](https://www.mvtec.com/company/research/datasets/mvtec-ad) for licensing and citation. Models are provided by the open-source [Anomalib](https://github.com/open-edge-platform/anomalib) ecosystem. Dataset assets and model weights are not committed to this repository.
