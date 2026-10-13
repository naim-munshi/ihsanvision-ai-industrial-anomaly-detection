# IhsanVision.AI
### Industrial Anomaly Detection & Visual Quality Inspection

[![CI](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/workflows/ci.yml/badge.svg?branch=feature%2Fportfolio-production-pipeline)](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)
![Anomalib](https://img.shields.io/badge/Anomalib-2.5.0-6A5ACD)
![Stage](https://img.shields.io/badge/stage-research%20prototype-orange)

**IhsanVision.AI** is a computer vision software prototype for detecting and localizing manufacturing defects. It combines unsupervised anomaly detection (**PaDiM** and **PatchCore**) with reproducible MVTec AD experiments, checkpoint-based inference, a **FastAPI service**, and a responsive browser inspection interface.

**Current stage: research prototype.** Real PaDiM training and single-image inference have been exercised on Kaggle; hosted API/UI operation and the PatchCore benchmark are **not yet validated**.

## Project status

| Component | Implementation | Validation |
| --- | --- | --- |
| Models | PaDiM and PatchCore training | PaDiM/bottle verified on Kaggle |
| Evaluation | Disjoint split audit and metrics export | PaDiM/bottle report captured |
| Inference | Trusted checkpoint → score + overlay | Single-image PaDiM smoke test passed |
| Service | FastAPI: `/health`, `/predict`, Swagger | Mac localhost real-checkpoint HTTP smoke passed; browser interaction pending |
| Interface | Responsive upload and result dashboard | Implemented; live run pending |
| Quality | pytest and GitHub Actions, Python 3.11/3.12 | CI passing on development branch |

## Verified experiment: PaDiM / MVTec AD bottle

An executed Kaggle run using **Anomalib 2.5.0, CPU, seed 42** produced:

| Metric | Held-out test |
| --- | ---: |
| Image AUROC | **0.9963** |
| Image F1 | **0.9592** |
| Pixel AUROC | **0.9768** |
| Pixel F1 | **0.6497** |

The run used **209 normal train images, 16 validation images and 67 final test images**, as recorded in the Kaggle output. Validation images were sampled from the original MVTec AD test folder, then excluded from the remaining test set. **These results cannot be directly compared with full-test MVTec leaderboard scores.**

The subsequent *real* checkpoint inference on a defective bottle image returned `predicted_anomaly=true`, raw anomaly score **0.8633**, and an image overlay. Its observed **~1.4-second** latency includes checkpoint loading; it is **not** steady-state API latency. A raw anomaly score is **not a probability**, and the overlay is *relative*, not a calibrated segmentation mask.

[Training provenance](reports/2026-10-13-kaggle-padim-bottle.md) · [Evaluation details](docs/evaluation.md)

## System overview

~~~text
MVTec AD
  ├─ train/good ──> PaDiM / PatchCore ──> trusted checkpoint
  └─ test/* ──────> validation holdout + remaining final test
                                        │
                                     metrics

Browser image upload ──> FastAPI ──> Anomalib Engine.predict
                                         │
                               score + label + overlay
                                         │
                                   web dashboard
~~~

See [Architecture](docs/architecture.md) for boundaries, decisions, and limitations.

## Quick start

### 1. Train and verify on Kaggle

Attach the MVTec AD dataset to a Kaggle notebook. Clone the **feature branch** while PR #2 is under review:

~~~python
!git clone -b feature/portfolio-production-pipeline https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
%cd /kaggle/working/ihsanvision-ai-industrial-anomaly-detection
!python -m pip install -r requirements.txt
~~~

Verify the dataset location:

~~~python
!python -m scripts.kaggle_preflight
~~~

Train (replace the dataset root with the value reported by preflight):

~~~python
!python -m src.train \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --category bottle --model padim \
  --output-dir /kaggle/working/ihsanvision_runs \
  --accelerator cpu
~~~

Validate the saved checkpoint against a defective test image:

~~~python
!python -m scripts.kaggle_smoke_test \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --output-dir /kaggle/working/ihsanvision_runs
~~~

Outputs include `model.ckpt`, `report.json`, `splits.json`, `smoke_report.json` and `smoke_overlay.png`. The PaDiM/bottle experiment above has **already completed**; keep the checkpoint rather than retraining unnecessarily. See the full [Kaggle workflow](docs/development.md#kaggle-experiments).

### 2. Run the web application locally

Use Python 3.11 or 3.12. Consult [PyTorch's platform-specific selector](https://pytorch.org/get-started/locally/) for CPU/CUDA/Apple Silicon installations.

~~~bash
git clone -b feature/portfolio-production-pipeline https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
cd ihsanvision-ai-industrial-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export IHSANVISION_CHECKPOINT="/absolute/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL="padim"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the UI and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for OpenAPI.

The local service is implemented and tested with mocked predictions. **A real Mac HTTP → API → model round trip is now verified; clicking/uploading in the browser UI remains pending.** Only load checkpoints you trust.

### 3. Compare PaDiM and PatchCore

~~~bash
python -m src.benchmark \
  --dataset-root /path/to/mvtec_ad \
  --categories bottle cable \
  --seed 42 \
  --output-dir runs/benchmark
~~~

Metrics are written to `comparison.csv` and `comparison.json`. PatchCore results have **not yet been run/verified**; do not publish comparative performance claims.

## Project structure

~~~text
ihsanvision-ai-industrial-anomaly-detection/
├── src/                     # Training, inference, API, config, benchmark
├── demo/index.html          # Browser inspection interface
├── scripts/                 # Dataset preflight and checkpoint smoke test
├── tests/                   # Offline unit and contract tests
├── notebooks/               # Original Kaggle exploration
├── reports/                 # Experiment observations and provenance
├── docs/                    # Architecture, evaluation, workflow, security, roadmap
├── .github/workflows/       # Python 3.11/3.12 CI
├── requirements.txt
└── requirements-dev.txt
~~~

## Local integration checks

On macOS, after the Kaggle checkpoint has been copied to a trusted path outside the repository, run the **artifact/dependency preflight**:

~~~bash
python -m scripts.local_preflight
~~~

This checks package availability, the checkpoint file, and consistency of the saved train/validation/test reports. It **does not deserialize the checkpoint** or claim successful real-model inference.

Start the local API on a free port (8001 if 8000 is occupied):

~~~bash
export IHSANVISION_CHECKPOINT="$HOME/ihsanvision-artifacts/padim-bottle-seed42/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn src.api:app --host 127.0.0.1 --port 8001
~~~

From a **second terminal**, exercise the API with a synthetic image:

~~~bash
cd ~/ihsanvision-portfolio
source .venv/bin/activate
python -m scripts.local_api_smoke --url http://127.0.0.1:8001
~~~

The user successfully exercised the real checkpoint through this HTTP path on macOS (see [Mac API smoke report](reports/2026-10-14-macos-api-smoke.md)). The test used a **synthetic** image and took **66.69 seconds including checkpoint load**, so it does **not** measure defect-detection accuracy or warm throughput.

An **experimental lazy model-cache** is now enabled by default. To test it, restart the server after updating Git and run the same smoke command *twice*, without restarting Uvicorn between calls. Compare `model_reused`, the raw anomaly scores and `latency_ms_including_load`:

~~~bash
python -m scripts.local_api_smoke --url http://127.0.0.1:8001
python -m scripts.local_api_smoke --url http://127.0.0.1:8001
~~~

If repeat inference fails or results diverge, stop the server and set `export IHSANVISION_MODEL_CACHE=0` before restarting Uvicorn. That restores the previously verified **load-every-request** code path. The per-process model cache has only been tested against fake Engines in CI; real-checkpoint verification is pending. See [deployment notes](docs/deployment.md).

## Quality checks

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

CI tests configuration, split leakage, image handling, API validation, and simulated training/inference contracts. They **do not replace real model execution**, and do not require downloading datasets or pretrained weights.

## Technical documentation

- [Architecture and design decisions](docs/architecture.md)
- [Evaluation methodology and measured results](docs/evaluation.md)
- [Development and Kaggle workflow](docs/development.md)
- [API deployment and security notes](docs/deployment.md)
- [Engineering roadmap and acceptance criteria](docs/roadmap.md)
- [macOS live API smoke evidence](reports/2026-10-14-macos-api-smoke.md)
- [Trained PaDiM model card](docs/model-card.md)
- [Contributing guide](CONTRIBUTING.md)

## Limitations and responsible use

This project is an **engineering portfolio prototype**, not certified industrial quality-control software. Missing before production: model-serving lifecycle, authentication, rate limiting, observability, throughput profiling, repeatability across seeds/categories, PatchCore evaluation, calibrated localization, and live web-app verification.

The API accepts images, **not user-uploaded checkpoints**. Lightning/PyTorch checkpoints use serialization that may execute arbitrary code when deserialized: load only weights from trusted sources. MVTec AD images and model weights are **not included** in this repository. Consult the [dataset publisher](https://www.mvtec.com/company/research/datasets/mvtec-ad) for licensing and permitted use; no repository code license has yet been selected.

## References

- Bergmann et al., *The MVTec Anomaly Detection Dataset*, International Journal of Computer Vision (2021).
- [Anomalib](https://github.com/open-edge-platform/anomalib), open-source anomaly detection framework.
- Defard et al., PaDiM (2021); Roth et al., PatchCore (2022).
