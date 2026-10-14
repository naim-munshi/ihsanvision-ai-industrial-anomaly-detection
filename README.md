# IhsanVision.AI

**Industrial anomaly detection research software** featuring a reproducible PaDiM/PatchCore evaluation pipeline, a FastAPI inference service, and a browser-based visual inspection dashboard.

> **Project status:** Research/portfolio prototype. The trained **PaDiM / MVTec AD bottle** model and local browser demonstration are verified. PatchCore training/comparison, security hardening and hosted production deployment remain unverified.

## What works

| Component | Evidence | Status |
| --- | --- | --- |
| PaDiM training and checkpoint evaluation | MVTec AD bottle, seed 42, held-out test subset | Verified on Kaggle |
| Image-level AUROC / F1 | **0.9963 / 0.9592** | Measured, restricted evaluation protocol |
| Pixel-level AUROC / F1 | **0.9768 / 0.6497** | Measured, restricted evaluation protocol |
| Prediction + relative heatmap | Saved checkpoint in Kaggle and macOS | Verified |
| FastAPI + responsive browser inspection | Normal and visibly defective bottle example | Verified locally |
| Warm model reuse | Same-process Mac PaDiM requests | Verified locally |
| Sequential inference latency | 10 reused requests, p50 **135.78 ms**, nearest-rank p95 **1,244.62 ms** | Preliminary, not a load test |
| PatchCore / PaDiM comparison | Reuse-safe experiment orchestration | Software implemented; real training pending |
| Automated code checks | Python 3.11 and 3.12 | GitHub Actions |

These performance and AUROC numbers come from **different protocols and purposes**. The dataset evaluation held out 16 images from MVTec's official test directory for validation, leaving **67 test images**. It is **not directly comparable with the full-test MVTec leaderboard**. The browser examples and synthetic API smoke input must not be presented as additional accuracy measurements.

## Quick start — local inspection (macOS/Linux)

**Requirements:** Python 3.11–3.13, a trusted Anomalib/Lightning `.ckpt` checkpoint, and adequate RAM for the model. PyTorch wheels differ across systems.

~~~bash
git clone https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
cd ihsanvision-ai-industrial-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

Until the restructuring is merged, choose `feature/portfolio-production-pipeline` when cloning, or run `git switch feature/portfolio-production-pipeline` and pull the branch. The repository's GitHub default setting may still be `MdNaim` during the migration.

Configure the **trusted** checkpoint outside Git (never upload a checkpoint through the HTTP API):

~~~bash
export IHSANVISION_CHECKPOINT="/absolute/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002
~~~

Open **http://127.0.0.1:8002/** for the interface and **http://127.0.0.1:8002/docs** for OpenAPI documentation.

From another terminal, run the synthetic API smoke check:

~~~bash
python -m scripts.local_api_smoke --url http://127.0.0.1:8002
~~~

This is a *service integration test*, not a defect accuracy test. See [developer workflow](docs/development.md) and [deployment/security notes](docs/deployment.md). If 8002 is occupied, inspect the listener with `lsof -nP -iTCP:8002 -sTCP:LISTEN` and choose a free port instead of killing an unknown process.

## Model experiments — Kaggle

**Existing PaDiM results should be preserved rather than retrained.** The saved run includes `model.ckpt`, `report.json` and `splits.json`. Dataset and weights must remain outside the source repository.

To train PatchCore without retraining PaDiM, follow the [incremental comparison procedure](docs/patchcore-comparison-workflow.md). Once the three trusted PaDiM artifacts are staged in the matching output folder, the command is:

~~~bash
python -m ihsanvision.benchmark \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --categories bottle --seed 42 \
  --output-dir /kaggle/working/ihsanvision_runs \
  --accelerator cpu --reuse-existing
~~~

Adjust the dataset path based on `python -m scripts.kaggle_preflight`. The benchmark checks **exact train/validation/test image paths before fitting PatchCore** and refuses missing or inconsistent PaDiM reference runs. Training and evaluation results must be collected from a real completed run; none are invented.

## Repository layout

~~~text
ihsanvision-ai-industrial-anomaly-detection/
├── ihsanvision/                 # Canonical Python application package
│   ├── api.py                   # FastAPI endpoints, upload handling
│   ├── schemas.py               # OpenAPI / Pydantic response contracts
│   ├── serving.py               # Cached trusted-checkpoint inference
│   ├── predict.py               # CLI inference, score + overlay
│   ├── train.py                 # Reproducible Anomalib experiment
│   ├── benchmark.py             # PaDiM/PatchCore comparison
│   └── config.py                # Models, configuration, split auditing
├── src/                         # Legacy import/CLI compatibility only
├── scripts/                     # Kaggle preflight, artifact staging, Mac smoke/latency checks
├── tests/                       # Fast mocked/offline regression tests
├── demo/index.html              # Local inspection UI
├── notebooks/                   # Historical exploratory notebooks
├── reports/                     # Observed experiment evidence and limitations
├── docs/                        # Architecture, evaluation, developer and deployment guides
├── .github/workflows/ci.yml     # Python 3.11/3.12 checks
├── pyproject.toml               # Python packaging / CLI entry points
├── requirements.txt             # Runtime dependency constraints
└── requirements-dev.txt         # Lightweight CI/test dependencies
~~~

The **canonical** imports and commands use `ihsanvision.*`. Older notebooks and workflows that use `src.train`, `src.predict`, `src.benchmark`, or `src.api` remain supported by **compatibility shims**; new features should not be added to those wrappers.

For an editable install, these command-line entry points are also available: `ihsanvision-train`, `ihsanvision-predict` and `ihsanvision-benchmark`.

## Development and testing

~~~bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m compileall -q ihsanvision src scripts tests
python -m pytest -q
~~~

CI validates packaging imports, canonical/legacy entry points, API contracts, leakage checks and offline inference behavior. It does **not** exercise a real model checkpoint or retrain on MVTec datasets. Running one CI test suite successfully does not imply production readiness.

## Project documentation

- [Architecture and design choices](docs/architecture.md)
- [Developer setup and execution](docs/development.md)
- [Evaluation methodology and measured results](docs/evaluation.md)
- [Trained PaDiM model card](docs/model-card.md)
- [Incremental Kaggle PatchCore comparison](docs/patchcore-comparison-workflow.md)
- [Deployment, limitations and security](docs/deployment.md)
- [Milestones and acceptance criteria](docs/roadmap.md)
- [Repository migration and branch conventions](docs/repository-structure.md)
- [Contribution guide](CONTRIBUTING.md)

### Supporting run records

- [Kaggle PaDiM bottle experiment](reports/2026-10-13-kaggle-padim-bottle.md)
- [Mac checkpoint-backed API smoke and cache](reports/2026-10-14-macos-api-smoke.md)
- [Mac ten-request inference latency sample](reports/2026-10-14-macos-10-request-inference-benchmark.md)
- [Normal and defective bottle UI screenshots — textual observation](reports/2026-10-14-browser-bottle-inspection.md)

## Safety, licensing and next steps

- Do not use a research-only anomaly score as a calibrated probability or a replacement for human quality-control decisions.
- Use only trusted checkpoint artifacts: serialized Lightning checkpoints should not be loaded from untrusted uploads or third parties.
- MVTec AD dataset rights are distinct from repository source rights. **No dataset images or model binaries are redistributed here.**
- A source-code license **has not yet been selected**; do not assume code is licensed for unrestricted reuse.
- Before hosting publicly: implement appropriate authentication, rate limits, request-admission controls, observability and deployment isolation.
- Pending: real PatchCore evaluation, multiple-seed/robustness analysis, error-state UI QA, and a reviewed portfolio release.

The stable `main` branch and short-lived `feature/*` development workflow are the intended long-term Git conventions; branch-default changes are tracked separately from code refactoring.
