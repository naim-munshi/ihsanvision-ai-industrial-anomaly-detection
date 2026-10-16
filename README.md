# IhsanVision.AI

**Industrial anomaly detection platform in development** featuring reproducible PaDiM/PatchCore training and evaluation, a FastAPI inference service, a browser inspection dashboard and an operator-controlled registry for additional trained product categories.

> **Project status:** Research/portfolio prototype. PaDiM and PatchCore **bottle** experiments have completed in Kaggle; the saved PatchCore checkpoint passed one held-out defective-image inference smoke test. Both model checkpoints and comparison evidence were backed up to macOS and passed SHA-256 checks against a bundled manifest. Independent JSON-content audit, broader validation, security hardening and hosted production deployment remain pending.

## What works

| Component | Evidence | Status |
| --- | --- | --- |
| PaDiM training and checkpoint evaluation | MVTec AD bottle, seed 42, held-out test subset | Verified on Kaggle |
| Image-level AUROC / F1 | **0.9963 / 0.9592** | Measured, restricted evaluation protocol |
| Pixel-level AUROC / F1 | **0.9768 / 0.6497** | Measured, restricted evaluation protocol |
| Prediction + relative heatmap | Saved checkpoint in Kaggle and macOS | Verified |
| FastAPI + responsive browser inspection | Normal and visibly defective bottle example | Verified locally |
| Warm model reuse | Same-process Mac PaDiM requests | Verified locally |
| Sequential inference latency | Both models profiled with same 900×900 image, 3 warmups + 20 cached requests each; HTTP p50 **388.87 ms PaDiM / 493.05 ms PatchCore**, p95 **766.46 / 514.82 ms** | Real Mac user-provided paired runs; not throughput or an SLA |
| PatchCore / PaDiM comparison | PatchCore image AUROC **1.0000**, image F1 **0.9901**, pixel AUROC **0.9857**, pixel F1 **0.7102** | Completed Kaggle log; macOS artifact checksum backup verified |
| PatchCore checkpoint smoke test | Held-out defective bottle, anomaly detected, overlay generated | Verified in Kaggle; smoke evidence ZIP downloaded and integrity-tested on Mac |
| Real Mac dual-model dashboard | Both models predicted normal for `normal_bottle.png` and flagged `defective_bottle.png`; overlays rendered for both | Four Safari observations **and four-request API acceptance passed**; source split provenance pending |
| Bounded local PaDiM concurrency | 2 parallel HTTP submissions, 10 measured requests: **10/10 HTTP 200**, **0 failures**, 10/10 cached; HTTP p50/p95 **633.59/1,019.31 ms** | Observed on Mac; no maximum-QPS/SLA claim |
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

### Multi-category trained model routing

Beyond the verified `bottle` models, the application can now register **trusted, independently trained checkpoints per industrial product category and model family**. The API and browser display only configured categories; untrained classes are not advertised as available. The original scope includes 15 MVTec AD categories plus selected MVTec AD 2 categories; **only the bottle models have experimental evidence in this repository so far**. See the [multi-category registry setup, API and evaluation gates](docs/multicategory-model-registry.md). Additional categories must be trained and scientifically verified before claiming coverage.

### Launch the complete local dual-model demo (one command)

Once the runtime dependencies are installed and both **personally trusted** checkpoint files are stored in the documented private artifact directory, run:

~~~bash
python -m scripts.run_local_demo
~~~

The launcher checks that `padim-bottle-seed42/model.ckpt` and `patchcore-bottle-seed42/model.ckpt` exist under `~/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/`, configures both models, starts a loopback-only API at **http://127.0.0.1:8003/** and opens the dashboard when ready. **Ctrl+C stops the server**. It does not train or download data or change model artifacts. Override the private model folder with `--artifacts-dir /absolute/private/folder`, use `--port 8004` if the default is already in use, or `--no-browser` on headless systems. Never load untrusted Lightning checkpoints or publish the localhost service to the internet.

This reduces startup to one command; real Mac end-to-end launcher verification is still required before calling its operation verified on the user's device.

**Repository default branch: `main` (verified 2026-10-14).** New contributors should use `main` and create focused `feature/*` branches. The historical `MdNaim` and integrated feature branches are kept temporarily for local checkout migration and audit history.

Configure the **trusted** checkpoint outside Git (never upload a checkpoint through the HTTP API):

~~~bash
export IHSANVISION_CHECKPOINT="/absolute/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002
~~~

Open **http://127.0.0.1:8002/** for the interface and **http://127.0.0.1:8002/docs** for OpenAPI documentation.

### Optional: select between PaDiM and PatchCore on the dashboard

To make both **locally trusted** checkpoints available in the model dropdown, use these server-side environment variables instead of the legacy single-checkpoint variable:

~~~bash
export IHSANVISION_PADIM_CHECKPOINT="$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/padim-bottle-seed42/model.ckpt"
export IHSANVISION_PATCHCORE_CHECKPOINT="$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/patchcore-bottle-seed42/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002
~~~

Run Uvicorn from a checkout with this feature (after merging, from updated `main`). The existing `IHSANVISION_CHECKPOINT` setting remains supported for single-model users. `GET /models` publishes available model **names only**, never local paths. The browser sends `POST /predict?model=padim` or `?model=patchcore`; unconfigured models are refused. Switching between model types incurs checkpoint reload in the current one-session cache, so first-request latency can be higher. Model scores are uncalibrated and **must not be compared directly across model families**.

### Real normal/defective two-model acceptance (M6)

After configuring both trusted checkpoints and starting the local API, provide **two locally available sample images** and capture four real API predictions without retraining. Run from the current code checkout with the same Python virtual environment used to start Uvicorn:

~~~bash
python -m scripts.dual_model_acceptance \
  --url http://127.0.0.1:8003 \
  --normal "/absolute/path/to/normal_bottle.png" \
  --defective "/absolute/path/to/defective_bottle.png" \
  --output "$HOME/ihsanvision-artifacts/dual_model_acceptance.json"
~~~

The script checks that both models are available, each model predicts both images, model routing and PNG overlays are valid, and the normal/defective labels match the manually supplied examples. The non-overwriting JSON report saves sample SHA-256 hashes and compact results **without** base64 images, checkpoint paths or model weights. A missed defect causes a **nonzero exit code** and is recorded as a result, not silently counted as a pass. This is a four-prediction qualitative acceptance check, **not** a claim about accuracy, held-out test membership, calibration or p95 latency. For additional detail, see [M6 acceptance procedure](docs/development.md#m6-real-dual-model-acceptance).

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
- [Operator-configured multi-category model registry](docs/multicategory-model-registry.md)
- [Developer setup and execution](docs/development.md)
- [Evaluation methodology and measured results](docs/evaluation.md)
- [Trained PaDiM model card](docs/model-card.md)
- [Incremental Kaggle PatchCore comparison](docs/patchcore-comparison-workflow.md)
- [Verified PaDiM/PatchCore summary generator](docs/verified-benchmark-summary.md)
- [Experiment environment snapshot](docs/environment-provenance.md)
- [Sequential localhost serving profiler and methodology](docs/serving-benchmark.md) — now includes a limited concurrent load probe
- [Inference API security baseline and limitations](docs/security-baseline.md)
- [Local-only Docker Compose packaging and verification gates](docs/container-deployment.md)
- [Read-only benchmark/smoke archive audit and portfolio release gates](docs/portfolio-release-readiness.md)
- [Deployment, limitations and security](docs/deployment.md)
- [Milestones and acceptance criteria](docs/roadmap.md)
- [Repository structure and branch conventions](docs/repository-structure.md)
- [Mac checkout, SSL installation and test troubleshooting](docs/mac-environment-troubleshooting.md)
- [Contribution guide](CONTRIBUTING.md)

### Supporting run records

- [Kaggle PaDiM bottle experiment](reports/2026-10-13-kaggle-padim-bottle.md)
- [Observed Kaggle PatchCore bottle result and limitations](reports/2026-10-15-kaggle-patchcore-bottle.md)
- [Mac checkpoint-backed API smoke and cache](reports/2026-10-14-macos-api-smoke.md)
- [Mac ten-request inference latency sample](reports/2026-10-14-macos-10-request-inference-benchmark.md)
- [Normal and defective bottle UI screenshots — textual observation](reports/2026-10-14-browser-bottle-inspection.md)
- [macOS dual-model dashboard normal-image observation](reports/2026-10-15-macos-dual-model-dashboard-normal-ui.md)
- [Real Mac four-request dual-model acceptance](reports/2026-10-16-macos-dual-model-api-acceptance.md)
- [Real Mac paired warm-serving latency profile](reports/2026-10-16-macos-paired-serving-profile.md)
- [Real Mac PaDiM finite concurrency-2 result](reports/2026-10-16-macos-padim-concurrent-probe.md)

## Safety, licensing and next steps

- Do not use a research-only anomaly score as a calibrated probability or a replacement for human quality-control decisions.
- Use only trusted checkpoint artifacts: serialized Lightning checkpoints should not be loaded from untrusted uploads or third parties.
- MVTec AD dataset rights are distinct from repository source rights. **No dataset images or model binaries are redistributed here.**
- A source-code license **has not yet been selected**; do not assume code is licensed for unrestricted reuse.
- For local demos, optional shared API-key authentication, per-process rate limiting, bounded admission and scrubbed request-id audit records are implemented. These **do not** replace TLS, gateway body limits, distributed quotas, identity/tenant authorization or container/network isolation. See the [security baseline](docs/security-baseline.md). Do not expose the local prototype publicly.
- Pending: original sample split/provenance validation and M6 error-state QA (formal four-request Mac acceptance passed); **M7 PatchCore concurrent, admission, repeated and memory profiling** (paired saved-JSON audit and small PaDiM concurrency-2 run observed on Mac), **M10 private evidence verification**, actual M9 Docker build/inference, multi-seed robustness, and a user-approved license/release. Packaging and a release gate are implemented, **not a successful hosted/public release**. Real Mac model selection and normal-image inference have been observed, not just mocked.

The default branch is `main`. Future changes follow `main` → short-lived `feature/*` or `chore/*` branch → pull request → CI → `main`.
