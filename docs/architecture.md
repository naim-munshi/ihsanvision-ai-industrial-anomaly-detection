# System Architecture

## Purpose and boundaries

IhsanVision.AI is a prototype for **one-class industrial anomaly detection and localization**. A model learns a distribution of normal product appearances from MVTec AD training images and scores new images for anomalies. A separate service exposes predictions to a browser interface.

The codebase is intentionally divided into three responsibilities:

1. **Experiments** (Kaggle or a workstation): data validation, model fit, evaluation, checkpoint export and benchmark.
2. **Inference** (Python runtime): input validation, checkpoint loading, anomaly score, optional binary label and relative localization overlay.
3. **Presentation** (FastAPI and static HTML): image upload, API response handling, and user-facing visual review.

GitHub is the source of truth for code and documentation. The dataset and large checkpoints are **external artifacts**, not Git-tracked assets.

## Component diagram

~~~text
                      OFFLINE / EXPERIMENTS
         ┌─────────────────────────────────────────┐
         │ MVTec AD dataset                        │
         │  ├── train/good                          │
         │  └── test (normal + defect classes)      │
         └───────────┬──────────────────┬──────────┘
                     │                  │
                     v                  v
         ihsanvision.train / Anomalib    Disjoint val/test
                     │                  │
               Padim/Patchcore          v
                     │          evaluator metrics
                     v                  │
         trusted model.ckpt       report.json
                     │            splits.json
                     │
                     v
                      ONLINE / INFERENCE
  Uploaded image ──> ihsanvision.api ──> ihsanvision.serving
                             ├── validate inputs
                             ├── load trusted checkpoint
                             ├── Engine.predict
                             └── score / label / overlay
                                          │
                                          v
                                 demo/index.html
~~~

## File ownership

| File | Responsibility |
| --- | --- |
| `ihsanvision/config.py` | Category/model choices, experiment settings, model factory, dataset split auditing and metric serialization |
| `ihsanvision/train.py` | Training, output directory, split-manifest creation, checkpoint export and held-out test report |
| `ihsanvision/benchmark.py` | Sequential PaDiM/PatchCore runs with comparison JSON/CSV |
| `ihsanvision/predict.py` | Single-image inference and visualization |
| `ihsanvision/api.py` | FastAPI application, upload validation and error translation |
| `demo/index.html` | No-build client-side dashboard |
| `scripts/kaggle_preflight.py` | Detect and inspect a MVTec directory without running models |
| `scripts/kaggle_smoke_test.py` | End-to-end *single-image* model check with saved checkpoint and overlay |
| `tests/` | Unit/API and dependency-isolated integration-contract tests |

## Contracts and artifacts

**Training input:** An MVTec AD root containing `<category>/train/good`, `<category>/test/*` and pixel-level ground-truth masks when pixel metrics are needed.

**Experiment outputs:**

~~~text
<output_dir>/<model>-<category>-seed<seed>/
    model.ckpt           # serialized PyTorch/Lightning model
    report.json          # settings, versions, evaluation metrics
    splits.json          # explicit train/validation/test paths
    smoke_report.json    # only after optional real smoke test
    smoke_overlay.png    # only after optional real smoke test
    engine/              # Anomalib/Lightning runtime artifacts
~~~

**Inference input:** A supported image file and a matching *trusted* local checkpoint. A model choice (PaDiM or PatchCore) must match the checkpoint.

**Inference response:** A raw anomaly score (not calibrated probability), an optional boolean predicted anomaly, inference duration, and optionally a base64 PNG overlay. No bounding boxes or calibrated pixel-level segmentation claims are made.

**Service API:** `GET /health`, `GET /models`, `GET /`, and `POST /predict` (optional server-allowlisted `?model=padim|patchcore`). Local checkpoint paths come exclusively from server environment variables, never from browser inputs. See [development.md](development.md#configure-both-model-choices).

## Design decisions

### Separate experiment execution from serving

Training requires large data and pretrained feature-extractor weights, while serving needs a trained checkpoint. Keeping Kaggle out of the runtime path makes local or future cloud deployment possible without Kaggle credentials or mounted MVTec data.

### Keep evaluation auditable

MVTec AD does not have a separate validation directory by default. The current implementation creates a disjoint validation subset from its test folder, tests on the remaining images, writes all paths, and rejects path overlap. This is a portfolio evaluation protocol; it is **not** full-test benchmark reporting. Details: [evaluation.md](evaluation.md).

### Do not confuse visualization with segmentation

`ihsanvision.predict` normalizes the anomaly map to the image's own minimum/maximum and blends color for a human-readable localization view. It is **not** a globally calibrated mask. Raw score units are model-specific and should not be compared across models without normalization/calibration.

### Prototype limitations

- The current FastAPI process lazily loads a trusted checkpoint, then **reuses the model/Engine for repeated requests** under a per-process inference semaphore. The initial request includes restoration cost; subsequent warm requests do not normally reload model weights. The dashboard can select PaDiM or PatchCore only when a trusted checkpoint is configured on the server; changing models evicts the one cached model and incurs a reload. An environment flag (`IHSANVISION_MODEL_CACHE=0`) restores the original cold-loading path. Multi-worker and PatchCore cache behavior have not been validated on Mac.
- API has no authentication, rate limiting or queueing; external deployment is not yet advised.
- Experiment backend is Anomalib 2.5.0; upgrading versions may affect checkpoint compatibility.
- CPU-only Kaggle runs do not establish accelerator portability.
- Dataset assets and checkpoints are intentionally excluded by `.gitignore`.

## Planned architecture evolution

The prototype has a lazily initialized per-process model cache and validated Mac PaDiM UI inference. Next improve startup lifecycle management, structured logging, request IDs, bounded request queues, model/version metadata, metrics and authentication. Only then evaluate batch serving, export formats and external hosting. See [roadmap.md](roadmap.md).
