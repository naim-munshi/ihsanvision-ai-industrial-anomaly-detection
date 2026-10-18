# Developer Workflow

## Overview

**Kaggle** is the compute workspace for experiments and dataset access. **GitHub** is the source-of-truth repository. **Local development** (VS Code/macOS/Linux) is where the service and UI should be validated. A remote deployment can be added after security and performance requirements are met.

The tested portfolio source tree was integrated through [merged pull request #2](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/pull/2). GitHub's default branch is now **`main` (verified 2026-10-14)**. Create focused `feature/*` or `chore/*` branches from `main` and submit PRs for review.

## Kaggle experiments

### Fresh Kaggle session

1. Use a Python Kaggle notebook and attach MVTec AD from **Add Input**.
2. Clone the default `main` branch:

~~~python
!git clone https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
%cd /kaggle/working/ihsanvision-ai-industrial-anomaly-detection
!python -m pip install -r requirements.txt
~~~

3. Use a fresh runtime after dependency updates if already-imported modules become inconsistent. Kaggle has many preinstalled packages; resolver warnings about unrelated packages do not necessarily imply the Anomalib pipeline has failed.
4. Locate the dataset:

~~~python
!python -m scripts.kaggle_preflight
~~~

5. Run a training experiment only if no checkpoint exists or you need a controlled new run:

~~~python
!python -m ihsanvision.train \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --category bottle --model padim \
  --output-dir /kaggle/working/ihsanvision_runs \
  --accelerator cpu --seed 42
~~~

6. Verify inference with saved outputs:

~~~python
!python -m scripts.kaggle_smoke_test \
  --dataset-root /kaggle/input/datasets/ipythonx/mvtec-ad \
  --output-dir /kaggle/working/ihsanvision_runs
~~~

7. Display the overlay inside the notebook:

~~~python
from IPython.display import display, Image
display(Image(filename="/kaggle/working/ihsanvision_runs/padim-bottle-seed42/smoke_overlay.png"))
~~~

### Continue an existing Kaggle notebook

If you previously cloned the old **feature** branch, you can update the existing checkout and use its compatible code; for new experiments, use `main` after fetching it instead of cloning multiple copies:

~~~python
%cd /kaggle/working/ihsanvision-ai-industrial-anomaly-detection
!git branch --show-current
!git pull --ff-only
~~~

The old feature branch remains temporarily for audit. **Existing single-branch clones** may need an explicit remote-tracking ref first: `git fetch origin main:refs/remotes/origin/main`, then `git switch -c main --track origin/main` (if the local `main` does not already exist). A failed `git switch main` leaves you on the previous branch; do not assume a subsequent `git pull` updates `main`.

### Incremental PatchCore experiment (do not rerun PaDiM)

The completed PaDiM bottle experiment should be reused. The `ihsanvision.benchmark --reuse-existing` mode requires its saved `model.ckpt`, `report.json`, and `splits.json`, checks split equivalence **before** new PatchCore training, and refuses partial or mismatched experiments.

The Bottle PatchCore job **has completed** and its saved results were documented in [the observed Kaggle run](../reports/2026-10-15-kaggle-patchcore-bottle.md). Keep both Bottle checkpoints; use the [safe resumable runner](resumable-multicategory-training.md) for the portfolio's additional cable/carpet jobs. The old PatchCore comparison workflow is retained as historical reproduction guidance.

### Never overwrite an interrupted direct training run

Direct `python -m ihsanvision.train` now **requires a completely new**
`<model>-<category>-seed<seed>` output directory. The process refuses any
existing run directory, even if it contains only an interrupted `engine/`,
`model.ckpt`, `splits.json`, or an incomplete `report.json`. No artifacts
are deleted or overwritten. Before a deliberate new experiment, select a
fresh output directory or seed; preserve and investigate partial artifacts.
The multi-category resumable planner separately inspects completed triples
and schedules only verified missing runs.

### Preserve data and models

Kaggle working storage may be reset between sessions. **Before closing the session**, save a notebook version and export the run artifacts from `/kaggle/working/ihsanvision_runs/` to storage you control. At minimum preserve:

- `model.ckpt` (large binary; keep private or use appropriate model artifact storage)
- `report.json` and `splits.json`
- `smoke_report.json` and `smoke_overlay.png` if created

Do not push MVTec images or the checkpoint binary into this Git repository. Do not assume Kaggle artifacts automatically appear on your Mac.

## Local development

### One-command demo startup

In an existing checkout with runtime dependencies installed and both trusted model checkpoints archived locally, run:

~~~bash
python -m scripts.run_local_demo
~~~

This binds only `127.0.0.1:8003`, opens the browser after localhost readiness, and refuses missing/empty model files or an occupied port without killing other processes. Override the default model archive directory with `--artifacts-dir`; use `--port 8004` for a free port; add `--no-browser` to suppress the new tab. The Python process must stay running while the dashboard is used; press Ctrl+C to stop it. This is a developer/local demonstration, not a public deployment or a substitute for the M8 security controls.


~~~bash
git clone https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
cd ihsanvision-ai-industrial-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Install the local package namespace in editable mode after the dependency installation:

~~~bash
python -m pip install -e . --no-deps
~~~

The current application code resides in `ihsanvision/`. Older `src.*` imports/CLI entry points remain available as compatibility shims (see [repository migration](repository-structure.md)).

Use a compatible PyTorch wheel for the target environment. Installation and first model initialization may need internet to download pretrained feature extraction weights.

If a Mac checkout was originally cloned from a single feature branch, or pip reports an SSL certificate error, see [non-destructive Mac troubleshooting](mac-environment-troubleshooting.md). The `--no-deps` flag does not disable build-isolation downloads; a suitable locally installed setuptools/wheel may allow `--no-build-isolation` instead.

### Lightweight tests (no model weights)

~~~bash
python -m pip install -r requirements-dev.txt
python -m compileall -q ihsanvision src scripts tests
python -m pytest -q
~~~

### Validate local artifacts before starting the server

~~~bash
python -m scripts.local_preflight
~~~

The script verifies the current environment and artifact files, checks for duplicate/overlapping split paths, and compares recorded split counts. It reads JSON, but **does not deserialize checkpoint bytes or run the model**.

### API and UI

Place or point to a **trusted** checkpoint outside version control:

~~~bash
export IHSANVISION_CHECKPOINT="/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL="padim"
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002
~~~

Open `http://127.0.0.1:8002/`. To smoke-test the running server from a separate Terminal:

~~~bash
cd ~/ihsanvision-portfolio
source .venv/bin/activate
python -m scripts.local_api_smoke --url http://127.0.0.1:8002
~~~

Use the **same port** for Uvicorn, the dashboard and the smoke script. If port 8002 is occupied, choose another free local port consistently. The script makes actual HTTP requests and validates a synthetic PNG prediction; **the synthetic input is not a meaningful model-quality test**. Passing a real image via `--image /path/to/sample.png` is supported. Actual Mac PaDiM checkpoint inference, repeat requests, and real normal/defective bottle UI examples have now been observed; see the [Mac evidence](../reports/2026-10-14-macos-api-smoke.md) and [browser UI observation](../reports/2026-10-14-browser-bottle-inspection.md).

Blocking inference runs in a threadpool; a per-process semaphore serializes model execution while a trusted checkpoint can be reused within the same server process.

#### Configure both model choices

The dashboard discovers available checkpoints using `GET /models`. Set **trusted local paths on the server** (never in a browser request):

~~~bash
export IHSANVISION_PADIM_CHECKPOINT="$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/padim-bottle-seed42/model.ckpt"
export IHSANVISION_PATCHCORE_CHECKPOINT="$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/patchcore-bottle-seed42/model.ckpt"
export IHSANVISION_MODEL=padim
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002
~~~

Now the UI displays **PaDiM** and **PatchCore** if both files exist. `GET /models` returns `available_models` and `default_model` without checkpoint paths, and `POST /predict?model=patchcore` loads the server-configured PatchCore weights. Without a model query, `POST /predict` uses the preferred available default. `IHSANVISION_CHECKPOINT` with `IHSANVISION_MODEL` still supports the original single-model launch; explicit per-model checkpoint settings take precedence, and invalid/missing configured files remain unavailable. A request for an unsupported model returns HTTP 400; a supported but unconfigured model returns HTTP 503.

The current in-process cache keeps **one checkpoint/model pair**; switching resets it, trading loading latency for bounded model memory. Keep the API bound to localhost: local API-key/auth, rate and upload limits are implemented, but TLS, identity and public deployment checks are not complete. Score units/thresholds can differ across model families; do not compare raw scores directly. CI checks API routing and operator workflows using mocked predictors. Real Bottle PaDiM/PatchCore browser and four-request HTTP inference **did pass on Mac** in [the October 16 evidence](../reports/2026-10-16-macos-dual-model-api-acceptance.md); the newest history/export/security revisions still need fresh real-Mac integration.

#**Bottle launcher safety:** `python -m scripts.run_local_demo` fails early if
an inherited `IHSANVISION_MODEL_REGISTRY` could override the two Bottle
checkpoints passed with `--artifacts-dir`. Explicitly remove that variable
for the two-model demo, or launch the general API with your reviewed registry.
It does not disable optional authentication or approval pins.

## M6 real dual-model acceptance

Use a Mac terminal with an active Python environment where Pillow and the source package are installed. Start the two-checkpoint local API on port 8003 in a **separate terminal** using the configuration above.

~~~bash
python -m scripts.dual_model_acceptance \
  --url http://127.0.0.1:8003 \
  --normal "/absolute/path/to/normal_bottle.png" \
  --defective "/absolute/path/to/defective_bottle.png" \
  --output "$HOME/ihsanvision-artifacts/dual_model_acceptance.json"
~~~

The check sends **four POST requests**: PaDiM normal, PaDiM defective, PatchCore normal, PatchCore defective. It requires correct model routing, finite raw scores, boolean predictions, valid full-size PNG overlays and expected image labels. It also records request-reported duration and cache-reuse fields, but **does not** calculate controlled performance statistics.

- Both input files must exist and have different bytes. Their raw file SHA-256 hashes are recorded in private JSON evidence; image bytes and overlays are not copied into the JSON.
- An unexpected label marks the report `label_mismatch` and returns exit code 1 after recording all four predictions. Do not ignore this outcome or tune to the held-out test set.
- The script **does not** establish whether these manually selected files belong to the saved train/validation/test split. Their source paths and pixel contents require a separate data-provenance audit.
- Run against `http://127.0.0.1:<port>` or `http://localhost:<port>` only. Do not use publicly hosted URLs, and do not commit demonstration images or large private reports to Git.
- `--output` never overwrites an existing report. Change the file name for a second run.
- Manually inspect rendered dashboard heatmaps for localization plausibility and test error states separately. PNG format/size validation is not an intersection-over-union or segmentation-quality test.

A real four-request run of this script **passed on Mac on 2026-10-16**; see [the dated acceptance report](../reports/2026-10-16-macos-dual-model-api-acceptance.md). The script's offline regression tests separately validate the client contract. The newer full local-stack acceptance runner still needs a fresh operator execution.

## Contribution procedure

1. Keep changes in a focused branch; submit a PR to `main` and avoid committing to legacy branches.
2. Describe how a change impacts model outputs, metrics or API contracts.
3. Add/adjust tests for new behavior.
4. Run offline CI before merging; use actual dataset checks for model-path changes.
5. Include commands, environment, sample artifacts and known limitations in PR description.
6. Never commit datasets, training weights, tokens, credentials, or fabricated metrics.

## Troubleshooting

| Symptom | Likely cause | Next check |
| --- | --- | --- |
| Missing `bottle/train/good` | Wrong Kaggle mount path | `python -m scripts.kaggle_preflight` |
| `ModuleNotFoundError` | Environment or repo current directory | Check `pwd`, dependency install and runtime restart |
| Missing `model.ckpt` | Not trained, wrong output directory or reset Kaggle session | Examine `/kaggle/working/ihsanvision_runs` and saved notebook outputs |
| API returns HTTP 503 | No configured trusted checkpoint | Check `IHSANVISION_CHECKPOINT` and filename |
| Pixel F1 below AUROC | Thresholded mask errors despite good score ranking | Investigate masks, calibration and visual overlays |
| CUDA/MPS errors | Accelerator or PyTorch wheel mismatch | Try CPU first and use official PyTorch installation guidance |

## Further reading

[Architecture](architecture.md) · [Evaluation](evaluation.md) · [Deployment](deployment.md) · [Roadmap](roadmap.md).
