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

See the [Kaggle PatchCore comparison workflow](patchcore-comparison-workflow.md) for restoring trusted artifacts from the private user-owned Kaggle input and running only the missing PatchCore job. No real PatchCore output is claimed yet.

### Preserve data and models

Kaggle working storage may be reset between sessions. **Before closing the session**, save a notebook version and export the run artifacts from `/kaggle/working/ihsanvision_runs/` to storage you control. At minimum preserve:

- `model.ckpt` (large binary; keep private or use appropriate model artifact storage)
- `report.json` and `splits.json`
- `smoke_report.json` and `smoke_overlay.png` if created

Do not push MVTec images or the checkpoint binary into this Git repository. Do not assume Kaggle artifacts automatically appear on your Mac.

## Local development

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
