# Independent Linux CPU image-build proof (without the owner's Mac)

The lightweight `container-config` CI checks only YAML. The separate,
heavier [CPU Docker build and HTTP smoke workflow](../.github/workflows/docker-cpu-smoke.yml)
attempts to **actually build** the Python 3.12 Linux application image using
the same Dockerfile used in `compose.yaml`, then starts the real Uvicorn API
in a Linux runner with:

- Nonroot UID 10001 and read-only container root.
- All capabilities dropped, `no-new-privileges`, finite PIDs, in-memory /tmp.
- Host port bound to **127.0.0.1 only**.
- A disposable **CI-only** API key, never a real operator secret.

The workflow asserts `GET /health` is live, `GET /models` has no fake
checkpoint entries, `GET /ready` returns **503** while no trusted weights
are available, the three operator dashboard HTML routes are served and
unauthenticated `POST /predict` fails with HTTP 401. It also checks the
built image user and absence of accidentally packed checkpoint/secret formats.

The build may download large Anomalib/PyTorch dependencies and is **not
included in every lightweight Python unit-test run**. Trigger manually in
GitHub Actions via **CPU Docker build and HTTP smoke (no checkpoints) → Run
workflow**, or review PR runs changing the Docker packaging workflow.

### What a successful cloud build can and cannot prove

**Can prove:** A real Linux application image builds on the GitHub runner;
Uvicorn starts inside that container, API routing/readiness/auth are exercised
over HTTP, and Docker confinement is verified in a live Linux container.

**Cannot prove:** GPU/Apple Silicon image compatibility, Anomalib
checkpoint deserialization, model inference/heatmap, private model-mount
rights, real end-to-end latency, restart persistence, production uptime,
security under public traffic, or P8 **fully complete**. The owner's real
Mac/Docker + trusted Bottle checkpoint acceptance is still mandatory. No
weights, MVTec images or user secrets are available to the cloud workflow.

When a build succeeds, record its **exact GitHub Actions run URL**, Git
SHA, runner architecture and summary in a dated report. Do not report
success based on the presence of this workflow alone.
