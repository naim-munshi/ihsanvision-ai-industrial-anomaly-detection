# Service operation, API and security

> **Current source of truth:** [portfolio v1.0 scope](portfolio-v1-plan.md),
> [real local acceptance](operational-acceptance.md),
> [local Docker setup](container-deployment.md), and
> [security baseline](security-baseline.md).
> This is a **localhost AI engineering portfolio prototype**, not a hosted,
> certified or public multi-tenant production service.

## Implemented application

- Single-image `POST /predict` with server-configured PaDiM or PatchCore
  checkpoints and bounded input validation; responses include an uncalibrated
  raw anomaly score, model decision and relative heatmap.
- `GET /models` lists configured models **without leaking checkpoint paths**.
  `GET /health` tests process liveness; `GET /ready` checks model
  configuration and, when enabled, exact operator-approved SHA-256 pins.
  Readiness **does not execute inference**.
- `/` is the single-image dashboard, `/batch` handles an operator-selected
  image batch and `/history` exposes opt-in private inspection history,
  human review, lot/model-version reporting and private CSV export.
- Shared-key authentication when configured, pre-multipart upload
  authentication and body-size guarding, bounded process-local admission,
  in-memory model reuse and optional private operator timing telemetry.
- Operator-controlled private model releases and manual rollback; no arbitrary
  model checkpoint supplied by the HTTP client is loaded.

All recent operator workflow/API protections have **offline regression and
mocked integration tests**. The earlier real Mac PaDiM/PatchCore Bottle
inference evidence is documented separately in
[the dated dual-model report](../reports/2026-10-16-macos-dual-model-api-acceptance.md).
Do not confuse those earlier real observations with current-code end-to-end
operator acceptance.

## Run locally

A trusted trained Bottle checkpoint for both model families is required:

```bash
python -m scripts.run_local_demo --no-browser
# Then open http://127.0.0.1:8003/
```

See [developer setup](development.md) and
[exact owner handoff](../FINISHING.md) for environment and private artifact
paths. The recommended local test uses
`python -m scripts.local_stack_acceptance` with one independently labeled
normal image and one defective image. For an authenticated disposable
history database, the optional `--require-auth --operator-workflow` mode
checks a real private inspection → human verdict → lot/model report →
metadata CSV → deletion sequence.

**Never publish operator keys, private database contents, MVTec images,
model binaries or raw private image paths.** Even when authentication is
enabled, localhost operation does not establish public-internet security.

## Container evidence

The [Linux CPU Docker Actions workflow](docker-cloud-build-evidence.md)
actually built the image and exercised real HTTP routes, private-key
refusal, readiness failure without weights, model-class construction and
installed dependency compatibility in GitHub CI. It did **not** run an
actual trained checkpoint or produce a defect heatmap in Docker.

Before calling portfolio **P8** complete, the owner must run the target
container with locally trusted read-only Bottle weights, validate both
models' actual predictions and heatmaps, then check startup/restart,
resource use, nonroot/read-only constraints and recovery. Follow
[local Docker acceptance instructions](container-deployment.md).

## Production and safety boundaries

The repository does **not** currently claim public TLS/gateway hardening,
individual operator authentication and role-based access, industrial
actuator safety certification, distributed quota guarantees, durable
observability or camera-line integration. These are outside the fixed
portfolio v1.0 scope.

For scientific limitations read the [model card](model-card.md) and
[evaluation protocol](evaluation.md). Raw anomaly scores are not
probabilities; heatmaps are not calibrated defect masks.
