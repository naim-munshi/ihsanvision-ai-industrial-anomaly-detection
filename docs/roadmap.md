# Engineering Roadmap

This is a **state-based roadmap**, not an assertion that all systems are already production-ready. Completion requires observable evidence.

## Milestones

| Stage | Description | State | Acceptance evidence |
| --- | --- | --- | --- |
| M1 | Source layout, dependencies, README, code hygiene | **Implemented** | Documented modules and reproducible commands |
| M2 | PaDiM training and checkpoint-backed held-out evaluation | **Verified on Kaggle** | Completed notebook run and metric report |
| M3 | Single-image model inference + heatmap | **Verified on Kaggle** | `single-image-inference-passed`, overlay screenshot |
| M4 | CI and dependency-isolated tests | **Verified** | Python 3.11/3.12 GitHub Actions pass |
| M5 | PatchCore comparison on same dataset/protocol | **Kaggle run and smoke passed; Mac archived benchmark audit succeeded** | Eight hashes OK; generated summary from saved comparisons/reports/manifests; separate smoke ZIP integrity passed |
| M6 | Real FastAPI + browser demo against trusted model | **PaDiM normal/defective and dual-model normal-image UI demonstrated** | New Mac screenshots: both model options predict normal with heatmaps on same displayed filename; defective dual-model follow-up and sample split audit pending |
| M7 | Persistent serving, tests and performance profiling | **Sequential profiler implemented; real Mac execution pending** | Historical 10-request PaDiM sample; new repeatable p50/p95 HTTP client profiler with mocked CI coverage, no concurrency or memory results yet |
| M8 | Authentication, rate limit, audit logging | **Local-process baseline implemented; deployment validation pending** | Optional shared API key, global per-process rate cap, finite admission and request ID logs tested offline; identity, proxy/TLS, secrets and load tests pending |
| M9 | Container packaging and hosted demo | **Local-only Docker packaging added; build and hosting pending** | Dockerfile + Compose, readonly model mounts and static guardrail tests; real Docker inference, remote TLS/gateway and rollback are unverified |
| M10 | Portfolio release | **Pending** | Demonstration, figures, reproducibility package and review |

## Immediate priorities

### Priority 1 — Preserve verified model artifacts

Export the Kaggle-generated `model.ckpt`, `report.json`, `splits.json`, `smoke_report.json` and `smoke_overlay.png` from the session. Avoid re-running an already successful experiment unless needed for repeatability.

**Definition of done:** The checkpoint is stored safely outside Git; the experiment JSONs are reviewable; no dataset redistribution.

### Priority 2 — PatchCore benchmark

The user-observed Kaggle [PatchCore/bottle run](../reports/2026-10-15-kaggle-patchcore-bottle.md) has now completed after preserving the prior PaDiM checkpoint with `--reuse-existing` and checking exact train/validation/test paths. Results are measured in the console log. The user has **downloaded and extracted** a 389.04 MB ZIP on macOS, with eight artifact hashes successfully matching the included `SHA256SUMS.txt`. A held-out defect-image PatchCore checkpoint smoke test also returned `single-image-inference-passed`. The user's Mac run of `scripts.summarize_comparison` produced `benchmark_summary.md` successfully from the saved reports and comparison files; it checks exact manifest/metric consistency without deserializing checkpoints. The newer smoke report/overlay were also downloaded separately and their ZIP passed `unzip -t`. An independent scientific replication remains future work. Do not rerun successful training to regenerate metrics.

**Definition of done:** Real PatchCore report; complete comparison CSV/JSON; clear hardware/split provenance.

### Priority 3 — Live app inspection

The Mac preflight, checkpoint-backed HTTP API smoke test and a pair of cached repeated calls already passed. A 10-request same-image latency sample is now recorded (all requests marked `model_reused=true`; p50 135.78 ms, p95 1244.62 ms due to one spike). The user provided screenshots from a real-browser normal and defective bottle test showing expected labels and localization images ([UI evidence](../reports/2026-10-14-browser-bottle-inspection.md)). A newer [dual-model normal-image browser test](../reports/2026-10-15-macos-dual-model-dashboard-normal-ui.md) additionally verified `GET /models` registration and both real checkpoint-backed model choices with displayed normal predictions and heatmaps. A repeatable [dual-model local API acceptance harness](../docs/development.md#m6-real-dual-model-acceptance) now exists, with offline unit coverage; its four real-checkpoint requests must still be executed on the Mac. Next, test **the same defective image with both models**, validate a subset explicitly within the saved held-out test manifest, and check error-state handling. Broader latency/throughput testing remains pending. A repeatable [sequential serving profiler](serving-benchmark.md) now separates warmup, client round-trip and model-internal prediction durations; real Mac runs and concurrent profiling remain outstanding. Resolve mismatches between CLI and web behavior.

**Definition of done:** Browser upload and score/label/overlay have been observed for normal/defective examples. Error-state coverage, traceable held-out sample provenance and release screenshot packaging remain pending.

### Priority 4 — Improve engineering robustness

A persistent model instance is implemented and observed working with real PaDiM. The `/predict` response now has a Pydantic/OpenAPI schema that rejects invalid scores and latency. Initial request IDs and scrubbed audit logs, optional shared-key authentication, rate cap and admission timeout are implemented in the [local security baseline](security-baseline.md). Continue with gateway upload controls, typed settings, full trace/profiling, account-aware quotas, privacy and cloud-ready operations. Expand integration tests to real-model checks in a dedicated environment.

**Definition of done:** Repeatable p50/p95 load benchmark with documented CPU/GPU and concurrency; no unsafe checkpoint upload path.

### Priority 5 — Portfolio finalization

Consolidate measured results, demo screenshots, software architecture, limitations, reproducibility guide, and a short demonstration video. [Docker packaging](container-deployment.md) now provides a local-only build path but must be built and tested before hosting claims. Select an intentional source-code license after reviewing dependency/dataset terms.

**Definition of done:** Clean GitHub release, green CI, reviewed documentation and a repeatable live demo.

## Release policy

Maintain the active work in a feature branch/PR until end-to-end verification. Do not claim "production-grade", "99% defect detection", "certified industrial inspection", or published benchmark superiority solely from a single category with a validation holdout.

## Source of truth

- [README](../README.md)
- [Evaluation report](../reports/2026-10-13-kaggle-padim-bottle.md)
- [Evaluation methodology](evaluation.md)
- [Deployment and security requirements](deployment.md)
