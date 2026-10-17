# Engineering Roadmap

This is a **state-based roadmap**, not an assertion that all systems are already production-ready. Completion requires observable evidence.

## Milestones

| Stage | Description | State | Acceptance evidence |
| --- | --- | --- | --- |
| M1 | Source layout, dependencies, README, code hygiene | **Implemented** | Documented modules and reproducible commands |
| M2 | PaDiM training and checkpoint-backed held-out evaluation | **Verified on Kaggle** | Completed notebook run and metric report |
| M3 | Single-image model inference + heatmap | **Verified on Kaggle** | `single-image-inference-passed`, overlay screenshot |
| M4 | CI and dependency-isolated tests | **Verified** | Python 3.11/3.12 GitHub Actions pass |
| M5b | Safe resumable multi-category training | **Dry-run/explicit bounded runner implemented; additional real category training pending** | Validate source datasets before any fits, reuse paired checkpoints, reject partial outputs and unequal split manifests; no new model results claimed |
| M5 | PatchCore comparison on same dataset/protocol | **Kaggle run and smoke passed; Mac archived benchmark audit succeeded** | Eight hashes OK; generated summary from saved comparisons/reports/manifests; separate smoke ZIP integrity passed |
| M6 | Real FastAPI + browser demo against trusted model | **Dual-model normal/defective browser & four-request API acceptance passed on Mac; remaining QA pending** | Both checkpoints routed and all four expected user-labeled classifications/overlays confirmed; file hashes archived; saved split provenance and error-state testing pending |
| M7 | Persistent serving, tests and performance profiling | **Paired sequential audit and small PaDiM concurrency-2 probe observed on Mac; PatchCore/memory pending** | Both models sequential: 20 cached calls and matching image hash; PaDiM 10/10 HTTP 200 at concurrency 2, with 0 failures and 10/10 reuse; repeated/load and memory testing pending |
| M8 | Authentication, rate limit, audit logging | **Local-process baseline implemented; deployment validation pending** | Optional shared API key, global per-process rate cap, finite admission and request ID logs tested offline; identity, proxy/TLS, secrets and load tests pending |
| M8b | Private inspection history, operator review and lot traceability | **Opt-in ledger, human review and private lot grouping implemented; Mac test pending** | Validated opaque lot IDs, per-lot counts and human/model disagreements; preserved legacy records through migration; production controls still pending |
| M8c | Per-inspection checkpoint provenance and version analytics | **Trusted file SHA-256 history linkage and authenticated version-group reporting implemented; Mac test pending** | Inspections grouped by category/model/precise SHA or unknown legacy; bounded lot filter, human review counts/disagreement reporting; no signed attestation or real Mac verification |
| M8d | Operator-approved checkpoint version pinning | **Opt-in strict local allowlist and fail-closed request checks implemented; Mac test pending** | Private versioned category/model SHA manifest, required operator API key, unpinned families hidden, unexpected bytes reject before inference; no signed approval governance |
| M9 | Container packaging and hosted demo | **Local-only Docker packaging added; build and hosting pending** | Dockerfile + Compose, readonly model mounts and static guardrail tests; real Docker inference, remote TLS/gateway and rollback are unverified |
| M10 | Portfolio release | **Evidence verifier and release checklist implemented; publishing pending** | Read-only local model artifact/benchmark/smoke audit and explicit scientific/legal/security acceptance gates; Mac run, media and license approval pending |

## Safe continuation of 15-category experiments

The [resumable MVTec AD training runner](resumable-multicategory-training.md)
now preflights selected classes, rejects incomplete model artifacts, detects
split mismatches and plans missing PaDiM/PatchCore jobs before running anything.
New training requires an explicit `--execute` flag and bounds 1–4 jobs per
invocation. This is **source/CI functionality**, not real experimental results.
All 15 category-specific training/evaluation runs and MVTec AD 2 adapters
remain operational/research work requiring datasets and compute.

## Full original platform vision (beyond bottle-only demo)

The original end state is an industrial quality-inspection software platform:
**normal-only training on the 15 MVTec AD categories**, selected MVTec AD 2
categories (separate dataset adapter and rights review), validated **PaDiM and
PatchCore category-specific inference** with raw scores, normal/defective
decisions and pixel-level localization, and an operator-friendly application
with configurable models, repeatable tests, reports, deployment and governance.
The first [multi-category registry implementation](multicategory-model-registry.md)
adds safe checkpoint discovery and routing, **not pre-trained multi-category
models**. Training, saved model quality, inspection workflows, production MLOps
and deployments still need implementation and real data/evidence. Do not
confuse a functioning bottle-only portfolio prototype with this full vision.

## Browser batch inspection milestone

A new operator-facing `/batch` dashboard now provides bounded multi-image
selection, trained category/model routing, sequential localhost predictions,
per-image classification/error status and private browser CSV export.
Automatic regression tests cover selection, invalid batches, injected filenames,
HTTP errors and category consistency. Actual macOS browser testing and
persistent lot history remain pending.

## Operator batch inspection software milestone

The repository now includes a **finite folder inspection workflow** with
selected product category/model, per-image defect decisions, raw scores,
validated heatmaps and separate JSON/CSV operator reports. This is a functional
manufacturing-workflow building block, not another latency benchmark. Offline
regression tests cover success, mixed errors, size bounds, filename safety and
non-overwriting. **Mac real-checkpoint end-to-end execution is pending**, as
are a web batch dashboard, review history, traceability, data rights and
deployment verification. See [batch inspection guide](batch-inspection.md).

## Immediate priorities

### Priority 1 — Preserve verified model artifacts

Export the Kaggle-generated `model.ckpt`, `report.json`, `splits.json`, `smoke_report.json` and `smoke_overlay.png` from the session. Avoid re-running an already successful experiment unless needed for repeatability.

**Definition of done:** The checkpoint is stored safely outside Git; the experiment JSONs are reviewable; no dataset redistribution.

### Priority 2 — PatchCore benchmark

The user-observed Kaggle [PatchCore/bottle run](../reports/2026-10-15-kaggle-patchcore-bottle.md) has now completed after preserving the prior PaDiM checkpoint with `--reuse-existing` and checking exact train/validation/test paths. Results are measured in the console log. The user has **downloaded and extracted** a 389.04 MB ZIP on macOS, with eight artifact hashes successfully matching the included `SHA256SUMS.txt`. A held-out defect-image PatchCore checkpoint smoke test also returned `single-image-inference-passed`. The user's Mac run of `scripts.summarize_comparison` produced `benchmark_summary.md` successfully from the saved reports and comparison files; it checks exact manifest/metric consistency without deserializing checkpoints. The newer smoke report/overlay were also downloaded separately and their ZIP passed `unzip -t`. An independent scientific replication remains future work. Do not rerun successful training to regenerate metrics.

**Definition of done:** Real PatchCore report; complete comparison CSV/JSON; clear hardware/split provenance.

### Priority 3 — Live app inspection

The Mac preflight, checkpoint-backed HTTP API smoke test and a pair of cached repeated calls already passed. A 10-request same-image latency sample is now recorded (all requests marked `model_reused=true`; p50 135.78 ms, p95 1244.62 ms due to one spike). The user provided screenshots from a real-browser normal and defective bottle test showing expected labels and localization images ([UI evidence](../reports/2026-10-14-browser-bottle-inspection.md)). A newer [dual-model normal-image browser test](../reports/2026-10-15-macos-dual-model-dashboard-normal-ui.md) additionally verified `GET /models` registration and both real checkpoint-backed model choices with displayed normal predictions and heatmaps. The repeatable [dual-model local API acceptance harness](../docs/development.md#m6-real-dual-model-acceptance) **has been executed successfully on Mac**: all four real requests matched user-provided labels, returned overlays, and recorded image SHA-256/cache reuse in the [2026-10-16 verified run](../reports/2026-10-16-macos-dual-model-api-acceptance.md). A later Safari screenshot also shows PaDiM flagging `defective_bottle.png` with score `0.8633`, 674.09 ms and a rendered anomaly overlay; A 2026-10-16 Safari screenshot now also shows **PatchCore flagging a browser file named `defective_bottle.png`** with score `0.9288`, 565.47 ms and a localization overlay. All four normal/defective × PaDiM/PatchCore UI cases have been observed, but byte-identical image provenance and held-out split membership remain unchecked. Next, independently validate the image-source/saved test manifest and check error-state handling. Broader latency/throughput testing remains pending. A repeatable [sequential serving profiler](serving-benchmark.md) now separates warmup, client round-trip and model-internal prediction durations; real Mac **PatchCore small concurrent probe**, controlled repeat runs and memory profiling remain outstanding. A bounded PaDiM concurrency-2 run completed with 10/10 HTTP 200 responses, 0 failed/rejected and 10/10 model reuse ([observed results](../reports/2026-10-16-macos-padim-concurrent-probe.md)). The paired saved-JSON audit completed successfully on 2026-10-16 with matching transmitted image SHA, runtime, workload and 100% cache reuse. Two real same-image sequential cached profiles have now been observed ([results](../reports/2026-10-16-macos-paired-serving-profile.md)). Resolve mismatches between CLI and web behavior.

**Definition of done:** Browser upload and score/label/overlay have been observed for normal/defective examples. Error-state coverage, traceable held-out sample provenance and release screenshot packaging remain pending.

### Priority 4 — Improve engineering robustness

A persistent model instance is implemented and observed working with real PaDiM. The `/predict` response now has a Pydantic/OpenAPI schema that rejects invalid scores and latency. Initial request IDs and scrubbed audit logs, optional shared-key authentication, rate cap and admission timeout are implemented in the [local security baseline](security-baseline.md). Continue with gateway upload controls, typed settings, full trace/profiling, account-aware quotas, privacy and cloud-ready operations. Expand integration tests to real-model checks in a dedicated environment.

**Definition of done:** Repeatable p50/p95 load benchmark with documented CPU/GPU and concurrency; no unsafe checkpoint upload path.

### Priority 4b — Operator inspection traceability

A metadata-only, **explicitly opt-in** private SQLite history and `/history`
operator UI have been added ([setup and limitations](private-inspection-history.md)).
Successful predictions can be browsed/deleted with an operator API key, without
saving images, filenames, heatmaps, checkpoint paths or secrets. This does not
claim immutable audit compliance, multi-tenant RBAC or a production database.
Actual checkpoint-backed Mac startup, restart persistence, and lifecycle QA
remain acceptance gates. Follow-up: site/lot identifiers, operator review
decisions, migration and retention controls.

### Priority 5 — Portfolio finalization

Consolidate measured results, demo screenshots, software architecture, limitations, reproducibility guide, and a short demonstration video. [Docker packaging](container-deployment.md) now provides a local-only build path but must be built and tested before hosting claims. Select an intentional source-code license after reviewing dependency/dataset terms. A [read-only portfolio evidence verifier and release checklist](portfolio-release-readiness.md) now exist; user approval and actual release verification are still required.

**Definition of done:** Clean GitHub release, green CI, reviewed documentation and a repeatable live demo.

## Release policy

Maintain the active work in a feature branch/PR until end-to-end verification. Do not claim "production-grade", "99% defect detection", "certified industrial inspection", or published benchmark superiority solely from a single category with a validation holdout.

## Source of truth

- [README](../README.md)
- [Evaluation report](../reports/2026-10-13-kaggle-padim-bottle.md)
- [Evaluation methodology](evaluation.md)
- [Deployment and security requirements](deployment.md)
