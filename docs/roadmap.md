# Engineering Roadmap

This is a **state-based roadmap**, not an assertion that all systems are already production-ready. Completion requires observable evidence.

## Milestones

| Stage | Description | State | Acceptance evidence |
| --- | --- | --- | --- |
| M1 | Source layout, dependencies, README, code hygiene | **Implemented** | Documented modules and reproducible commands |
| M2 | PaDiM training and checkpoint-backed held-out evaluation | **Verified on Kaggle** | Completed notebook run and metric report |
| M3 | Single-image model inference + heatmap | **Verified on Kaggle** | `single-image-inference-passed`, overlay screenshot |
| M4 | CI and dependency-isolated tests | **Verified** | Python 3.11/3.12 GitHub Actions pass |
| M5 | PatchCore comparison on same dataset/protocol | **Pending** | Metric tables, timings, model configs and hardware |
| M6 | Real FastAPI + browser demo against trusted model | **HTTP service verified; browser upload pending** | Real Mac API pass; normal and defective browser uploads still needed |
| M7 | Persistent serving, tests and performance profiling | **PaDiM reuse verified on Mac; profiling pending** | 721.99 ms first / 123.54 ms reused on one pair; record p50/p95 with more samples |
| M8 | Authentication, rate limit, audit logging | **Pending** | Security tests and privacy policy |
| M9 | Container packaging and hosted demo | **Pending** | Verified deployment, CI release and rollback |
| M10 | Portfolio release | **Pending** | Demonstration, figures, reproducibility package and review |

## Immediate priorities

### Priority 1 — Preserve verified model artifacts

Export the Kaggle-generated `model.ckpt`, `report.json`, `splits.json`, `smoke_report.json` and `smoke_overlay.png` from the session. Avoid re-running an already successful experiment unless needed for repeatability.

**Definition of done:** The checkpoint is stored safely outside Git; the experiment JSONs are reviewable; no dataset redistribution.

### Priority 2 — PatchCore benchmark

Run `python -m src.benchmark` for the same category and seed (both model trainings are performed in that command). Compare model metrics **only if the saved train/validation/test path manifests match exactly**; the benchmark now raises instead of emitting a misleading CSV when partitions differ. Record training duration and inference time, not just AUROC/F1.

**Definition of done:** Real PatchCore report; complete comparison CSV/JSON; clear hardware/split provenance.

### Priority 3 — Live app inspection

The Mac preflight, checkpoint-backed HTTP API smoke test and a pair of cached repeated calls already passed. Next, collect a repeatable serving latency sample with `python -m scripts.benchmark_local_api --url http://127.0.0.1:8002 --runs 10`, then test normal and defective input images through the browser. Resolve mismatches between CLI and web behavior.

**Definition of done:** Browser works for real image upload; resulting score/label/overlay visible; failure states explained; local screenshots and API response saved.

### Priority 4 — Improve engineering robustness

A persistent model instance is implemented and observed working with real PaDiM. The `/predict` response now has a Pydantic/OpenAPI schema that rejects invalid scores and latency. Continue with request IDs, structured logs, configurable settings, response timeouts, and upload throttling. Expand integration tests to real-model checks in a dedicated environment.

**Definition of done:** Repeatable p50/p95 load benchmark with documented CPU/GPU and concurrency; no unsafe checkpoint upload path.

### Priority 5 — Portfolio finalization

Consolidate measured results, demo screenshots, software architecture, limitations, reproducibility guide, and a short demonstration video. Select an intentional source-code license after reviewing dependency/dataset terms.

**Definition of done:** Clean GitHub release, green CI, reviewed documentation and a repeatable live demo.

## Release policy

Maintain the active work in a feature branch/PR until end-to-end verification. Do not claim "production-grade", "99% defect detection", "certified industrial inspection", or published benchmark superiority solely from a single category with a validation holdout.

## Source of truth

- [README](../README.md)
- [Evaluation report](../reports/2026-10-13-kaggle-padim-bottle.md)
- [Evaluation methodology](evaluation.md)
- [Deployment and security requirements](deployment.md)
