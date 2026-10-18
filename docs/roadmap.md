# Engineering history and milestone status

> **Historical milestones — not the active scope checklist.** The
> [AI Engineer portfolio v1.0 plan](portfolio-v1-plan.md) and
> [GitHub P1–P10 acceptance tracker](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53)
> define project completion. The original 15-category, MVTec AD2,
> enterprise/factory-scale roadmap is **not required** for the job portfolio.

## What has actually been delivered

| Earlier milestone | Evidence-backed status | References |
| --- | --- | --- |
| Core PaDiM experiment | Bottle real Kaggle training, held-out metrics | [PaDiM report](../reports/2026-10-13-kaggle-padim-bottle.md) |
| PaDiM/PatchCore paired evaluation | Bottle real Kaggle results, matching saved image split paths | [PatchCore report](../reports/2026-10-15-kaggle-patchcore-bottle.md), [model card](model-card.md) |
| Interactive local API/UI | Real Bottle predictions and overlays previously checked on Mac | [Mac acceptance](../reports/2026-10-16-macos-dual-model-api-acceptance.md) |
| Runtime profiling | Limited warm sequential HTTP latency on real Mac | [serving profile](../reports/2026-10-16-macos-paired-serving-profile.md) |
| Batch/history/review/lot/CSV | Implemented with automated mocked/offline tests; current full Mac acceptance outstanding | [operator guide](private-inspection-history.md), [local acceptance](operational-acceptance.md) |
| Checkpoint provenance, registry and model approvals | Implemented; owner approval and real checkpoint lifecycle acceptance outstanding | [release runbook](operator-model-releases.md) |
| API security and monitoring | Process-local safeguards, authenticated private telemetry, CI regression tests; not a public TLS service | [security baseline](security-baseline.md) |
| Docker packaging | **Actual** Linux CPU image build, HTTP smoke, Anomalib and model-class initialization plus `pip check` passed in GitHub Actions | [cloud build evidence](docker-cloud-build-evidence.md) |
| Additional classic categories | Cable/Carpet scripts, split checks and reporting tools exist; **four genuine fits still pending** | [owner handoff](../FINISHING.md) |
| Release | No owner-approved source LICENSE, real media or `v1.0.0` GitHub release yet | [release checklist](portfolio-release-readiness.md) |

These entries describe different evidence levels. A green offline CI, model
class initialization without pretrained weights and a real model inference
are **not equivalent**. Neither a passed Docker image build nor a real
Bottle experiment proves the entire application is ready for factory
deployment.

## Active portfolio v1.0 acceptance gates

The scope is deliberately **three categories** — Bottle, Cable and Carpet
— with PaDiM and PatchCore, a demonstrable local operator workflow,
reproducible measured results, a trained-checkpoint Docker demo, and
rights-cleared presentation. More general category coverage, AD2, managed
cloud hosting, RBAC and conveyor automation are not v1.0 requirements.

- **P1–P3:** earlier Bottle training/inference evidence and source CI exist.
- **P4–P6:** genuine Cable/Carpet fits, paired metrics and grounded FP/FN
  analysis remain external compute/evidence tasks.
- **P7:** latest real Mac operator workflow, including persistence/restart,
  remains to be executed with trusted Bottle weights.
- **P8:** GitHub Linux Docker startup/model initialization is verified, but
  trusted-checkpoint Docker inference/restart on target hardware remains.
- **P9–P10:** original permitted screenshots/video, release rights and code
  license decision, full source review and versioned release remain.

Use [FINISHING.md](../FINISHING.md) rather than historical milestone prose
as the executable plan. Do not create new feature milestones without
explicitly changing the fixed portfolio scope.

## Historical work kept outside portfolio scope

The repository retains MVTec AD2 adapters and broader multicategory
utilities as research/reference code, but no real AD2 score is claimed.
Earlier scripts, experiment observations and dated reports remain in Git
for traceability and do not imply full 15-category training.
