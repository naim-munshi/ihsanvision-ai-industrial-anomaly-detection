# Changelog

All notable user-visible changes are recorded against Git revisions. This
project has **not** yet published its portfolio `v1.0.0` release.

## [Unreleased] — AI Engineer portfolio candidate

### ML evaluation and reproducibility

- Saved real Kaggle PaDiM and PatchCore Bottle experiments and documented a
  distinct validation/test split with paired model path-equivalence checks.
- Added resumable MVTec AD training planning that refuses incomplete or
  overwritten experiments.
- Added model/report/split validation, SHA-256 integrity checking and
  benchmark tables reconstructed **only from completed model artifacts**.
- Corrected scientific terminology to **normal-only unsupervised** detection
  and hardened saved-report validation against Boolean/perfect-score corruption.
- Made original eight-file Bottle archive verification independent of optional
  Markdown summaries and later smoke ZIPs; missing extras are explicitly **not checked**.
- Enabled restoring **both** completed Bottle model artifact triples without
  checkpoint deserialization or retraining.
- Added a private per-image FP/FN error analysis tool; actual cable/carpet
  experimental data remain **uncollected**.

### Local software and operator workflow

- Added FastAPI model routing, model cache and score/overlay responses.
- Added browser single-image and batch inspection, optional private history,
  human review, lot reporting and CSV export.
- Added operator-reviewed private checkpoint registry candidates, staged
  approved SHA-256 pins, manual model-release rollback instructions and
  config-aware readiness checks.
- Added request authentication, upload/admission limits, private single-process
  latency telemetry and negative/abuse tests.

### Engineering and release safety

- Added Python 3.11/3.12 and browser JavaScript CI tests, static Docker
  Compose security checks and a [successful actual Linux CPU image build and
  no-checkpoint HTTP API smoke](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/runs/39095740246).
- Verified no-network PaDiM/PatchCore model-class construction in the real Linux
  container and captured dependency-version evidence (not checkpoint inference).
- Added real local acceptance tools and opt-in authenticated operator
  write/review/report/cleanup; actual latest Mac checks are still pending.
- Added a Git-tracked public-source hygiene check to flag private data,
  unreviewed media and high-confidence committed secrets.
- Added a [dated reachable Git history security scan](reports/2026-10-19-git-history-audit.md)
  with zero *recognized-pattern* findings; this does not establish total
  secrets or licensing clearance.
- Added the use-restricted [Responsible Use License v1.0](LICENSE) for project
  source; separate dataset/media/pretrained-weight rights remain unresolved.
- Added repository-wide Markdown link verification and safe merged-topic-branch cleanup.
- Rebuilt project README/model card and froze the AI Engineer portfolio
  release criteria, demo storyboard and owner handoff.

### Unverified/blocked before `v1.0.0`

- **Cable + Carpet:** four missing real model experiments and comparative
  failure-case analysis.
- **Current runtime:** latest full Mac operator workflow and private
  checkpoint-backed Docker acceptance.
- **Presentation and legal:** authentic rights-cleared screenshots/video,
  legal review of the use-restricted source license, upstream dataset/weight/media
  rights clearance, and a real clean tagged GitHub release.

**Release note:** A green offline CI or cloud image build without checkpoint
weights does **not** prove actual industrial inference or production readiness.
The [frozen portfolio gates](docs/portfolio-v1-plan.md), not a percentage or
the number of merged PRs, determine release completion.
