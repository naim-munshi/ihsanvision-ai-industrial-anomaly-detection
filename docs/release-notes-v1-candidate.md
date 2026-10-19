# IhsanVision.AI v1.0.0 — candidate release notes (NOT PUBLISHED)

**Draft only — publication blocked.** Update this page with dated verified
Cable/Carpet data, latest private Mac/Docker acceptance, approved original
demo screenshots/video and completed third-party rights review **before**
creating any `v1.0.0` tag or GitHub release. This Markdown page is not a
GitHub Release and is intentionally not promoted as one.

## Release goal

An **AI Engineer job portfolio prototype** combining normal-image-only
unsupervised industrial visual anomaly detection (PaDiM/PatchCore), a FastAPI
model-serving application, browser single/batch inspection, human review and
private traceability. This is **not a factory safety-certified system**.

## Features implemented and CI-tested

- Trusted local checkpoint-backed image inference with raw anomaly score,
  predicted normal/anomaly label and **relative**, noncalibrated heatmap.
- Category/model routing, optional in-process checkpoint cache, bounded upload
  and admission, opt-in API-key protection and local-only operator telemetry.
- Browser single-result and multi-image batch interfaces.
- Optional authenticated private SQLite inspection history, lot/version
  summaries, independent human verdict and metadata-only CSV exports.
- Paired evaluation split auditing, resumable MVTec AD experiments, honest
  model scorecards and operator-supplied held-out FP/FN audit tools.
- Local-only Docker/Compose configuration with read-only external checkpoints,
  nonroot process and bounded temporary storage. **Actual Linux CPU Docker
  image build/API startup/model-class initialization and dependency checks
  passed without private weights**; this is not trained-checkpoint inference.
- Automated Python 3.11/3.12 tests, browser JavaScript tests, source-release
  scanning, Markdown link integrity and a limited Git-history secrets audit.

## Documented real model evidence (Bottle only, until next training)

Historical real Kaggle observations, seed 42, original MVTec AD / bottle.
209 normal train, 16 validation samples held out from the original test folder,
67 remaining test images; exact paired path manifests. Not comparable to
official original-full-test MVTec AD leaderboard scores.

| Metric | PaDiM | PatchCore |
| --- | ---: | ---: |
| Image AUROC | 0.9963 | 1.0000 |
| Image F1 | 0.9592 | 0.9901 |
| Pixel AUROC | 0.9768 | 0.9857 |
| Pixel F1 | 0.6497 | 0.7102 |

Original observed real Mac warm sequential HTTP profiles on 20 measured calls
per model: PaDiM p50/p95 **388.87 / 766.46 ms** and PatchCore **493.05 /
514.82 ms**. These are not production SLA measurements or proof of current
build's real-model operator workflow.

## Must replace draft with real verified v1 evidence

- **P4/P5:** Add actual Cable and Carpet PaDiM/PatchCore metrics from the
  original six saved private model/report/split triples; no estimated values.
- **P6:** Link the real three-category comparison and grounded FP/FN review,
  without exposing private image paths or image files.
- **P7:** Link dated, sanitized *latest-build* real Mac acceptance covering
  batch, SQLite persistence after restart, human review and CSV.
- **P8:** Link genuine target-machine Docker **checkpoint-backed** PaDiM and
  PatchCore inference, heatmap, restart and memory observations.
- **P9:** Add exactly approved and rights-cleared original demo screenshots and
  ~60–120 s video (no MVTec images or third-party content without permission).
- **P10:** Review code/contribution copyright, the custom use-restricted
  [Responsible Use License v1.0](../LICENSE), MVTec/pretrained-weight/media
  restrictions, fresh Git history scan, all final CI and release archive.
  Publish only after all other acceptance gates pass.

## Use conditions and limitations

The original project code is source-available under a use-restricted
responsible-use license: violent, fraudulent, abusive-surveillance,
discriminatory or unsafe high-risk uses are prohibited. It is **not**
OSI-approved open source. This does **not** authorize commercial MVTec AD
dataset use or third-party model/media redistribution.

Unsupervised models trained on normal images can miss costly defects and
produce false positives under lighting/domain shift. Anomaly scores are
not probabilities; relative heatmaps are not certified defect masks.
Human oversight and local pilot validation are required before any
industrial deployment.

**Release rule:** The single authoritative P1–P10 gate checklist is
[GitHub issue #53](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53).
Do not publish this draft or create a tag until all ten gates have genuine
supporting evidence. This draft itself contains no manufactured data,
unlicensed images or private checkpoint binaries.
