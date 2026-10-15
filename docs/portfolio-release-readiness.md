# IhsanVision.AI — Portfolio release readiness

## Status: prototype; no public release approved

This document is a **release gate**, not a declaration of production readiness. Trained models and observed browser demos already exist, but there is no independently verified hosted demo, end-to-end real Mac Docker build, full threat-model review, multi-category training study or chosen source-code license.

## Research artifacts and provenance gate (automated)

Run from an updated project checkout on the **Mac that holds the private Kaggle outputs**:

~~~bash
python -m scripts.verify_portfolio_evidence \
  --runs-dir "$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15" \
  --smoke-zip "$HOME/Downloads/ihsanvision_patchcore_smoke_evidence.zip" \
  --output "$HOME/ihsanvision-artifacts/portfolio_evidence_audit.json"
~~~

This command is read-only toward original evidence and does not load PyTorch checkpoints, retrain models or send files across the network. It checks SHA-256 for both model checkpoints and their JSON reports/splits plus comparison outputs against the **same bundled manifest** the user already verified, reruns exact metric/split consistency from the existing comparison summary module, verifies the saved Markdown report and checks the PatchCore smoke report and overlay ZIP. It refuses unexpected archive entries and overwriting an existing output JSON. The generated audit JSON deliberately contains **no private absolute paths, source image bytes, model weights or base64 heatmaps**.

The status **artifact_evidence_verified** is *not* an independent certification of where the checkpoint originally came from, scientific replication, measured model generality, or approval to deploy. It specifically reports public_release_authorized=false.

## Portfolio narrative architecture

~~~mermaid
flowchart LR
    A[Inspector browser] --> B[FastAPI prediction API]
    B --> C[Optional API key and per-process admission]
    C --> D[PaDiM or PatchCore inference session]
    E[Trusted private checkpoints] --> D
    D --> F[Raw anomaly score and relative overlay]
    F --> A
    G[MVTec AD training inputs] --> H[Kaggle Anomalib experiments]
    H --> I[Private checkpoint, report and split archives]
    I --> E
    I --> J[Read-only benchmark evidence verifier]
    K[Offline GitHub Actions tests] --> B
~~~

This is a **logical prototype architecture**. It does not imply hosted cloud infrastructure, real multi-tenant authentication, distributed queues, production monitoring or an operational training pipeline.

## Gate-by-gate launch checklist

| Gate | Current evidence | What is still needed |
| --- | --- | --- |
| Reproducible model experiments | PaDiM/PatchCore bottle logs, raw reports and manifests saved privately; local verified comparison summary generated | Record runtime provenance for both; multiple categories/seeds, controlled reproduction |
| Private artifact preservation | Both checkpoint ZIP archive checksum verified on Mac; separate smoke ZIP passes integrity | Run new read-only audit and establish a second independent private backup |
| Real model browser workflow | PaDiM normal/defect observed; PaDiM/PatchCore normal-image Safari tests passed | Same defective image through *both* models; verify source split/path and overlay plausibility |
| Runtime performance | Historical 10-request PaDiM sample; newer profiler implemented and offline-tested | Real matching-input p50/p95 runs, cold vs warm trials, peak memory, concurrency/admission test |
| Security | Optional shared key, 429 limiter, finite admission and request IDs tested offline | HTTPS reverse proxy, pre-body authentication/limits, managed secrets, account roles, logs/privacy review, incident handling |
| Container/deployment | Nonroot local-only Docker/Compose source and static tests | Successful image build and checkpoint inference on target machine, deployment/rollback checks |
| UX accessibility | Browser interface renders and demo cases observed | Error-state, small-screen, keyboard-only and screen-reader testing |
| Legal | Research dataset/model attribution is documented | User approval of source-code license, verify rights for weights, pretrained assets, MVTec visual redistribution |
| Portfolio presentation | README, architecture, evaluation notes and CI | Approved anonymized visuals, short demo video, final results table, release notes and version tag |

**Do not publish MVTec training/test images, model binaries, notebook secrets or user identifiers by default.** Prefer links to project source and written measurements with methodological caveats; any data/weight release needs licensing approval. Do not select a legal license on the user's behalf without their decision.

## GitHub release acceptance

Before a v1.0 tag, check the merged main commit has green Python CI; run real M6 and M7 checks on the same Mac; confirm M9 Docker and API security integration; review all media rights, README claims and license; document known limitations; create a reproducible tag/release with concise changelog; verify the source tarball contains no private datasets or secrets. **No tag or GitHub release is created by this checklist.**

## Reproducibility / interpretation warning

Image AUROC can reach 1.0 on a small held-out test split without guaranteeing 100% future defect detection. The validation subset is drawn from MVTec AD's original test directory, so the reported PaDiM/PatchCore numbers do not match the external full-test benchmark protocol. Raw anomaly scores and contrast-normalized overlays are not calibrated probabilities or validated segmentation masks. A working web UI and finite inference time do not certify industrial safety or reliability.
