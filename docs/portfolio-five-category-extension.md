# IhsanVision.AI — owner-approved five-category portfolio extension

**Status (2026-10-20 JST):** The owner expanded the pre-release portfolio target from three classic MVTec AD categories to **five**: `bottle`, `cable`, `carpet`, `metal_nut`, and `capsule`. Each category has **two** normal-only anomaly detectors: PaDiM and PatchCore (**10 model/category runs in total**). This amendment supersedes **only the category/model-count scope and related comparison/training instructions** in [the original three-category v1.0 acceptance plan](portfolio-v1-plan.md) and [FINISHING.md](../FINISHING.md); its engineering, security, rights, and real-runtime release gates still apply. The owner must confirm final release naming and amend release notes **before tagging**; no new release is represented as published.

## What has actually been observed

The owner supplied executed Kaggle logs on 2026-10-20 for Cable/Carpet/Metal Nut/Capsule × PaDiM/PatchCore, using a GPU-enabled MVTec AD mount and saved seed-42 model/report/split triples. The resumable runner reported `trained_and_verified` for all eight new runs and ultimately reported **zero pending jobs** for Cable/Carpet and for Metal Nut/Capsule separately. Original Bottle model evidence is documented in [PaDiM](../reports/2026-10-13-kaggle-padim-bottle.md) and [PatchCore](../reports/2026-10-15-kaggle-patchcore-bottle.md). **The owner subsequently assembled all ten private artifact triples on their Mac and ran the read-only `scripts.summarize_comparison --from-saved-runs` command successfully on 2026-10-20.** The report audited the saved checkpoint/report/split file contracts and exact same-category PaDiM/PatchCore train/validation/test path parity. This is owner-provided live Mac console evidence of saved-artifact consistency, not independent replication of training, matching dataset bytes, or successful live inference. A persistently saved private Markdown report and actual false-positive/negative review are still outstanding.

| Category | Model | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Source status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bottle | PaDiM | 0.9963 | 0.9592 | 0.9768 | 0.6497 | Historical original run |
| bottle | PatchCore | 1.0000 | 0.9901 | 0.9857 | 0.7102 | Historical original run |
| cable | PaDiM | 0.8324 | 0.7660 | 0.9536 | 0.4307 | Owner Kaggle log |
| cable | PatchCore | 0.9911 | 0.9726 | 0.9848 | 0.6400 | Owner Kaggle log |
| carpet | PaDiM | 0.9855 | 0.9640 | 0.9884 | 0.5267 | Owner Kaggle log |
| carpet | PatchCore | 0.9831 | 0.9565 | 0.9897 | 0.5457 | Owner Kaggle log |
| metal_nut | PaDiM | 0.9533 | 0.9351 | 0.9404 | 0.6343 | Owner Kaggle log |
| metal_nut | PatchCore | 0.9963 | 0.9583 | 0.9865 | 0.8349 | Owner Kaggle log |
| capsule | PaDiM | 0.8738 | 0.9333 | 0.9827 | 0.4248 | Owner Kaggle log |
| capsule | PatchCore | 0.9922 | 0.9708 | 0.9898 | 0.4995 | Owner Kaggle log |

**Interpretation:** These metrics are from a validation holdout taken from MVTec's original test directory, with evaluation on the remainder. They **must not** be labeled as official full-test leaderboard scores. Pixel-F1 remains materially weaker than pixel-AUROC in some categories; a high AUROC does not establish accurate per-defect localization, real factory reliability, or calibrated confidence. Comparisons are paired **within** a category, not a ranking across different categories.

## Archive and verification gate — outstanding owner actions

1. **Mac SHA-256 verification observed:** the owner downloaded all eight new ZIP archives (Cable, Carpet, Metal Nut, Capsule × both models) and reported **eight `OK` results** from `shasum -a 256 -c SHA256SUMS-eight-kaggle.txt`. Maintain a second durable private copy; manifest matching verifies transport/integrity relative to that manifest, not model training provenance. Do not upload MVTec images, checkpoints, split paths, auth tokens, or other restricted/private artifacts to public GitHub.
2. **All ten artifact triples were reported restored** to a distinct private Mac runs directory, with the original Bottle archive still preserved. Do **not** retrain or overwrite Bottle. Maintain offline private backups and investigate any future checksum changes. Original image-path manifests from the Bottle Kaggle session and newer sessions may differ: the comparator verifies PaDiM/PatchCore equality *per category*, not provenance or image hashes across sessions.
3. **Ten-run read-only comparison passed on the owner's Mac (2026-10-20):** the utility inspected all ten checkpoint/report/split triples, verified five same-category paired path manifests and emitted the metrics and split counts. For reproducible private evidence, save a copy at a new location using:

   ```bash
   python -m scripts.summarize_comparison \
     --runs-dir /absolute/private/ihsanvision_runs \
     --from-saved-runs \
     --categories bottle cable carpet metal_nut capsule --seed 42 \
     --output /absolute/private/portfolio-five-category.md
   ```

   Save to a **new** output filename; the command deliberately refuses overwrites. This write-to-file step remains unconfirmed. Public metrics must be sourced from validated saved reports, with source paths/weights kept private.
4. Independently review **real** held-out false positives/negatives and visibly failed overlays, or explicitly document where none were observed. The existing [error-case audit protocol](portfolio-error-analysis.md) must be adapted or invoked with these five selected categories using actual prediction records, never fabricated rows.

## Browser/deployment release acceptance — outstanding real tests

- Register only approved trusted local checkpoints for **all five categories × two models** in a private [model registry](multicategory-model-registry.md). Untrusted checkpoints must never be loaded or accepted from user upload.
- Test **real normal and defective images**, both models, category selection and model routing, overlays, batch uploads, history/restart persistence, operator review, lot grouping, CSV export, authorization and error cases on the Mac. The previous verified Bottle-only test does **not** prove the new four categories work in the latest browser build.
- Perform actual checkpoint-backed Docker CPU acceptance in a local container, with non-root/read-only constraints and secure localhost binding. Confirm available memory and realistic inference latency. Models trained under Kaggle CUDA should still be validated for CPU inference; do not assume runtime success from architecture compatibility.
- Publish only permission-cleared *real* screenshots/video, correct limits, safe README/model cards, security/CI evidence and owner-reviewed dataset/pretrained/media rights. Do not claim factory certification, production-grade reliability or SaaS.
- Release only after original gates **P1–P10**, updated expanded-scope evidence for **Cable, Carpet, Metal Nut and Capsule**, the ten-model comparison, and release notes/rights checks are verifiably complete. The [issue #53 evidence tracker](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53) records the dated owner log evidence.

## Explicit non-goals

The remaining ten classic MVTec AD categories, MVTec AD 2, Kubernetes, multi-tenant SaaS, hardware actuators, and commercial service-level guarantees are still out of scope. No further category training is required for this extended portfolio target.
