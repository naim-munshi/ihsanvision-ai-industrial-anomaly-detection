# IhsanVision.AI — portfolio v1.0 release acceptance

**Current status: not released.** This is the operational checklist for the [fixed AI engineer portfolio v1.0 scope](portfolio-v1-plan.md), **not** a factory-production certification program. Do not create a `v1.0.0` tag or mark the portfolio as finished until its ten gates are satisfied.

## Evidence that already exists

- [Bottle PaDiM](../reports/2026-10-13-kaggle-padim-bottle.md) and [Bottle PatchCore](../reports/2026-10-15-kaggle-patchcore-bottle.md) were genuinely trained/evaluated in Kaggle. The original artifacts are privately preserved and were checked against a bundled SHA-256 manifest on Mac.
- [Mac Bottle dual-model acceptance](../reports/2026-10-16-macos-dual-model-api-acceptance.md) showed valid overlay PNGs and four manually labeled predictions.
- [Mac paired serving profile](../reports/2026-10-16-macos-paired-serving-profile.md) measured same-input warm HTTP p50/p95 latency on both models.
- Python 3.11/3.12, JavaScript dashboard and static Docker Compose security checks execute in [GitHub Actions](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/workflows/ci.yml).

These different observations are **not interchangeable**. An offline mocked API test is not real checkpoint inference; a checksum match is not scientific replication; a four-image smoke test is not accuracy measurement. Results use a validation subset held out of the original MVTec test set rather than the official full-test benchmark protocol.

## A. Validate preserved original model evidence (Mac/private)

Use an updated checkout and independently owned original run archive. First confirm the saved full reference manifest and `comparison.json` are the correct files for the same benchmark:

```bash
python -m scripts.summarize_comparison \
  --runs-dir "$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15"
```

Then run the read-only portfolio evidence verifier, supplying the **actual** saved PatchCore smoke ZIP:

```bash
python -m scripts.verify_portfolio_evidence \
  --runs-dir "$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15" \
  --smoke-zip "$HOME/Downloads/ihsanvision_patchcore_smoke_evidence.zip" \
  --output "$HOME/ihsanvision-artifacts/portfolio-evidence-v1-audit.json"
```

The verifier checks the bundled eight-file SHA-256 manifest, pairwise saved report/split consistency, saved Markdown summary and smoke ZIP. No model deserialization or dataset redistribution occurs. The user must inspect the output and keep it privately; a passing report is **artifact consistency evidence**, not proof of generalizable model accuracy.

## B. Complete only the missing portfolio model pairs (Kaggle or compute)

The final v1.0 target is **Bottle, Cable, Carpet × PaDiM/PatchCore** (six total). Bottle is already trained, so **four model/category runs** remain. Use the [dry-run-first planner](resumable-multicategory-training.md) on your licensed dataset and preserved benchmark output folder; execute one missing run at a time.

When six saved model/report/split trios actually exist, create their
checked comparison without retraining:

```bash
python -m scripts.summarize_comparison \
  --runs-dir /absolute/private/ihsanvision_runs \
  --from-saved-runs --categories bottle cable carpet --seed 42
```

The command must fail when any pair is missing or splits differ; no new
category performance is currently claimed.

For independently labeled real test predictions, run the [private error-case audit](portfolio-error-analysis.md) and manually review actual FP/FN overlays. No evaluation data or images are public by default.

A final per-category benchmark record must include counts, seed, train/val/test path-equivalence checks, metrics, original checkpoints and at least some grounded error-case analysis. The evaluation protocol must remain unchanged across paired models. Do not compare to official full-test leaderboards; do not invent absent scores or force a false positive/false negative example where none occurred.

## C. Run current application end to end using real trusted checkpoints

On the Mac, install the documented ML dependencies and start `python -m scripts.run_local_demo` with the two trusted Bottle weights. In a separate terminal, run:

```bash
python -m scripts.local_stack_acceptance \
  --url http://127.0.0.1:8003 \
  --normal /absolute/private/normal_bottle.png \
  --defective /absolute/private/defective_bottle.png \
  --output "$HOME/ihsanvision-artifacts/portfolio-local-stack-v1.json"
```

If private server key authentication is enabled, privately set `IHSANVISION_CLIENT_API_KEY` and add `--require-auth`. With private history and authentication deliberately enabled, rerun the same command with `--require-auth --operator-workflow` using a new output file. It tracks and cleans **all four dual-model test inspections plus one synthetic operator-review inspection** while checking authentication, review, lot/version summaries, private CSV and telemetry. Additionally **manually** verify the latest `/batch` and authenticated `/history` workflow: upload multiple photos, record/restart persistence, independent operator verdict, per-lot summary and metadata-only CSV. The local-stack tool does not itself exercise browser clicks, SQLite mutation or screenshot accuracy.

Record a reproducible short summary without private image files, full local paths or credentials. Inspect overlays; check visible failures, empty/error responses and small-screen/keyboard interaction.

## D. Verify a real Docker build and actual inference

The GitHub CI `container-config` job only validates static Compose security settings. On an actual Docker machine with trusted private weights:

```bash
# Set IHSANVISION_MODELS_DIR and IHSANVISION_API_KEY privately as documented.
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
```

Then run the same local-stack acceptance with `--url http://127.0.0.1:8080`, choosing a **new private output report path**. Confirm the container is non-root, read-only and loopback-mapped; verify restart, readiness, actual checkpoint inference and memory footprint. See [complete local container procedure](container-deployment.md). No public hosted service is required for the job portfolio.

## E. Publish a trustworthy recruiter experience

Only after verifying the final software:

- Add **three permission-cleared original screenshots**: single-image normal/defect result with relative heatmap, PaDiM/PatchCore view, and batch/history operator report. They must show actual responses, not invented detection scores.
- Record a **60–120 second real demo**: model select → upload → overlay → batch report → human review. Link the video prominently from README. A private unlisted link is acceptable only if recruiters can access it.
- Ensure README's top section contains the honest PaDiM/PatchCore metric comparison, service performance evidence, architecture diagram, quick-start, tests and limitations.
- Keep MVTec images, model weights, split file paths, credentials, customer/lot secrets and serialized checkpoints out of the **public** release. The source code currently has **no LICENSE**; the owner must select a legally appropriate license before publication. Review dataset, pretrained-weight and demo-media rights separately.

Use the real [model card](model-card.md), [evaluation methodology](evaluation.md) and [security boundaries](security-baseline.md). Do not claim certified defect segmentation or general manufacturing accuracy.

## F. Version and release

Only after all ten [portfolio acceptance gates](portfolio-v1-plan.md) pass, create a reviewed changelog/tag/release `v1.0.0` from a clean green-`main` commit. Check the source archive for secrets and disallowed dataset imagery, verify all README links/media, and validate the example commands from a fresh environment. Do not fake a GitHub release, live URL, cloud deployment or benchmark.

**Stop condition:** All ten frozen portfolio gates are verified with dated evidence and a real `v1.0.0` release exists. That—not a percentage or PR count—is the definition of portfolio completion.
