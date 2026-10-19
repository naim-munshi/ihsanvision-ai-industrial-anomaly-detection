# IhsanVision.AI — final owner handoff for the AI Engineer portfolio

**Scope is frozen:** Bottle, Cable and Carpet × PaDiM/PatchCore; one usable
FastAPI operator app; reproducible experiments, honest results, Docker
acceptance, permitted demonstration and reviewed `v1.0.0` release.

This is the minimal order to finish the **remaining real-world gates**.
All scripts live in this repository. No commercial SaaS, Kubernetes, MVTec
AD2, 15-category leaderboard or factory hardware is required. Check the
[fixed acceptance tracker](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53)
for verified status; scripts existing in Git do **not** mean they were run.

## A. On Kaggle or your own training computer — only four missing jobs

Use **authorized MVTec AD** files and your privately preserved original
Bottle run/checkpoints. Both PaDiM and PatchCore Bottle are already recorded
as completed; do not retrain or overwrite those saved results. Ensure the
saved Bottle artifacts are restored to your private runs directory and the
MVTec dataset layout is complete. If using a different dataset revision/mount
path, investigate before claiming comparable experiments.

**Restore both Bottle models first** when starting from a clean Kaggle session.
Attach your privately preserved original model/report/split files using a
private input (not GitHub), then stage each model into the *same* private runs
directory. Set the source folders to the actual locations containing those
three saved files:

```bash
python -m scripts.stage_reference_artifacts \
  --source-dir /absolute/private/original-padim-bottle-seed42 \
  --output-dir /absolute/private/ihsanvision_runs \
  --model padim --category bottle --seed 42

python -m scripts.stage_reference_artifacts \
  --source-dir /absolute/private/original-patchcore-bottle-seed42 \
  --output-dir /absolute/private/ihsanvision_runs \
  --model patchcore --category bottle --seed 42
```

This only **copies and validates** trusted saved artifacts, never trains or
deserializes a checkpoint. Skip staging if the two run directories already
exist. Both source manifests must still match your current dataset's actual
resolved image paths; the planner cannot silently move Kaggle paths.

```bash
# Always inspect a no-training dry-run first:
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable carpet --seed 42

# Only after reading the plan, execute ONE missing job at a time:
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable carpet --seed 42 \
  --accelerator auto --execute --max-jobs 1
```

Repeat the second command only as needed until the plan says **zero missing
jobs**; expect four new fits, not six, when the two Bottle artifacts are
already correctly installed. Back up each `model.ckpt`, `report.json` and
`splits.json` **privately**.

Only when all six pairs are completed:

```bash
python -m scripts.summarize_comparison \
  --runs-dir /absolute/private/ihsanvision_runs \
  --from-saved-runs --categories bottle cable carpet --seed 42 \
  --output /absolute/private/portfolio-three-category.md
```

This rejects incomplete results and unfair model splits. It does not invent
accuracy numbers or compare against the official MVTec full-test benchmark.

For actual independently labeled held-out test predictions, prepare the
**private** seven-column CSV described in
[error analysis](docs/portfolio-error-analysis.md). Then run:

```bash
python -m scripts.audit_error_cases \
  --input /absolute/private/ihsanvision-test-predictions.csv \
  --categories bottle cable carpet \
  --output /absolute/private/portfolio-error-audit.md
```

Never publish private sample IDs, datasets, checkpoints, raw CSV or
unlicensed example images. Review genuine false-positive/false-negative
cases; do not invent an error when the sample has none.

## A2. On your Mac — independently verify the preserved Bottle archive

The original Bottle backup contains **eight checksum-listed files plus
`SHA256SUMS.txt`**. An extra `benchmark_summary.md` is **not required**:
the verifier reconstructs the paired comparison in memory. A later-created
smoke ZIP is also optional; if not supplied, the report explicitly states
that the smoke evidence was not checked. Use a **new private output path**:

```bash
python -m scripts.verify_portfolio_evidence \
  --runs-dir "$HOME/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15" \
  --output "$HOME/ihsanvision-artifacts/portfolio-bottle-audit-v1.json"
```

If you have the separately preserved **actual** PatchCore smoke ZIP, add
`--smoke-zip /absolute/private/actual-patchcore-smoke.zip` and save a
**different** audit file. Do not invent smoke evidence or overwrite existing
private reports. Passing the archive check does not fulfill P4–P10.

## B. On your Mac — real Bottle app end-to-end

Use your existing trusted Bottle artifacts stored **outside Git**:

```text
~/ihsanvision-artifacts/benchmark-bottle-seed42-2026-10-15/
  padim-bottle-seed42/model.ckpt
  patchcore-bottle-seed42/model.ckpt
```

The one-command Bottle launcher intentionally refuses an inherited
`IHSANVISION_MODEL_REGISTRY`: the registry takes precedence over the
explicit Bottle model environment paths and could otherwise silently run
different weights. If you previously configured a multicategory registry,
**unset it** in the Bottle demo terminal (`unset IHSANVISION_MODEL_REGISTRY`)
after verifying which models you intend to test. Keep any approved-model
manifest and API key deliberately configured; the launcher does not bypass
operator approvals. For an intentional multi-category deployment, run the
server using the reviewed registry instead of this Bottle-only launcher.

From the updated repo:

```bash
git pull --ff-only origin main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# Use a temporary/private operator DB for test records:
export IHSANVISION_HISTORY_DB="$HOME/ihsanvision-artifacts/portfolio-test-inspections.sqlite3"
export IHSANVISION_API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
export IHSANVISION_CLIENT_API_KEY="$IHSANVISION_API_KEY"
python -m scripts.run_local_demo --no-browser
```

In **another terminal** in the same venv, export the **same** private key via
a secure local method; do not paste or commit it into screenshots/issues.
Use real normal and anomalous Bottle inputs you have the right to inspect.
Store reports at **new** private paths; scripts do not overwrite old reports:

```bash
python -m scripts.local_stack_acceptance \
  --url http://127.0.0.1:8003 \
  --normal /absolute/private/normal_bottle.png \
  --defective /absolute/private/defective_bottle.png \
  --output "$HOME/ihsanvision-artifacts/portfolio-stack-v1.json" \
  --require-auth --operator-workflow
```

The optional operator mode creates and **targets deletion of only its own
five test inspections**; do this against a disposable/private test DB first.
If deletion fails, the command must fail: review the specific test rows,
never bulk-delete real operator data.

Manually check live **/**, **/batch**, **/history**: normal/defect overlays,
model switch, multiple-image batch, human verdict, lot grouping, CSV,
error states, phone-sized display and keyboard interaction. Confirm private
history survives a controlled service restart. Preserve a sanitized summary,
not real images, local file paths or API keys.

## C. On a Mac with Docker running — target image and real inference

The [successful GitHub cloud workflow](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/runs/39095740246) proved that the Linux CPU image builds and the no-model API starts,
but **did not** test your private models or target Mac Docker runtime. Review
[container instructions](docs/container-deployment.md).

```bash
cp .env.docker.example .env
# Privately edit .env: real absolute IHSANVISION_MODELS_DIR and a NEW long API key.
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/models
```

Using the private key configured in `.env`, run the real Bottle
`scripts.local_stack_acceptance` against
`http://127.0.0.1:8080` with `--require-auth` and a **new output path**.
Do not use `--operator-workflow` unless the Docker service also has a
deliberately configured private test history DB. Verify successful
predictions/overlays and restart/readiness behavior. Record memory footprint
and security settings; stop with `docker compose down`. Never claim P8
complete based only on a config check or a cloud image build with no weights.

## D. Make the real 60–120 second recruiter demo

Use [the exact demo storyboard](docs/portfolio-demo-storyboard.md). Capture
three authentic permission-cleared screenshots and one short **actual
software recording**. Show BOTH model families, a real normal and real
defective prediction, relative heatmaps, batch and human review. Add no
fake scores, fabricated overlays or unlicensed MVTec test images.

Before placing actual media inside `docs/media/` or `demo/media/`,
the owner must review rights and create
`docs/release-media-rights.md` listing each media file, original creator,
usage permission and publication decision. This document must describe
**actual permissions**, never placeholders. The GitHub source-hygiene CI
will reject media without a rights register.

## E. Release — only once every gate passes

A conservative, prewritten [v1.0.0 candidate release-note draft](docs/release-notes-v1-candidate.md)
is already in Git. It currently contains only **historical verified Bottle
observations** and explicitly lists P4–P10 evidence still missing. After all
real gate evidence exists, revise the draft to include the actual Cable/Carpet
metrics, successful current-build Mac/Docker validation, permission-cleared
media and legal review; only then copy the approved final notes into an
actual GitHub release. **Do not publish the draft as if v1.0 were finished.**


1. Recheck [fixed P1–P10 acceptance gates](docs/portfolio-v1-plan.md) and
   update [GitHub issue #53](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53)
   with dated private evidence/status and no secrets.
2. Run `python -m pytest -q`, dashboard tests, static
   `python -m scripts.check_release_hygiene --ci`, source/license review and
   final real Mac/Docker acceptance.
3. **Owner reviews the existing responsible-use source license**
   ([LICENSE](LICENSE)) and obtains qualified legal advice if needed. Audit
   separate MVTec, pretrained-weight and demo-media permissions. Having
   this source-code license does not authorize commercial MVTec data use.
4. Confirm README/demo links and all public claims match actual evidence.
   Review the whole Git history for accidental secrets or restricted assets.
5. Only then create the annotated `v1.0.0` tag and GitHub release from
   verified green `main` with the final changelog.

**Finish criterion:** All ten v1 gates actually verified and a reviewed
`v1.0.0` release exists. A merged PR, CLI command or mock test is never
a substitute for real evidence.
