# ChatGPT Work Local — final five-category portfolio acceptance handoff

**Owner request (2026-10-20):** Do not send the owner through another series of manual tests. Delegate the remaining **real** UI, operator-workflow, failure-case, and Docker checks to an agent working with the owner's explicit local-file, localhost-browser and terminal permissions. Use **ChatGPT Work Local in the macOS desktop app** where available; **Codex Local** is another suitable agent for repository/terminal testing. ChatGPT Work **Cloud** and external browsers cannot connect to a Mac-only `127.0.0.1:8004` service by default. Do **not** expose the localhost API publicly or upload private model weights to make it reachable.

## Inputs already on the owner's Mac (not in this GitHub repository)

- Source: `~/ihsanvision-ai-industrial-anomaly-detection` (update `main`).
- Python venv: `~/ihsanvision-artifacts/venv-py312`.
- Trusted trained weights and saved reports: `~/ihsanvision-artifacts/portfolio-five-category-runs` (five categories × PaDiM/PatchCore; ten model.ckpt/report.json/splits.json triples).
- Private category/model registry: `~/ihsanvision-artifacts/portfolio-model-registry.json`.
- Private checkpoint approvals: `~/ihsanvision-artifacts/portfolio-approved-models.json`.
- Private key file: `~/ihsanvision-artifacts/portfolio-operator-key.txt` (mode 0600). The agent must never print, copy into Git, or paste its contents into a chat or issue.
- Authorized, locally extracted images: `~/ihsanvision-artifacts/private-ui-samples/{bottle,cable,carpet,metal_nut,capsule}/{normal,defective}.png`.
- Owner-run real-image API result report: `~/ihsanvision-artifacts/private-real-sample-acceptance.json` (private).
- Locally running trusted API: `http://127.0.0.1:8004` (if still running); it is not an internet URL.
- Optional original demonstration ZIP: `~/Downloads/ihsanvision-private-ui-samples.zip`. If the owner attaches a ZIP in chat, inspect it privately but **do not treat a ZIP alone as model checkpoint or localhost access**.

## Verified owner console evidence — do not repeat as new agent work

1. All ten model/category pairs trained on MVTec AD with owner-supplied executed logs. All eight non-Bottle checkpoint archives downloaded and SHA-256 verified on Mac; two original Bottle backups were verified separately.
2. The repository's read-only `scripts.summarize_comparison --from-saved-runs` verified all ten checkpoint/report/split triples and within-category PaDiM/PatchCore identical train/validation/test paths. This **does not** verify original dataset image hashes or rerun training.
3. `GET /ready` returned `status=config_ready`, `model_pairs=10`, `approved_sha256_verified=true`, `inference_verified=false` (readiness by design is not a prediction).
4. All **10/10** models subsequently passed **real checkpoint-backed localhost API inference** with a generated PNG and a valid PNG heatmap (Cable two in one earlier run; other eight in a later run).
5. The owner extracted ten original-MVTec-test-folder sample photos and ran **20/20 authenticated real-image HTTP API requests** (5 categories × 2 models × normal/defective); all request/PNG contracts passed, with **18 sample-kind label matches and two observed label mismatches**, not an accuracy rate.
6. Exactly two observed normal-image **false-positive candidates** in manually selected examples:
   - `metal_nut / padim / normal.png`: predicted anomaly `true`, raw score `0.6143437027931213`.
   - `capsule / padim / normal.png`: predicted anomaly `true`, raw score `0.5434991717338562`.
   Both defective counterparts were reported anomalous. Both PatchCore normal counterparts were reported normal. These are **not** proven calibration bugs or independent estimates of false-positive rate. Score scales must not be compared between models.
7. **NOT YET VERIFIED:** actual browser clicks, visual quality of heatmaps, batch page, actual persistent history/review/lot/CSV journey after restart, real checkpoint-backed Docker acceptance, legal/demo media approval, and final reviewed release.

## Agent objective

Complete remaining portfolio verification **without sending repeated shell commands to the owner**. Operate only with the permissions the owner grants. Preserve existing artifacts, DB records, trained checkpoints, project history, and uncommitted work. Report `PASS`, `FAIL`, or `BLOCKED` separately for each gate, with real observed evidence, not just the existence of tests.

### 1. Runtime and privacy preflight

- Confirm running `http://127.0.0.1:8004/health`, `/ready` and `/models` from the same **local Mac**. Assert ten approvals and both models for five categories. If local server is stopped, use the owner's existing private registry/approval/history/key settings to start the server **locally** on loopback. Do not print secret values. Do not overwrite private registry, checkpoints or logs. If the existing server is running, reuse it.
- Validate the existing 20-request private JSON report against its documented schema, keeping it off GitHub and not regenerating merely for decoration.
- Confirm private samples are truly distinct per category and are from the correct original dataset category. MVTec **test** directory samples may overlap the research evaluation splits; label them *manual demonstrations*, not independent validation.
- Run `pytest`, JS/browser tests and repo CI as appropriate. Use a separate branch and narrowly scoped PRs for fixes. Do not introduce irrelevant product features.

### 2. Visual real-image browser acceptance (the main missing gate)

Use locally available desktop/browser permissions (or an installed browser automation tool). If Work is running **in the cloud** and cannot access `127.0.0.1:8004`, mark this `BLOCKED` and propose moving to Work Local/Codex Local; **never open a public tunnel**.

At `http://127.0.0.1:8004/`, for each of 5 categories:
- Assert category selector has the five approved trained categories and both model choices.
- Provide the local key only to the local UI control; it must never appear in screenshots, logs, browser URLs, or public reports.
- For each PaDiM/PatchCore model, load its same-category normal and defective samples, use **Analyze image**, verify the selected category/model is echoed, no error appears, input preview is correct, result label and finite raw score appear and PNG overlay actually renders and is visually inspectable.
- Perform a total of **20 browser interactions** where practical (or document any subset/limitation precisely). Capture only private, rights-restricted screenshots; preferably capture no raw dataset images when posting public evidence. Cross-check browser decisions against the owner's private 20-request JSON. A model mismatch on those two PaDiM normal samples is a **documented model failure candidate**, not an API-routing failure.
- Inspect suspected failure-case overlays for `metal_nut/padim/normal` and `capsule/padim/normal`, noting only defensible visible observations (lighting, viewpoint, regions highlighted if actually visible); do not invent reasons or edit/force predictions.

### 3. Operator workflow and security

- With an explicit **disposable private test DB or carefully isolated lot**, verify `/batch` accepts multiple correct-category images, preserves model/category selection, shows individual decisions and downloads a valid CSV.
- Verify `/history`: unauthorized access denied, authenticated records and checkpoint identity available, change human review verdict, filter by synthetic opaque lot, lot aggregates, checkpoint-version groups, private CSV formula safety, and lock/reset behavior.
- Verify persisted test record survives **one real server restart** and read access still works after restart. Delete **only newly created test records**, never pre-existing records. Verify no accidental leakage of API key, raw images, checkpoint filesystem paths or private identifiers.
- Run existing `scripts.local_stack_acceptance` using operator's authorized Bottle samples, `--require-auth --operator-workflow` **only** with the owner's separate disposable history database and confirmation that its cleanup semantics are safe. Otherwise run safer individual tests and explicitly mark unverified pieces.
- Negative tests: missing/wrong key 401, rejected bad MIME/oversized or invalid uploads, unsupported category/model, throttling/admission as appropriate, no raw stack traces in browser.
- Capture browser screenshots/logs privately only. Never upload original MVTec images or any protected file to GitHub.

### 4. Docker acceptance if Docker daemon is available and owner permits local container operations

- Review `docs/container-deployment.md`, `compose.yaml` and `.env.docker.example` first. Run existing container test checks without changing the owner's unrelated containers or Docker projects.
- On Apple Silicon, note architecture and resource limits honestly. Build a CPU image only with necessary permission, register real checkpoints through a private read-only volume, ensure binding remains `127.0.0.1:8080`, non-root UID 10001, dropped capabilities and readonly filesystem. Confirm genuine CPU inference on authorized samples (not only `/ready`), cold/warm timings and clean restart, and isolate test history.
- If the existing Compose configuration only supports two Bottle weights, **do not misrepresent it as validated for ten models**; either create a scoped 5-category compose extension through a PR and test it, or report exactly the partial Docker scope.
- If Docker/CPU runtime unavailable, mark Docker `BLOCKED` with evidence. Never claim actual Docker success from CI static checks.

### 5. Deliverables and completion discipline

Produce a concise **private** acceptance report under the owner's `~/ihsanvision-artifacts` directory with:
- Matrix: model/category × two sample outcomes, API contract, UI rendering and heatmap review, with two PaDiM normal false-positive candidates.
- Browser UI, batch workflow, private history, security controls, restart persistence, Docker actual checkpoints, rights/demo images: separate `PASS`/`FAIL`/`BLOCKED`, commands/screenshot references and pertinent failures.
- Clear model limitations: one seed, MVTec original test-derived demo images, val held out from original test, no industrial certification, normalized heatmap is not a precise segmentation mask, uncalibrated scores.
- A minimal necessary PR for verified bugs only, with applicable GitHub CI status, preserving all private data. Review release version/scope because original `v1.0.0` text was 3-category frozen and was later amended to 5 categories.
- Public-safe text summary to [issue #53](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53) that does not reveal source photo data, secrets, exact local paths or private images.
- **Do not auto-tag or publish a release**, make public demo assets, modify credentials, retrain models, or commit checkpoints without owner review.

## Work Local / Codex Local instruction to execute this handoff

“Use the local folder `~/ihsanvision-ai-industrial-anomaly-detection` and the private assets already on this Mac. Read `docs/work-operator-full-acceptance-handoff.md` and current issue #53. Perform all safe unattended remaining acceptance tasks, including desktop browser inspection at localhost, with the permissions available. Fix actual reproducible bugs via PRs and return a fully evidenced pass/fail/blocked matrix. Do not ask me to type one command per step; pause only for required permission, authentication, or a truly unsafe/irreversible action. Do not print or transmit keys, MVTec photos, model weights or private paths. Don't claim the project is release-ready until every required gate has actual evidence.”
