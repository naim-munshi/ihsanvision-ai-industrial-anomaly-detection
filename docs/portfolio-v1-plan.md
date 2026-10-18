# IhsanVision.AI — fixed AI engineering portfolio v1.0 plan

**Frozen scope:** Three classic MVTec AD categories (**bottle, cable, carpet**) with two anomaly detectors (**PaDiM and PatchCore**); one usable operator inspection application; transparent evaluation, runtime evidence, documentation, permitted demo media and a reviewed GitHub release.

**Audience:** AI/ML Engineer and Applied ML Software Engineer interviewers. A successful repository must demonstrate reproducible ML methodology **and** the ability to ship/test model-backed software. Industrial/commercial certification is **not** claimed.

This is the **only v1.0 completion definition**. Additional MVTec AD categories, AD2, Kubernetes, account tenancy, public cloud SaaS, conveyor hardware, camera integration and commercial SLAs are explicitly **out of scope**. Do not add them to inflate progress. Expanding scope later requires a separately named version.

## Fixed acceptance gates

Each gate is either **Verified**, **Pending**, or **Blocked by external runtime/input**. Count only fully verified gates as completed. There is no made-up completion percentage or credit for simply opening a pull request.

| ID | Gate (v1.0 required) | Current evidence | Pass condition |
| --- | --- | --- | --- |
| P1 | Real paired Bottle research results | **Verified**: [PaDiM](../reports/2026-10-13-kaggle-padim-bottle.md), [PatchCore](../reports/2026-10-15-kaggle-patchcore-bottle.md) | Preserve original metrics, seed/split description and provenance; never represent these as full-test leaderboard scores |
| P2 | Bottle checkpoint-backed inference + heatmap | **Verified on prior Mac build**: [four-request acceptance](../reports/2026-10-16-macos-dual-model-api-acceptance.md) | Both model families route and return expected types, normal/defect labels and valid heatmaps for the documented manually chosen samples |
| P3 | Source quality and CI | **Verified on main**: Python 3.11/3.12, JS dashboards, static Docker contract | Main CI passes on final tag; regression/negative tests cover core contracts |
| P4 | Cable PaDiM and PatchCore | **Pending real training** | Both trusted checkpoint/report/splits trios exist; equal path manifests; four core held-out metrics and counts verified |
| P5 | Carpet PaDiM and PatchCore | **Pending real training** | Same evidence standard as P4; scores never invented |
| P6 | 3-category comparison and failure analysis | **Pending** (Bottle results available) | Machine-validated saved reports + per-category sample counts, image/pixel AUROC/F1 and documented actual error cases or an explicit statement that none were observed |
| P7 | Latest operator workflows in a real runtime | **Pending latest-code Mac run**; mocked integration tests pass | With trusted local weights: upload, normal/defect heatmap, batch, persisted history after restart, independent human review, lot grouping and private CSV all pass; record a no-secret acceptance log |
| P8 | Actual local Docker image acceptance | **Blocked by Docker/runtime + private weights**; static Compose CI passes | Image builds, non-root/read-only constraints hold, both Bottle checkpoints infer through container port, readiness and restart succeed |
| P9 | Recruiter-ready presentation and permitted media | **In progress**; documentation exists but no actual media published | Concise README, correct paired model card, architecture/evaluation/limitations, three approved screenshots (single result, heatmap comparison, batch/history), short real recording and clear setup |
| P10 | Versioned release, licensing and safety review | **Pending** | Operator chooses code license, reviews dataset/pretrained/image rights, no credentials or restricted assets in source/media, all P1–P9 gates passed; only then tag/release `v1.0.0` |

**Initial baseline:** P1, P2, P3 have previous documented real/CI evidence; P4–P10 are not fully verified. Some P7–P9 code exists, but that does not constitute real acceptance. The final v1.0 code must also pass the acceptance suite **after** changes are merged.

## Execution order — no extra feature work

1. **Code safety and documentation** — preserve run artifacts, fix stale model cards and docs, make README readable for a recruiter, maintain CI.
2. **Real Bottle end-to-end acceptance** — use local known-trust weights and existing scripts. Confirm batch/history/review/export/monitoring and capture *actual* screenshots only with reuse permission.
3. **Compute-limited 3-category benchmark** — keep existing Bottle; fit only missing cable/carpet jobs (four total), then verify split parity and compare real metrics.
4. **Deployable demonstration** — build the image in a real Docker environment, use private read-only checkpoint mounts and a non-public port; record startup/restart, health, memory and bounded requests.
5. **Portfolio publication** — publish only approved media and non-sensitive results, finalize selected source license, cut verified `v1.0.0` release.

### Training on a real Kaggle/compute session (only when provided)

Run dry-run first with an authorized dataset root and private runs folder **outside the dataset**. If the saved Bottle artifacts are restored to the same runs directory, they must be detected/reused rather than retrained:

```bash
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable carpet --seed 42
```

Inspect `training_jobs_pending` and ensure only cable/carpet pairs are pending. Execute at most one missing fit per invocation:

```bash
python -m scripts.resume_mvtec_training \
  --dataset-root /absolute/private/MVTecAD \
  --runs-dir /absolute/private/ihsanvision_runs \
  --categories bottle cable carpet --seed 42 \
  --accelerator auto --execute --max-jobs 1
```

Do not claim training happened merely because these commands appear in GitHub. On a completed experiment directory, the existing `scripts.summarize_comparison` verifies `comparison.json`, per-model reports and exact split equivalence before rendering a metrics table; see [benchmark summary](verified-benchmark-summary.md). All training checkpoints, raw images and private path manifests remain **outside Git**.

### Demonstration and permission policy

The final demo should cover: launch app → select Bottle/PaDiM → normal input → defect input with relative heatmap → switch PatchCore → batch/lot inspection → human review → metadata CSV. Record actual software, not simulated results. Use only self-created or explicitly permitted imagery; don't publish MVTec dataset photographs or masks without distribution rights. A short silent or captioned video is enough; there is no requirement for a hosted public ML inference endpoint. [Operator acceptance](operational-acceptance.md) · [Private history](private-inspection-history.md).

### Evidence hierarchy

- **Observed real model result:** original private run plus honest public methodology/report.
- **Automated tested implementation:** Python/JS GitHub CI (no loaded PyTorch checkpoint).
- **Real local integration:** dated, sanitized, reproducible output from Mac/real checkpoint.
- **Container acceptance:** actual Docker build + live checkpoint-backed HTTP call.
- **Not verified:** test script exists but operator has not executed it.

Keep these categories separate in README, model card, CV and interviews. Never extrapolate one-category AUROC into a universal industrial performance guarantee.

## Portfolio interview explanation (defensible)

> I built a reproducible industrial anomaly detection prototype with PaDiM/PatchCore, audited disjoint train/validation/test image paths, shipped model-backed FastAPI inference and a browser operator workflow, and measured warm-serving latency. The measured Kaggle results currently cover Bottle; the additional category experiments and latest deployment acceptance are separate release gates. The solution is supervised and is not production-certified.

Customize the final statement **only after** all release evidence exists. Avoid including unverified category scores or public hosting claims.

## Explicit non-goals for v1.0

Enterprise identity/RBAC, Kubernetes, fine-tuning arbitrary foundation models, MVTec AD2 leaderboard evaluation, automatic factory actuator control, real-time camera integration, LLM integration, monetization and hosting-dependent 24/7 SLAs. These are *not* reasons to delay a successful portfolio release.
