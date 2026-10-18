# Portfolio source release hygiene — automatically checked in GitHub Actions

`python -m scripts.check_release_hygiene --ci` examines **Git-tracked
source files** only, before anyone distributes the portfolio source code.
It is a cheap **static check**; it never trains, loads checkpoints, connects
to the internet, builds Docker or contacts the owner's Mac.

## What this check blocks

- Committed checkpoint/weight formats such as `.ckpt`, `.pth`, `.pt`,
  `.onnx`, plus private archives, SQLite databases and credential key files.
- Files tracked from `data/`, `datasets/`, `runs/`, `outputs/`,
  `results/`, `private/`, `weights/` or `checkpoints/`.
- Real tracked `.env` configuration files (documented `.env.example`
  and `.env.docker.example` examples are permitted).
- A few **high-confidence** GitHub/AWS tokens and PEM private-key signatures
  inside tracked UTF-8 source. The matched secret **bytes are never printed**
  in the finding.
- Oversized/unknown binary source or tracked symlink entries requiring review.
- Demo media outside `docs/media/` or `demo/media/`. Any committed media
  needs a *separate, owner-approved* rights register at
  `docs/release-media-rights.md`.

It prints a structured result and refuses GitHub CI if the public source
hygiene conditions fail. It deliberately reports
`v1_portfolio_release_verified: false` **even when source hygiene passes**.
An existing `LICENSE` file, permission-cleared media, real six-model benchmark
evidence, Mac inference, real Docker build and a tagged release are
**separate acceptance gates**; the script does not infer them from filenames.

This check is **not a full secret scanner**, vulnerability scanner, software
bill of materials, legal license clearance, or independently verified human
approval. New secret formats, steganographic material, base64 credentials and
historical Git objects can evade it. Before release, the owner must also
review the **full Git history**, dependencies, upstream terms, data and
media provenance, and the actual source archive. If a real token was ever
committed, removing it from HEAD does not revoke it; rotate it immediately.

## Historic commits and deleted assets

The normal CI source-hygiene check only inspects Git **HEAD**. A deleted
credential or trained checkpoint might still be retrievable from earlier
commits. Before the v1.0 release, run the separate
[Git history audit](../.github/workflows/git-history-audit.yml) under
**Actions → Review reachable Git history for release → Run workflow**.

This workflow performs a full-history checkout and fetches current remote
branches and tags. The offline
[`scripts.audit_git_history`](../scripts/audit_git_history.py) scans:

- All historical filenames reachable through locally fetched refs for
  previously committed model weights, SQLite files, dataset directories,
  environment files and media requiring redistribution rights review.
- Small/medium historical Git blobs for a deliberately narrow set of
  high-confidence leaked API key/private-key patterns, even when a sensitive
  file was deleted later.
- Shallow history, missing content coverage and oversized blobs are reported
  as limitations; a missing history cannot be declared clean.

The report **does not print secret bytes, original private paths, or
customer filenames**: it uses anonymous path references and shortened Git
object hashes. Review the **actual GitHub Actions result and its 14-day
artifact**. No passing script, even on a complete checkout, proves full
license approval or absence of every secret type. Any potentially exposed
credential must be rotated, and any restrictive historical media requires
human review. Do not force-push to erase evidence or rewrite shared branches
without an explicit separately approved remediation plan.

The workflow runs on a change to its script/workflow or can be manually
re-run before the final release. It is **not** part of the fast Python
unit-test matrix and does not load ML checkpoints.

## Owner's final publication steps

1. Produce the actual Bottle/Cable/Carpet evaluations and the private error
   audit; do not check private CSV, model checkpoints or dataset images into
   Git.
2. Capture three real **rights-cleared** screenshots (single-result overlay,
   model switch, batch/history) and a short screen recording. Keep them
   outside Git until permission and privacy have been reviewed. Store
   approved media only in the designated folders and document the owner,
   source, usage permission and publication date in
   `docs/release-media-rights.md`. A file existing in that location does not
   itself prove legal permission.
3. The owner **reviews the existing responsible-use source license** and
   verifies their copyright/contribution rights, MVTec and third-party
   obligations. The static source checker cannot approve legal terms.
4. Run the static check, Python/JS tests, real Mac and Docker acceptance,
   benchmark validation, and inspect release archives for secrets and
   restricted assets.
5. Only then publish `v1.0.0` and close
   [fixed acceptance tracker](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/issues/53).

For evaluation-specific evidence and safeguards, see
[portfolio v1.0 definition](portfolio-v1-plan.md) and
[release readiness](portfolio-release-readiness.md).
