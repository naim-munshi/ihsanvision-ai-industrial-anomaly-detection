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
