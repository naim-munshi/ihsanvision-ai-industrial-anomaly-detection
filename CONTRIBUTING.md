# Contributing to IhsanVision.AI

This repository contains a research/portfolio prototype, **not** safety-certified manufacturing inspection software.

## Git workflow

1. Base each focused `feature/<task>` branch on the latest `main`.
2. Keep commits small, descriptive, and reviewable. Do not force-push shared branches or push directly to `main`.
3. Open a pull request targeting `main` with a problem statement, test steps, before/after behavior, limitations and relevant issue references.
4. Keep the PR in draft until the requested functionality and tests are complete. Review CI on Python 3.11 and 3.12 before requesting merge.
5. After a PR is merged and any independent work is preserved, delete its short-lived topic branch. Preserve legacy `MdNaim` and `feature/portfolio-production-pipeline` until their historical Mac/Kaggle uses are resolved. GitHub's automatic head-branch deletion and `main` protection are **currently disabled**; the owner should enable them through [the branch governance guide](docs/repository-structure.md).

## Python application layout

- Add new application code only to **`ihsanvision/`**, importing from `ihsanvision.*`.
- **`src/` is compatibility-only** for existing Mac commands, notebooks and consumers; do not duplicate model code there.
- Add scripts/workflow helpers to `scripts/`, offline regression tests to `tests/`, verified experiment evidence to `reports/`, and explanations to `docs/`.
- Use `pyproject.toml` for packaging metadata/console scripts. Keep compatible runtime constraints in `requirements.txt`.
- Avoid checking in large weights, images, dataset copies, tokens, generated outputs, notebook checkpoints or environment files.

**Scope discipline:** The current `v1.0` goal is the [fixed AI Engineer portfolio](docs/portfolio-v1-plan.md): Bottle/Cable/Carpet, PaDiM/PatchCore, real operator demo, reproducible evaluation and release evidence. Avoid adding enterprise SaaS or speculative feature work. A PR with no new measured model evidence must not claim a completed model or production deployment.

## Local checks

~~~bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m compileall -q ihsanvision src scripts tests
python -m pytest -q
~~~

For a real model change, *also* execute the appropriate trusted-checkpoint or dataset test in a separate environment. Green offline CI alone does not establish actual GPU/CPU model behavior.

## Review checklist

- Does the change preserve previously verified checkpoint compatibility?
- Does a dataset or training change audit and preserve disjoint train/validation/test splits?
- Are all metrics accompanied by dataset category, split, threshold protocol, versions and hardware?
- Are score/heatmap semantics precise (raw score ≠ probability, overlay ≠ segmentation truth)?
- Do API errors avoid disclosing internal paths and secrets? Are uploads bounded and validated?
- Are model dependencies, `pyproject.toml`, and CLI commands consistent?
- Has the documentation distinguished measured results from pending work?
- Did automated tests pass on Python 3.11 and 3.12?

## Security and licensing

Do not paste user secrets or model artifacts into public issues or PRs. Coordinate security disclosures privately with the repository maintainer. Review the MVTec AD dataset license before redistribution or commercial use.

The original repository code is offered under the [IhsanVision.AI Responsible Use License v1.0](LICENSE), a **source-available, use-restricted, non-OSI** license. Harmful uses are prohibited, and changes distributed to others must preserve the terms for covered source code. Contributions require the contributor's permission for the applicable licensing; third-party dataset, model or dependency licenses remain separate. See [the licensing guide](docs/licensing.md).
