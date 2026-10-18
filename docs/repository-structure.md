# Repository structure and branch governance

## Current GitHub branch audit (2026-10-18)

The default branch is **`main`** and current CI is green. A read-only GitHub
audit found **67 remote branches**, **no open PRs**, and approximately
**60 merged PRs**. Many merged topic branches were retained because repository
setting `delete_branch_on_merge` is **false**. GitHub reported `main` as
**not protected**, with no repository rulesets. The project's PR review policy
is documented below but is **not technically enforced** until the owner
configures branch protection.

**Safe owner cleanup:** In the repository's **Settings → General → Pull
Requests**, enable *Automatically delete head branches* for future PRs.
For existing branches, use the PR's **Delete branch** action **only on
merged PRs after confirming the head branch is no longer being used**.
Preserve `main`, historical `MdNaim`, and
`feature/portfolio-production-pipeline` until the owner's old Mac/Kaggle
checkouts have been migrated. Deleting a merged topic branch does not
unmerge the PR, but any unmerged work on a reused branch must be preserved
first. Do not force-push or bulk-delete branches merely because they share
a name with a historical PR.

In **Settings → Rules → Rulesets** (or **Branches → Add branch protection
rule**, depending on the GitHub UI), protect `main` with required PR review
and relevant checks from [CI](../.github/workflows/ci.yml):
`unit-tests (3.11)`, `unit-tests (3.12)`, `source-hygiene`,
`container-config`. Restrict force pushes/deletion; avoid requiring the
heavy optional Docker build on every documentation-only change. The owner
must choose and apply these settings in GitHub; changing Markdown does not
change GitHub repository settings.

## Architecture decision (2026-10-14)

The historical repository used the Python package name `src`. This made imports like `src.api` work but is not a meaningful library/module namespace.

The project now uses the canonical import namespace `ihsanvision`. The code was **moved without changing algorithm logic** to `ihsanvision/{api,schemas,serving,predict,train,benchmark,config}.py`. The historical `src/*.py` modules are intentionally small compatibility entry points so existing Mac commands, Kaggle notebooks and offline tests continue to work. Do not put new business logic in `src`.

`scripts/` contains operational commands, `tests/` isolated tests, `demo/` static frontend, `docs/` maintainable guidance, `reports/` observed (not fabricated) evidence, and `notebooks/` prior exploration.

`pyproject.toml` is the package and CLI-entrypoint manifest; `requirements.txt` pins the compatible runtime baseline, and `requirements-dev.txt` keeps GitHub CI independent of huge machine-learning downloads. Keep runtime version constraints aligned when updating any of these.

## Standard branch policy

- `main` — **verified current GitHub default branch** and integration/release source. Changes should reach it through reviewed pull requests with green CI; branch protection is **not yet enforced in GitHub settings**.
- `feature/<short-task>` — focused work branches originating at the latest `main`, opened as PRs back into `main`.
- `MdNaim` — legacy default name from the initial repository setup, retained until migration and review are completed. Do not add new commits to it.
- `feature/portfolio-production-pipeline` — historical integrated development work; PR #2 has now been merged into `main`; the historical feature branch remains for traceability. The branch can remain for traceability after merging.

**Important:** GitHub's actual default branch is repository *metadata*, not determined by a `README.md` or `git branch -M` command on the maintainer's Mac. Setting a local branch name does **not** change the GitHub default.

### Default branch verified

GitHub now reports **`main` as the default branch**. PR #2 and PR #3 are merged, and Python 3.11/3.12 CI passed on the integrated source. Preserve the old branches until any remaining single-branch Mac/Kaggle checkouts have migrated; there is no need to rewrite Git history.

### Local Mac migration

The established Mac project at `~/ihsanvision-portfolio` initially tracked a single development feature branch. Do not overwrite or delete that checkout simply because `main` is now default. See [single-branch fetch and SSL setup troubleshooting](mac-environment-troubleshooting.md).

For an existing checkout that needs the new canonical package:

~~~bash
cd ~/ihsanvision-portfolio
git status -sb
git fetch origin main:refs/remotes/origin/main
git switch -c main --track origin/main  # only if local main does not exist
git pull --ff-only
source .venv/bin/activate
python -m compileall -q ihsanvision src scripts tests
~~~

Use `python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8002` when running from the updated environment; old `src.api:app` remains a compatibility route. If the port is taken, inspect its listener instead of terminating unknown processes.

When ready to move development work to a new clean checkout on the default branch:

~~~bash
git clone https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection.git
cd ihsanvision-ai-industrial-anomaly-detection
git branch --show-current
~~~

The default is verified as `main`; in single-branch historical clones, local refspecs may still need an explicit fetch.

## Tests and release gates

- A clean packaging smoke test (`pip install -e . --no-deps`) and Python 3.11/3.12 unit tests must pass before integrating structural changes.
- The existing PaDiM training, FastAPI local demo, saved model cache and comparison logic are covered by observed experiments and mocked unit tests, as documented.
- **Bottle PatchCore results are observed and reported**, whereas Cable/Carpet training, current Mac operator acceptance and trusted-checkpoint Docker inference are pending. Public multi-user hosting is outside portfolio v1.0.
- Preserve source history and training artifacts outside Git; never force-push protected or shared branches during a rename.

## What is **not** included in the reorganization

No dataset or model checkpoints were moved/committed. No actual MVTec images were redistributed, the Kaggle notebook was not rewritten, and the default-branch switch was performed separately by the repository owner. No fabricated experiment metrics were introduced.
