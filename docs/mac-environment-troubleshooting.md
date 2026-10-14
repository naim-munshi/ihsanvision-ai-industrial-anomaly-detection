# macOS checkout and installation troubleshooting

This document addresses the observed MacBook Air / Python 3.12.4 checkout after IhsanVision.AI switched its GitHub default branch to `main`. Work within the existing checkout; do not delete local checkpoints, experiment reports, the virtual environment, or the old feature branch.

## 1. `fatal: invalid reference: main` after `git fetch origin`

The historical clone tracked a single feature branch and did not automatically create a local `main` or fetch its remote-tracking ref. A failed `git switch main` leaves the existing branch selected. The following `git pull --ff-only` would consequently update the **old feature branch**, not `main`.

From the existing checkout:

~~~bash
cd ~/ihsanvision-portfolio
git status -sb
git fetch origin main:refs/remotes/origin/main
git branch -a
~~~

If `git status -sb` shows no local modifications, **and `git branch` does not already list `main`**, create a local tracking branch:

~~~bash
git switch -c main --track origin/main
git pull --ff-only
git status -sb
~~~

If local `main` already exists, use `git switch main` and `git pull --ff-only` instead. Never force-reset or delete the old branch to fix a missing ref. On a separate clean checkout, ordinary `git clone` now uses the verified GitHub default `main`.

## 2. Editable `pip install -e . --no-deps` fails on SSL during setuptools install

The observed message included `CERTIFICATE_VERIFY_FAILED`, `self-signed certificate in certificate chain` and retry attempts against `https://pypi.org/simple/setuptools/`. Because pip **isolates its build environment by default**, `--no-deps` alone does *not* prevent pip from downloading PEP 517 build dependencies (`setuptools>=69` and `wheel` from `pyproject.toml`).

You do **not** need an editable package install merely to run the application from the repository root; the canonical `ihsanvision` directory is importable directly:

~~~bash
cd ~/ihsanvision-portfolio
source .venv/bin/activate
python -c "from ihsanvision.api import app; print(app.title)"
python -m compileall -q ihsanvision src scripts tests
~~~

If you need the editable `ihsanvision-train` etc. console scripts, **first inspect the already installed local build tools**:

~~~bash
python -m pip show setuptools wheel
~~~

Only when suitable versions of `setuptools>=69` and `wheel` are already installed in this virtual environment, you can try the build **without network build-isolation downloads**:

~~~bash
python -m pip install -e . --no-deps --no-build-isolation
~~~

If setuptools/wheel is missing or too old, do not blindly bypass the build requirements. Fix access to your *trusted* package index or try a network without HTTPS interception and install the missing packages normally. On institution-managed networks, request the official institutional root CA and configure Python/pip using the institution's security instructions. **Do not use** `--trusted-host`, `--cert ""`, or disable TLS certificate verification as a workaround.

## 3. `No module named pytest`

`pytest` is a developer dependency, **not** a core runtime dependency; having `requirements.txt` installed does not necessarily install it.

~~~bash
python -m pip show pytest
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

On the same intercepting network, the install may fail for the same certificate reason. Use a trusted network (for example a working personal hotspot), or resolve the certificate chain with your administrator. The project's **GitHub Actions** independently runs the offline unit tests on Python 3.11 and 3.12. Skipping local pytest does *not* prove that a fresh editable install worked on your Mac.

## 4. Port already in use

If Uvicorn fails with `[Errno 48] address already in use`:

~~~bash
lsof -nP -iTCP:8002 -sTCP:LISTEN
~~~

If the process is an already-running project Uvicorn server, stop it from its own Terminal using Control-C and restart after pulling updates. If it is unfamiliar, do not kill it without inspection; pick another free port, and match it in the smoke-test URL. Keep the server Terminal running while issuing smoke requests from another Terminal.

## 5. Non-destructive verification of the canonical code

From the updated `main` checkout, with the already installed runtime requirements:

~~~bash
python -c "from ihsanvision.api import app; print(app.title)"
python -m scripts.local_preflight
~~~

On your Mac, the verified checkpoint was stored at `~/ihsanvision-artifacts/padim-bottle-seed42/model.ckpt`. This artifact **is not in GitHub** and must not be deleted during repository cleanup.

For a complete model API smoke test, export the trusted checkpoint path, start Uvicorn in one terminal, and use `python -m scripts.local_api_smoke --url http://127.0.0.1:8002` in another. See [deployment notes](deployment.md).

**Do not** use `git clean -fdx` or `rm -rf` on the source checkout to solve these issues. These commands may delete virtual environments, experiment results, or other untracked local work.
