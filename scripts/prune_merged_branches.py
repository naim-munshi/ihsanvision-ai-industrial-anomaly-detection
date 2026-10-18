"""Safely prune only unchanged, already-merged GitHub PR head branches.

Designed for a GitHub Actions job after main is updated. It NEVER deletes:
main, historical migration branches, protected branches, open PR heads, or a
branch whose current commit differs from the final merged PR head. GitHub PR
history and squash-merge commits are retained. A dry-run is the default.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from urllib import error, parse, request

REPOSITORY = "naim-munshi/ihsanvision-ai-industrial-anomaly-detection"
KEEP = {"main", "MdNaim", "feature/portfolio-production-pipeline"}
TOPIC_PREFIXES = ("feature/", "feat/", "fix/", "docs/", "ci/", "chore/")
VALID_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9/_.-]{1,150}\Z")
MAX_PAGES = 10
MAX_DELETE = 80


def candidates(branches: list[dict], closed: list[dict],
               open_pulls: list[dict], repo: str = REPOSITORY) -> list[tuple[str, str]]:
    """Select HEADS whose exact current SHA is still the merged PR HEAD SHA.

    Pure helper; incomplete/foreign/malformed response records are ignored.
    """
    open_heads = {
        pull.get("head", {}).get("ref")
        for pull in open_pulls
        if (pull.get("head") or {}).get("repo") and
        pull["head"]["repo"].get("full_name") == repo
    }
    latest: dict[str, dict] = {}
    for pull in closed:
        if not pull.get("merged_at"):
            continue
        head = pull.get("head") or {}
        name = head.get("ref")
        head_repo = head.get("repo") or {}
        sha = head.get("sha")
        if (not isinstance(name, str) or not isinstance(sha, str)
                or head_repo.get("full_name") != repo):
            continue
        old = latest.get(name)
        if old is None or pull["merged_at"] > old["merged_at"]:
            latest[name] = {"merged_at": pull["merged_at"], "sha": sha}

    selected = []
    for branch in branches:
        name = branch.get("name")
        if (not isinstance(name, str) or not VALID_BRANCH.fullmatch(name)
                or ".." in name or name in KEEP or name in open_heads
                or not name.startswith(TOPIC_PREFIXES)
                or branch.get("protected") is not False):
            continue
        recorded = latest.get(name)
        current = (branch.get("commit") or {}).get("sha")
        if recorded and current and current == recorded["sha"]:
            selected.append((name, current))
    return sorted(set(selected))


def _request_api(token: str, url: str, method: str = "GET"):
    api_base = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if not api_base.startswith("https://"):
        raise ValueError("GitHub API must use HTTPS")
    req = request.Request(
        api_base + "/" + url.lstrip("/"),
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "IhsanVision-repository-housekeeping",
        },
        method=method,
    )
    with request.urlopen(req, timeout=25) as reply:
        if method == "DELETE":
            return None
        return json.loads(reply.read(5_000_000))


def _fetch_all(token: str, path: str) -> list[dict]:
    result: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        data = _request_api(token, f"{path}?per_page=100&page={page}")
        if not isinstance(data, list):
            raise ValueError("Unexpected GitHub list response")
        result.extend(data)
        if len(data) < 100:
            return result
    raise RuntimeError("GitHub history exceeds bounded pagination; refusing cleanup")


def cleanup(token: str, *, apply: bool = False) -> dict:
    """Double-check a current branch's commit SHA immediately before DELETE."""
    if not token:
        raise ValueError("GitHub Actions token required")
    root = f"repos/{REPOSITORY}"
    branches = _fetch_all(token, root + "/branches")
    closed = _fetch_all(token, root + "/pulls?state=closed") if False else None
    # Add query parameters explicitly so every page retains state=closed.
    closed = _fetch_all(token, root + "/pulls") if False else _fetch_pulls(token, root, "closed")
    open_pulls = _fetch_pulls(token, root, "open")
    selected = candidates(branches, closed, open_pulls)
    if len(selected) > MAX_DELETE:
        raise RuntimeError("Too many eligible refs; manual review required")
    deleted: list[str] = []
    skipped: list[str] = []
    for name, sha in selected:
        if not apply:
            continue
        # Re-check exact live SHA and current branch protection before
        # deletion. The initial candidate scan is never sufficient alone.
        encoded = parse.quote(name, safe="")
        current = _request_api(token, f"{root}/branches/{encoded}")
        if (not isinstance(current, dict) or current.get("protected") is not False
                or (current.get("commit") or {}).get("sha") != sha):
            skipped.append(name)
            continue
        ref = parse.quote(name, safe="/")
        _request_api(token, f"{root}/git/refs/heads/{ref}", method="DELETE")
        deleted.append(name)
    return {
        "mode": "apply" if apply else "dry_run",
        "eligible_merged_unchanged_branches": len(selected),
        "eligible_names": [name for name, _ in selected],
        "deleted_branches": deleted,
        "skipped_changed_or_protected": skipped,
        "preserved_branches": sorted(KEEP),
    }


def _fetch_pulls(token: str, root: str, state: str) -> list[dict]:
    all_pulls: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        response = _request_api(
            token, f"{root}/pulls?state={state}&per_page=100&page={page}"
        )
        if not isinstance(response, list):
            raise ValueError("Unexpected GitHub PR response")
        all_pulls.extend(response)
        if len(response) < 100:
            return all_pulls
    raise RuntimeError("Too many pull requests for bounded cleanup")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete only the verified eligible merged PR heads")
    args = parser.parse_args(argv)
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        report = cleanup(token, apply=args.apply)
    except (OSError, error.URLError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Safe merged-branch pruning refused: {exc}") from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
