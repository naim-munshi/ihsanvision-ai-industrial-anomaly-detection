"""Safe GitHub branch lifecycle tests; no API access or deletions in CI."""
import pytest

from scripts import prune_merged_branches as prune


def branch(name, sha="a" * 40, protected=False):
    return {"name": name, "commit": {"sha": sha}, "protected": protected}


def pull(name, sha="a" * 40, *, repo=prune.REPOSITORY,
         merged="2026-10-18T01:00:00Z"):
    return {
        "merged_at": merged,
        "head": {"ref": name, "sha": sha, "repo": {"full_name": repo}},
    }


def test_only_unchanged_same_repo_merged_pr_heads_are_eligible():
    branches = [
        branch("main"),
        branch("MdNaim"),
        branch("feature/portfolio-production-pipeline"),
        branch("feature/merged-ok"),
        branch("security/completed-audit"),
        branch("fix/updated-after-merge", "b" * 40),
        branch("docs/unmerged"),
        branch("chore/protected", protected=True),
        branch("ci/open-pr"),
        branch("arbitrary-branch"),
        branch("feature/../main"),
    ]
    history = [
        pull("feature/merged-ok"),
        pull("security/completed-audit"),
        pull("fix/updated-after-merge"),
        pull("chore/protected"),
        pull("ci/open-pr"),
        pull("feature/portfolio-production-pipeline"),
        pull("feature/merged-ok", repo="other-owner/repo", merged="2026-10-19T01:00:00Z"),
    ]
    open_prs = [pull("ci/open-pr", merged=None)]
    assert prune.candidates(branches, history, open_prs) == [
        ("feature/merged-ok", "a" * 40),
        ("security/completed-audit", "a" * 40),
    ]


def test_latest_merged_pr_revision_controls_reused_branch():
    name = "feature/reused"
    history = [
        pull(name, sha="a" * 40, merged="2026-10-17T01:00:00Z"),
        pull(name, sha="b" * 40, merged="2026-10-18T01:00:00Z"),
    ]
    assert prune.candidates([branch(name, "a" * 40)], history, []) == []
    assert prune.candidates([branch(name, "b" * 40)], history, []) == [
        (name, "b" * 40)
    ]


def test_missing_or_malformed_metadata_is_ignored():
    assert prune.candidates([
        branch("fix/live"), branch("chore/weird", protected=None),
        {"name": "docs/missing", "protected": False, "commit": {}},
    ], [{"merged_at": "x", "head": None}, {"head": {}}], []) == []


def test_default_cleanup_is_dry_run_and_does_not_call_delete(monkeypatch):
    selected = [branch("fix/merged")]
    monkeypatch.setattr(prune, "_fetch_all", lambda *_: selected)
    monkeypatch.setattr(
        prune, "_fetch_pulls",
        lambda _token, _root, state: [pull("fix/merged")] if state == "closed" else [],
    )
    requests = []
    monkeypatch.setattr(
        prune, "_request_api", lambda *args, **kwargs: requests.append((args, kwargs))
    )
    report = prune.cleanup("fake-token", apply=False)
    assert report["eligible_merged_unchanged_branches"] == 1
    assert report["deleted_branches"] == []
    assert report["mode"] == "dry_run"
    assert requests == []


def test_apply_revalidates_sha_before_targeted_delete(monkeypatch):
    monkeypatch.setattr(
        prune, "_fetch_all",
        lambda *_: [branch("fix/unchanged"), branch("docs/changed")],
    )
    monkeypatch.setattr(
        prune, "_fetch_pulls",
        lambda _token, _root, state: (
            [pull("fix/unchanged"), pull("docs/changed")]
            if state == "closed" else []
        ),
    )
    deleted = []

    def fake_api(_token, route, method="GET"):
        if method == "DELETE":
            deleted.append(route)
            return None
        if route.endswith("docs%2Fchanged"):
            return branch("docs/changed", "b" * 40)
        return branch("fix/unchanged")

    monkeypatch.setattr(prune, "_request_api", fake_api)
    result = prune.cleanup("fake-token", apply=True)
    assert result["deleted_branches"] == ["fix/unchanged"]
    assert result["skipped_changed_or_protected"] == ["docs/changed"]
    assert deleted == [
        f"repos/{prune.REPOSITORY}/git/refs/heads/fix/unchanged"
    ]


def test_no_token_refuses_without_touching_remote():
    with pytest.raises(ValueError, match="token required"):
        prune.cleanup("")
