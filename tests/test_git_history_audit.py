"""Offline Git-history audit tests using temporary synthetic commits only."""
from pathlib import Path
import subprocess

import pytest

from scripts.audit_git_history import audit, main


def command(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, timeout=20,
    )


def repo(tmp_path: Path) -> Path:
    command(tmp_path, "init", "-q")
    command(tmp_path, "config", "user.name", "CI Synthetic Test")
    command(tmp_path, "config", "user.email", "ci-example@example.invalid")
    (tmp_path / "README.md").write_text("# Safe source\n")
    command(tmp_path, "add", "README.md")
    command(tmp_path, "commit", "-qm", "Initial safe source")
    return tmp_path


def test_clean_history_has_no_known_patterns_but_never_certifies_release(tmp_path):
    root = repo(tmp_path)
    report = audit(root)
    assert report["status"] == "no_known_patterns_found"
    assert report["reachable_blob_count"] >= 1
    assert report["shallow_repository"] is False
    assert report["full_rights_or_secret_clearance_verified"] is False
    assert report["v1_portfolio_release_verified"] is False
    assert report["findings"] == []


def test_deleted_secrets_and_model_artifact_remain_detectable(tmp_path):
    root = repo(tmp_path)
    secret = "ghp_" + "A" * 42
    (root / ".env").write_text("GH_TOKEN=" + secret)
    (root / "weights").mkdir()
    (root / "weights" / "model.ckpt").write_bytes(b"small trained-model-placeholder")
    command(root, "add", ".env", "weights/model.ckpt")
    command(root, "commit", "-qm", "Accidental source objects")

    (root / ".env").unlink()
    (root / "weights" / "model.ckpt").unlink()
    command(root, "add", "-u")
    command(root, "commit", "-qm", "Removed private items from current source")

    report = audit(root)
    kinds = {finding["type"] for finding in report["findings"]}
    assert "private_environment_file" in kinds
    assert "private_dataset_or_artifact_folder" in kinds
    assert "potential_github_classic_token" in kinds
    assert report["status"] == "findings_require_review"
    # Never write token strings or deleted historical private paths to CI reports.
    printed = str(report)
    assert secret not in printed
    assert "weights/model.ckpt" not in printed
    assert ".env" not in printed


def test_historical_images_are_marked_for_license_review(tmp_path):
    root = repo(tmp_path)
    media = root / "docs" / "media"
    media.mkdir(parents=True)
    (media / "historic.png").write_bytes(b"example-binary-image")
    command(root, "add", "docs/media/historic.png")
    command(root, "commit", "-qm", "Old screenshot")
    assert "historical_media_requires_rights_review" in {
        finding["type"] for finding in audit(root)["findings"]
    }


def test_cli_preserves_existing_evidence_file_and_writes_new_report(tmp_path):
    root = repo(tmp_path)
    output = tmp_path / "audit.json"
    main(["--repo", str(root), "--output", str(output)])
    assert output.exists()
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        main(["--repo", str(root), "--output", str(output)])
    assert output.read_bytes() == original


def test_requires_real_git_checkout(tmp_path):
    with pytest.raises(ValueError, match="Git checkout"):
        audit(tmp_path)
