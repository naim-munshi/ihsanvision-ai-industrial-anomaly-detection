"""No-runtime checks for public source release leakage and honest release status."""
from pathlib import Path

import pytest

from scripts.check_release_hygiene import audit_source


def source(tmp_path: Path, files: dict[str, bytes | str]) -> list[str]:
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return list(files)


def test_clean_source_is_not_misrepresented_as_a_finished_release(tmp_path):
    paths = source(tmp_path, {
        "README.md": "# Research portfolio, no production claims",
        "src/worker.py": "print('offline source')",
        ".env.docker.example": "EXAMPLE_VARIABLE=replace-me",
    })
    report = audit_source(tmp_path, paths)
    assert report["status"] == "source_hygiene_passed"
    assert report["tracked_files_checked"] == 3
    assert report["license_selected"] is False
    assert report["release_media_count"] == 0
    assert report["v1_portfolio_release_verified"] is False
    assert report["pending_external_release_gates"]


@pytest.mark.parametrize("filename", [
    "weights/model.ckpt", "model.pth", "data/bottle/test/good/000.png",
    "runs/padim/report.json", "db/inspection.sqlite3",
    ".env", "reports/private-evidence.zip",
])
def test_reject_tracked_private_artifacts(tmp_path, filename):
    paths = source(tmp_path, {filename: "private bytes"})
    report = audit_source(tmp_path, paths)
    assert report["status"] == "blocked"
    assert report["source_hygiene_findings"]


def test_private_tokens_are_detected_without_echoing_secret_bytes(tmp_path):
    secret = "ghp_" + "A" * 42
    aws = "AKIA" + "B" * 16
    path = "docs/leaked.md"
    names = source(tmp_path, {path: f"example {secret} and {aws}"})
    report = audit_source(tmp_path, names)
    joined = "\n".join(report["source_hygiene_findings"])
    assert "github_classic_token" in joined
    assert "aws_access_key" in joined
    assert secret not in joined and aws not in joined


def test_private_key_pem_is_flagged_without_logging_key(tmp_path):
    key = "-----BEGIN " + "PRIVATE KEY-----"
    names = source(tmp_path, {"docs/leak.md": key})
    report = audit_source(tmp_path, names)
    assert "private_key_material" in str(report["source_hygiene_findings"])


def test_media_needs_approved_destination_and_rights_register(tmp_path):
    names = source(tmp_path, {
        "docs/media/dashboard.png": b"fake-png-but-would-be-rejected-without-rights",
    })
    result = audit_source(tmp_path, names)
    assert result["status"] == "blocked"
    assert "rights register" in str(result["source_hygiene_findings"])

    names = source(tmp_path, {
        "docs/release-media-rights.md": "Owner must manually review actual rights",
    }) + names
    result = audit_source(tmp_path, names)
    assert result["status"] == "source_hygiene_passed"
    assert result["release_media_count"] == 1
    assert result["v1_portfolio_release_verified"] is False


def test_media_outside_designated_folder_is_rejected_even_with_register(tmp_path):
    names = source(tmp_path, {
        "random-test.png": b"irrelevant",
        "docs/release-media-rights.md": "Evidence to review",
    })
    report = audit_source(tmp_path, names)
    assert report["status"] == "blocked"
    assert "outside designated" in str(report["source_hygiene_findings"])


def test_path_traversal_duplicate_and_symlink_are_rejected(tmp_path):
    names = source(tmp_path, {"README.md": "Hello"})
    report = audit_source(tmp_path, names + ["README.md", "../outside.ckpt"])
    assert report["status"] == "blocked"
    assert "Invalid or duplicate" in str(report["source_hygiene_findings"])
    symlink = tmp_path / "linked.txt"
    symlink.symlink_to(tmp_path / "README.md")
    assert audit_source(tmp_path, ["linked.txt"])["status"] == "blocked"


def test_source_over_size_limit_requires_manual_review(tmp_path):
    names = source(tmp_path, {"README.md": b"x" * 1_000_001})
    result = audit_source(tmp_path, names)
    assert result["status"] == "blocked"
    assert "Oversized" in str(result["source_hygiene_findings"])


def test_adding_license_does_not_imply_real_release(tmp_path):
    names = source(tmp_path, {"LICENSE": "User-chosen license text"})
    result = audit_source(tmp_path, names)
    assert result["license_selected"] is True
    assert result["v1_portfolio_release_verified"] is False
