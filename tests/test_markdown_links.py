"""Prevent broken links in every public IhsanVision portfolio Markdown file."""
from pathlib import Path

from scripts.check_markdown_links import audit_markdown, local_target


def test_all_portfolio_markdown_local_links_resolve():
    root = Path(__file__).resolve().parents[1]
    assert audit_markdown(root) == []


def test_checker_skips_fenced_shell_examples_and_external_urls(tmp_path):
    doc = tmp_path / "README.md"
    target = tmp_path / "docs" / "valid.md"
    target.parent.mkdir()
    target.write_text("# Good")
    doc.write_text(
        "# Portfolio\n"
        "[local](docs/valid.md) and [homepage](https://example.org)\n"
        "Here is an [in-page](#section).\n"
        "```bash\n"
        "# example with [nonexistent](private_checkpoint.ckpt)\n"
        "```\n"
        "An inline `[literal](unreal.md)` must be ignored.\n"
    )
    assert audit_markdown(tmp_path) == []


def test_checker_detects_broken_inline_and_reference_links(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Landing\n"
        "Read [missing](docs/missing.md).\n"
        "[appendix]: docs/other_missing.md\n"
    )
    failures = audit_markdown(tmp_path)
    assert len(failures) == 2
    assert "does not exist" in failures[0][3]
    assert "does not exist" in failures[1][3]


def test_checker_blocks_repo_escape_and_absolute_local_paths(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("[unsafe](../../outside.md)\n[absolute](/etc/passwd)\n")
    failures = audit_markdown(tmp_path)
    assert len(failures) == 2
    assert "escapes" in failures[0][3]
    assert "absolute" in failures[1][3]


def test_url_with_fragment_still_checks_its_local_file(tmp_path):
    doc = tmp_path / "README.md"
    guide = tmp_path / "docs" / "guide.md"
    guide.parent.mkdir()
    guide.write_text("# Heading")
    doc.write_text("[guide](docs/guide.md#heading)\n")
    assert audit_markdown(tmp_path) == []
    assert local_target(tmp_path, doc, "docs/guide.md#heading")[1] is None
