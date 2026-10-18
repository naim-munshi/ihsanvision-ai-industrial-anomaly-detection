"""Keep recruiter-facing Markdown readable; no network or model weights required."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANDING_DOCS = (
    "README.md", "FINISHING.md", "CONTRIBUTING.md", "CHANGELOG.md",
    "docs/roadmap.md", "docs/deployment.md", "docs/repository-structure.md",
    "docs/portfolio-v1-plan.md",
)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


@pytest.mark.parametrize("relative_path", LANDING_DOCS)
def test_recruiter_markdown_code_fences_are_well_formed(relative_path):
    """Reject accidental opening/closing fences with prose glued to them."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    opening = None
    for line_number, line in enumerate(source.splitlines(), 1):
        match = FENCE.match(line)
        if not match:
            continue
        fence, remainder = match.groups()
        if opening is None:
            opening = (fence, line_number)
        elif fence[0] == opening[0][0] and len(fence) >= len(opening[0]):
            assert not remainder.strip(), (
                f"{relative_path}:{line_number}: closing fence must stand alone"
            )
            opening = None
    assert opening is None, f"{relative_path}: unclosed fenced code block: {opening}"


def test_readme_has_clear_standalone_reproducibility_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Explore the project" in readme
    assert "## Reproduce the application locally" in readme
    assert "## Reproduce the experiments" in readme
    assert "## Portfolio release gates" in readme
    assert "``` Preserve the existing completed Bottle artifacts" not in readme
    assert "Preserve the existing completed Bottle artifacts" in readme
    assert "neither CI path executes checkpoint-backed" in readme.lower()


def test_current_status_documents_are_not_confused_with_full_factory_vision():
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    deployment = (ROOT / "docs/deployment.md").read_text(encoding="utf-8")
    assert "Historical milestones" in roadmap
    assert "portfolio v1.0 plan" in roadmap
    assert "not a hosted" in deployment
    assert "pre-multipart upload" in deployment
