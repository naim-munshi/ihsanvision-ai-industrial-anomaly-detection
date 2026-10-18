"""Offline contract checks for portfolio documentation; no ML runtime or web calls."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_DOCUMENTS = (
    Path("README.md"),
    Path("docs/portfolio-v1-plan.md"),
    Path("docs/portfolio-release-readiness.md"),
    Path("docs/model-card.md"),
    Path("docs/evaluation.md"),
    Path("docs/architecture.md"),
    Path("docs/development.md"),
    Path("docs/verified-benchmark-summary.md"),
    Path("docs/resumable-multicategory-training.md"),
)
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^\)\n]+)\)")


@pytest.mark.parametrize("document", PORTFOLIO_DOCUMENTS)
def test_markdown_relative_links_resolve_inside_repository(document: Path) -> None:
    file = ROOT / document
    body = file.read_text(encoding="utf-8")
    assert body, document
    for uri in LINK.findall(body):
        # External sources and pure in-page anchors are not local artifact paths.
        destination = uri.split(" ", 1)[0].strip("<>")
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (file.parent / unquote(parsed.path)).resolve()
        assert path.is_relative_to(ROOT.resolve()), f"{document}: unsafe link {uri}"
        assert path.exists(), f"{document}: missing target {uri}"


def test_main_portfolio_scope_is_auditable_and_honest() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs/portfolio-v1-plan.md").read_text(encoding="utf-8")
    model = (ROOT / "docs/model-card.md").read_text(encoding="utf-8")
    for category in ("bottle", "cable", "carpet"):
        assert category in readme.lower()
        assert category in plan.lower()
    for family in ("PaDiM", "PatchCore"):
        assert family in readme
        assert family in model
    for measured in ("0.9963", "1.0000", "0.6497", "0.7102"):
        assert measured in readme
    # Explicit gaps prevent future documentation from silently overstating CI.
    assert "neither CI path executes checkpoint-backed PaDiM/PatchCore inference" in readme
    assert "not a production-certified" in readme
    assert "not yet executed" in readme.lower()
    assert "P10" in plan
    assert "four model/category runs" in (
        ROOT / "docs/portfolio-release-readiness.md"
    ).read_text(encoding="utf-8")
