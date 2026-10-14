"""Repository structure/documentation checks; no model weights or network needed."""
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_package_and_compatibility_are_both_preserved():
    canonical = ROOT / "ihsanvision"
    legacy = ROOT / "src"
    for name in ("api", "benchmark", "config", "predict", "schemas", "serving", "train"):
        original = canonical / (name + ".py")
        shim = legacy / (name + ".py")
        assert original.is_file(), f"missing canonical module: {original}"
        assert shim.is_file(), f"missing legacy compatibility shim: {shim}"
        assert "ihsanvision." + name in shim.read_text()


def test_docs_internal_links_resolve():
    markdown_files = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "reports").rglob("*.md"),
        *(ROOT / ".github").rglob("*.md"),
    ]
    failures = []
    for source in markdown_files:
        text = source.read_text(encoding="utf-8")
        # Read local Markdown file links; skip external URLs and fragment-only links.
        for match in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
            uri = match.group(1)
            if uri.startswith(("https://", "http://", "mailto:", "#", "data:")):
                continue
            relative = unquote(uri.partition("#")[0])
            if not relative:
                continue
            destination = (source.parent / relative).resolve()
            if not destination.is_file():
                failures.append(f"{source.relative_to(ROOT)}: {uri}")
    assert not failures, "Broken local Markdown links:\n" + "\n".join(failures)


def test_no_weights_or_dataset_images_in_source_tree():
    forbidden = ("*.ckpt", "*.pth", "*.pt", "*.onnx")
    for pattern in forbidden:
        assert not list(ROOT.rglob(pattern)), f"ML binary found in source tree: {pattern}"
    # One historical notebook and the Markdown reports are intentional, not duplicates.
    assert (ROOT / "notebooks" / "ihsanvision_mvtec_padim.ipynb").is_file()
    assert (ROOT / "reports" / "2026-10-14-browser-bottle-inspection.md").is_file()
