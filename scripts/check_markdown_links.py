"""Check all tracked-style Markdown prose links, not only handpicked README files.

GitHub-facing offline check: no web requests, model data, or dependencies.
Skips fenced code snippets and externally hosted URLs; validates local paths
and file existence across README, docs, reports and contribution guides.
"""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")
MAX_DOCUMENT_SIZE = 512 * 1024


def local_target(root: Path, source: Path, raw_destination: str) -> tuple[Path | None, str | None]:
    """Return (local target, error) while avoiding URLs and in-page anchors."""
    text = raw_destination.strip().split(None, 1)[0].strip("<>")
    try:
        target = urlsplit(text)
    except ValueError:
        return None, "invalid link syntax"
    if target.scheme or target.netloc or not target.path:
        return None, None
    if "\\" in target.path:
        return None, "backslash in local path"
    try:
        path = unquote(target.path)
        relative = Path(path)
        if relative.is_absolute():
            return None, "absolute filesystem path used as Markdown link"
        resolved = (source.parent / relative).resolve()
        if not resolved.is_relative_to(root.resolve()):
            return None, "relative link escapes source repository"
        if not resolved.exists():
            return None, "local link target does not exist"
        if not resolved.is_file() and not resolved.is_dir():
            return None, "local link target is not a regular file or directory"
        return resolved, None
    except (OSError, RuntimeError, ValueError):
        return None, "could not resolve local link"


def audit_markdown(root: Path) -> list[tuple[str, int, str, str]]:
    """Deterministic (document, line, destination, problem) broken link list."""
    root = root.resolve()
    problems: list[tuple[str, int, str, str]] = []
    for document in sorted(root.rglob("*.md")):
        relative = document.relative_to(root).as_posix()
        if any(part.startswith(".") and part not in (".github",) for part in document.relative_to(root).parts):
            continue
        if not document.is_file() or document.stat().st_size > MAX_DOCUMENT_SIZE:
            problems.append((relative, 0, "(document)", "missing or oversize Markdown file"))
            continue
        active_fence: str | None = None
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            fence = FENCE.match(line)
            if fence:
                marker = fence.group(1)
                if active_fence is None:
                    active_fence = marker
                elif marker[0] == active_fence[0] and len(marker) >= len(active_fence):
                    active_fence = None
                continue
            if active_fence is not None:
                continue
            prose = INLINE_CODE.sub("", line)
            destinations = [m.group(1) for m in INLINE_LINK.finditer(prose)]
            ref = REFERENCE.match(prose)
            if ref:
                destinations.append(ref.group(1))
            for link in destinations:
                _, problem = local_target(root, document, link)
                if problem:
                    problems.append((relative, line_number, link, problem))
    return problems


def main() -> None:
    problems = audit_markdown(ROOT)
    for document, line, link, problem in problems:
        print(f"{document}:{line}: {problem}: {link}")
    if problems:
        raise SystemExit(f"{len(problems)} broken/unsafe local Markdown link(s)")
    print("All repository Markdown local file/directory links resolve.")


if __name__ == "__main__":
    main()
