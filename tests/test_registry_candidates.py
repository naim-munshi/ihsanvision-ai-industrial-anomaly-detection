"""Offline adversarial tests for unapproved candidate checkpoint registry."""
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_registry_candidates import build_candidates, fingerprint, main


def test_fingerprint_is_actual_file_content(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"first")
    first = fingerprint(checkpoint)
    checkpoint.write_bytes(b"second")
    assert first != fingerprint(checkpoint)


def test_rejects_duplicate_and_unknown_categories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        build_candidates(tmp_path, ["bottle", "bottle"], 42)
    with pytest.raises(ValueError, match="Unsupported"):
        build_candidates(tmp_path, ["not-a-category"], 42)
    with pytest.raises(ValueError, match="seed"):
        build_candidates(tmp_path, ["bottle"], -1)


def test_rejects_symlinked_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    folder = root / "padim-bottle-seed42"
    folder.mkdir(parents=True)
    actual = tmp_path / "outside.ckpt"
    actual.write_bytes(b"trusted")
    (folder / "model.ckpt").symlink_to(actual)
    with patch("scripts.build_registry_candidates.read_completed_run",
               return_value=({"evaluation": {"protocol": "test"}}, {})):
        with pytest.raises(ValueError, match="non-symlink"):
            build_candidates(root, ["bottle"], 42)


def test_missing_run_does_not_generate_candidates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_candidates(tmp_path, ["bottle"], 42)


def test_cli_rejects_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "candidates.json"
    output.write_text("preserve")
    with pytest.raises(SystemExit, match="refusing overwrite"):
        main(["--runs-dir", str(tmp_path / "runs"), "--categories", "bottle",
              "--output", str(output)])
    assert output.read_text() == "preserve"


def test_cli_refuses_output_inside_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(SystemExit, match="outside experiment runs"):
        main(["--runs-dir", str(runs), "--categories", "bottle",
              "--output", str(runs / "candidate.json")])
