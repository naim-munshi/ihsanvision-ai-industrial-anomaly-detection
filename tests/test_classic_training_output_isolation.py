"""Offline tests for classic MVTec training source/output separation."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.resume_mvtec_training import execute, plan, require_external_runs


@pytest.mark.parametrize("relative", [".", "runs", "bottle/train/good/generated"])
def test_rejects_nested_outputs_before_scanning(tmp_path: Path, relative: str) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    with pytest.raises(ValueError, match="outside the dataset root"):
        plan(source, source / relative, ["bottle"], 42)


def test_rejects_symlink_alias(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the dataset root"):
        require_external_runs(source, alias / "runs")


def test_rejects_nested_outputs_before_trainer(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    trainer = Mock()
    with pytest.raises(ValueError, match="outside the dataset root"):
        execute({"schema_version": 1, "jobs": [{"category": "bottle", "model": "padim"}]},
                source, source / "outputs", seed=42, accelerator="cpu", trainer=trainer)
    trainer.assert_not_called()


def test_external_runs_are_allowed(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    assert require_external_runs(source, tmp_path / "runs") is None
