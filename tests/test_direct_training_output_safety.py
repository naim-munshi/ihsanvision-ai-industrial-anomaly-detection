"""Safety regression for direct train and benchmark entrypoints."""
from pathlib import Path

import pytest

from ihsanvision.config import TrainConfig


def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "bottle" / "test").mkdir()
    return root


@pytest.mark.parametrize("relative", [".", "runs", "bottle/train/good/output"])
def test_direct_training_rejects_dataset_outputs(tmp_path: Path, relative: str) -> None:
    root = dataset(tmp_path)
    with pytest.raises(ValueError, match="outside the dataset root"):
        TrainConfig(dataset_root=root, output_dir=root / relative).validate()


def test_direct_training_rejects_symlink_alias(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the dataset root"):
        TrainConfig(dataset_root=root, output_dir=alias / "artifacts").validate()


def test_direct_training_accepts_external_output(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    TrainConfig(dataset_root=root, output_dir=tmp_path / "private-runs").validate()
