"""Offline safety regression: never contaminate AD2 source data with run artifacts."""
from pathlib import Path
from unittest.mock import patch

import pytest

from ihsanvision.train_ad2 import AD2TrainConfig


def config(source: Path, output: Path) -> AD2TrainConfig:
    return AD2TrainConfig(dataset_root=source, output_dir=output, category="can", model="padim")


@pytest.mark.parametrize("relative_output", [".", "outputs", "train/good/results"])
def test_refuses_output_inside_dataset(tmp_path: Path, relative_output: str) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    with patch("ihsanvision.train_ad2.preflight_ad2") as preflight:
        with pytest.raises(ValueError, match="outside the dataset root"):
            config(source, source / relative_output).validate()
        preflight.assert_not_called()


def test_refuses_symlink_alias_into_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with patch("ihsanvision.train_ad2.preflight_ad2") as preflight:
        with pytest.raises(ValueError, match="outside the dataset root"):
            config(source, alias / "outputs").validate()
        preflight.assert_not_called()


def test_allows_external_output_and_runs_preflight(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    with patch("ihsanvision.train_ad2.preflight_ad2", return_value={"train_normal": 2}) as preflight:
        assert config(source, tmp_path / "artifacts").validate() == {"train_normal": 2}
        preflight.assert_called_once_with(source, "can")
