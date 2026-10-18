"""Direct training must not overwrite any completed OR interrupted runs.

These tests use only temporary fixture directories, not Anomalib or model data.
"""
from pathlib import Path

import pytest

from ihsanvision.config import TrainConfig
from ihsanvision.train import prepare_fresh_run_dir


def config(tmp_path: Path) -> TrainConfig:
    root = tmp_path / "dataset"
    (root / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "bottle" / "test").mkdir()
    return TrainConfig(dataset_root=root, output_dir=tmp_path / "runs",
                       category="bottle", model="padim", seed=42)


def test_new_run_directory_created_once(tmp_path: Path):
    cfg = config(tmp_path)
    folder = prepare_fresh_run_dir(cfg)
    assert folder == cfg.output_dir / "padim-bottle-seed42"
    assert folder.is_dir()
    with pytest.raises(FileExistsError):
        prepare_fresh_run_dir(cfg)


@pytest.mark.parametrize("partial_name", ["model.ckpt", "report.json", "splits.json", "engine"])
def test_direct_train_preserves_partial_artifacts(tmp_path: Path, partial_name: str):
    cfg = config(tmp_path)
    folder = cfg.output_dir / "padim-bottle-seed42"
    folder.mkdir(parents=True)
    path = folder / partial_name
    if partial_name == "engine":
        path.mkdir()
    else:
        path.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        prepare_fresh_run_dir(cfg)
    assert path.is_dir() if partial_name == "engine" else path.read_bytes() == b"do-not-overwrite"


def test_symlinked_run_path_is_never_followed(tmp_path: Path):
    cfg = config(tmp_path)
    target = tmp_path / "other"
    target.mkdir()
    (target / "model.ckpt").write_bytes(b"keep")
    cfg.output_dir.mkdir()
    (cfg.output_dir / "padim-bottle-seed42").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare_fresh_run_dir(cfg)
    assert (target / "model.ckpt").read_bytes() == b"keep"


def test_dataset_nested_output_remains_forbidden(tmp_path: Path):
    cfg = config(tmp_path)
    nested = TrainConfig(dataset_root=cfg.dataset_root,
                         output_dir=cfg.dataset_root / "runs")
    with pytest.raises(ValueError, match="outside the dataset root"):
        prepare_fresh_run_dir(nested)
    assert not (cfg.dataset_root / "runs").exists()
