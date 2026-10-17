"""Offline tests for official AD2 layout, private-set isolation and training safety."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ihsanvision.ad2 import preflight_ad2, verify_ad2_splits
from ihsanvision.registry import AD2_CATEGORIES
from ihsanvision import train_ad2


def dataset(tmp_path, category="sheet_metal"):
    root = tmp_path / "AD2"
    paths = (
        "train/good/001.png",
        "validation/good/002.png",
        "test_public/good/003.png",
        "test_public/bad/004.png",
        "test_public/ground_truth/bad/004_mask.png",
    )
    for name in paths:
        path = root / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dataset-fixture-nonzero")
    return root


def manifest(root, category="sheet_metal"):
    category_root = root / category
    return {
        "train": [str(category_root / "train/good/001.png")],
        "val": [str(category_root / "validation/good/002.png")],
        "test": [
            str(category_root / "test_public/good/003.png"),
            str(category_root / "test_public/bad/004.png"),
        ],
    }


def test_official_eight_category_names_including_wallplugs():
    assert AD2_CATEGORIES == (
        "can", "fabric", "fruit_jelly", "rice",
        "sheet_metal", "vial", "wallplugs", "walnuts",
    )
    assert "wall_plugs" not in AD2_CATEGORIES


def test_public_only_preflight_counts_and_no_private_paths(tmp_path):
    root = dataset(tmp_path)
    private = root / "sheet_metal" / "test_private" / "unknown.png"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(b"private-test-no-ground-truth")
    mixed = root / "sheet_metal" / "test_private_mixed" / "unknown.png"
    mixed.parent.mkdir()
    mixed.write_bytes(b"unknown")
    result = preflight_ad2(root, "sheet_metal")
    assert result == {
        "dataset": "mvtec_ad_2",
        "category": "sheet_metal",
        "train_normal": 1,
        "validation_normal": 1,
        "public_test_normal": 1,
        "public_test_anomalous": 1,
        "public_test_masks": 1,
        "private_test_evaluated": False,
        "test_protocol": "public_only",
    }
    assert "private" not in json.dumps(result) or result["private_test_evaluated"] is False
    assert str(tmp_path) not in json.dumps(result)
    verify_ad2_splits(manifest(root), root, "sheet_metal")


@pytest.mark.parametrize("folder", [
    "train/good/001.png",
    "validation/good/002.png",
    "test_public/good/003.png",
    "test_public/bad/004.png",
    "test_public/ground_truth/bad/004_mask.png",
])
def test_missing_required_image_refuses_dataset(tmp_path, folder):
    root = dataset(tmp_path)
    (root / "sheet_metal" / folder).unlink()
    with pytest.raises(ValueError):
        preflight_ad2(root, "sheet_metal")


def test_mismatched_extra_or_empty_masks_refused(tmp_path):
    root = dataset(tmp_path)
    bad_mask = root / "sheet_metal/test_public/ground_truth/bad/004_mask.png"
    bad_mask.rename(bad_mask.with_name("incorrect_mask.png"))
    with pytest.raises(ValueError, match="correspondence"):
        preflight_ad2(root, "sheet_metal")
    bad_mask.with_name("incorrect_mask.png").unlink()
    bad_mask.write_bytes(b"")
    with pytest.raises(ValueError):
        preflight_ad2(root, "sheet_metal")


def test_unknown_category_or_unsafe_file_refused(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        preflight_ad2(root, "wall_plugs")
    original = root / "sheet_metal" / "train/good/001.png"
    original.unlink()
    original.symlink_to(root / "sheet_metal" / "validation/good/002.png")
    with pytest.raises(ValueError):
        preflight_ad2(root, "sheet_metal")


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_private_or_wrong_split_never_passes_training_manifest(tmp_path, split):
    root = dataset(tmp_path)
    result = manifest(root)
    result[split] = [str(root / "sheet_metal/test_private/secret.png")]
    with pytest.raises(ValueError):
        verify_ad2_splits(result, root, "sheet_metal")


def test_dry_run_never_trains_or_creates_run_directory(tmp_path, monkeypatch, capsys):
    root = dataset(tmp_path)
    output = tmp_path / "runs"
    monkeypatch.setattr(train_ad2, "run", lambda *_: pytest.fail("dry-run started training"))
    train_ad2.main([
        "--dataset-root", str(root), "--category", "sheet_metal",
        "--output-dir", str(output), "--model", "patchcore",
    ])
    message = capsys.readouterr().out
    assert "DRY RUN" in message
    assert "public_good=1" in message
    assert not output.exists()


def test_execution_requires_explicit_noncommercial_acknowledgement(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    output = tmp_path / "runs"
    monkeypatch.setattr(train_ad2, "run", lambda *_: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="acknowledge-noncommercial"):
        train_ad2.main([
            "--dataset-root", str(root), "--category", "sheet_metal",
            "--output-dir", str(output), "--execute",
        ])
    assert not output.exists()


def test_existing_run_never_overwritten_before_loading_ml(tmp_path):
    root = dataset(tmp_path)
    output = tmp_path / "runs"
    cfg = train_ad2.AD2TrainConfig(root, output, "sheet_metal", "padim")
    folder = cfg.run_folder()
    folder.mkdir(parents=True)
    previous = folder / "model.ckpt"
    previous.write_bytes(b"previous-private-weights")
    with pytest.raises(FileExistsError):
        train_ad2.run(cfg)
    assert previous.read_bytes() == b"previous-private-weights"


def test_runtime_adapter_forces_public_eval_and_dedicated_validation(
    tmp_path, monkeypatch
):
    root = dataset(tmp_path)
    output = tmp_path / "runs"
    called = {}

    class Series:
        def __init__(self, values):
            self.values = values
        def tolist(self):
            return self.values

    class Samples:
        columns = ["image_path"]
        def __init__(self, values):
            self.values = values
        def __getitem__(self, name):
            assert name == "image_path"
            return Series(self.values)

    class FakeAD2:
        def __init__(self, **kwargs):
            called["dataset_kwargs"] = kwargs
            self.train_data = SimpleNamespace(samples=Samples(manifest(root)["train"]))
            self.val_data = SimpleNamespace(samples=Samples(manifest(root)["val"]))
            self.test_data = SimpleNamespace(samples=Samples(manifest(root)["test"]))
        def setup(self):
            called["setup"] = True

    class FakeEngine:
        def __init__(self, **kwargs):
            called["engine_kwargs"] = kwargs
            self.trainer = SimpleNamespace(save_checkpoint=self.save)
        def fit(self, model, datamodule):
            called["fit"] = True
            assert isinstance(datamodule, FakeAD2)
        def save(self, path):
            Path(path).write_bytes(b"private-fixture-weights")
        def test(self, *, model, datamodule, ckpt_path):
            called["test_ckpt"] = ckpt_path
            assert datamodule.test_data.samples["image_path"].tolist() == manifest(root)["test"]
            return [{"image_AUROC": 0.8}]

    anomalib = ModuleType("anomalib")
    anomalib.__version__ = "2.5.0"
    anomalib.data = ModuleType("anomalib.data")
    anomalib.data.MVTecAD2 = FakeAD2
    anomalib.engine = ModuleType("anomalib.engine")
    anomalib.engine.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "anomalib", anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.data", anomalib.data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", anomalib.engine)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake-torch"))
    monkeypatch.setattr(train_ad2, "make_model", lambda model: object())
    monkeypatch.setattr(train_ad2, "seed_everything", lambda seed: None)

    cfg = train_ad2.AD2TrainConfig(root, output, "sheet_metal", "padim")
    report = train_ad2.run(cfg)
    assert called["dataset_kwargs"]["test_type"] == "public"
    assert called["setup"] and called["fit"]
    assert called["test_ckpt"].endswith("/model.ckpt")
    assert report["evaluation"]["test_type"] == "public"
    assert report["evaluation"]["metrics"]["image_AUROC"] == 0.8
    saved_manifest = json.loads((cfg.run_folder() / "splits.json").read_text())
    verify_ad2_splits(saved_manifest, root, "sheet_metal")
    assert (cfg.run_folder() / "report.json").is_file()
