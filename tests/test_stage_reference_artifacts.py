"""Staging either preserved Bottle model never retrains or overwrites artifacts."""
import json

import pytest

from scripts.stage_reference_artifacts import stage_reference


def prepare_source(folder, model="padim"):
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = b"previously verified checkpoint fixture"
    (folder / "model.ckpt").write_bytes(ckpt)
    split = {
        "train": ["/kaggle/input/mvtec/bottle/train/good/001.png"],
        "val": ["/kaggle/input/mvtec/bottle/test/broken/001.png"],
        "test": ["/kaggle/input/mvtec/bottle/test/good/002.png"],
    }
    (folder / "splits.json").write_text(json.dumps(split))
    (folder / "report.json").write_text(json.dumps({
        "model": model, "category": "bottle",
        "config": {"seed": 42},
        "split_counts": {"train": 1, "val": 1, "test": 1},
        "train_seconds": 29,
        "evaluation": {"metrics": {
            "image_AUROC": .99,
            "image_F1Score": .95,
            "pixel_AUROC": .96,
            "pixel_F1Score": .64,
        }},
    }))
    return ckpt


def test_stage_preserves_exact_artifacts_and_not_extra(tmp_path):
    source = tmp_path / "saved" / "padim-bottle-seed42"
    expected = prepare_source(source)
    (source / "optional-smoke-overlay.png").write_bytes(b"not needed")
    output = tmp_path / "out"
    staged = stage_reference(source, output)
    assert staged == output / "padim-bottle-seed42"
    assert (staged / "model.ckpt").read_bytes() == expected
    assert {f.name for f in staged.iterdir()} == {
        "model.ckpt", "splits.json", "report.json"
    }
    assert (source / "model.ckpt").read_bytes() == expected


def test_stage_fails_before_overwriting(tmp_path):
    source = tmp_path / "saved"
    prepare_source(source)
    output = tmp_path / "out"
    existing = output / "padim-bottle-seed42"
    existing.mkdir(parents=True)
    (existing / "note.txt").write_text("keep this")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage_reference(source, output)
    assert (existing / "note.txt").read_text() == "keep this"


def test_stage_rejects_invalid_manifest_without_copying(tmp_path):
    source = tmp_path / "saved"
    prepare_source(source)
    report = json.loads((source / "report.json").read_text())
    report["split_counts"]["test"] = 99
    (source / "report.json").write_text(json.dumps(report))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="split counts"):
        stage_reference(source, output)
    assert not (output / "padim-bottle-seed42").exists()


def test_stage_missing_checkpoint_does_not_create_run(tmp_path):
    source = tmp_path / "saved"
    prepare_source(source)
    (source / "model.ckpt").unlink()
    output = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="model.ckpt"):
        stage_reference(source, output)
    assert not (output / "padim-bottle-seed42").exists()


def test_stage_preserves_existing_patchcore_without_trainer_or_deserialization(tmp_path):
    source = tmp_path / "saved" / "patchcore-bottle-seed42"
    expected = prepare_source(source, model="patchcore")
    destination = stage_reference(
        source, tmp_path / "out", category="bottle", seed=42, model="patchcore"
    )
    assert destination.name == "patchcore-bottle-seed42"
    assert (destination / "model.ckpt").read_bytes() == expected
    assert {item.name for item in destination.iterdir()} == {
        "model.ckpt", "report.json", "splits.json"
    }


def test_patchcore_staging_refuses_wrong_model_and_existing_destination(tmp_path):
    source = tmp_path / "source"
    prepare_source(source, model="padim")
    output = tmp_path / "private-out"
    with pytest.raises(ValueError, match="Report does not match"):
        stage_reference(source, output, model="patchcore")
    assert not (output / "patchcore-bottle-seed42").exists()

    prepare_source(source, model="patchcore")
    destination = stage_reference(source, output, model="patchcore")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage_reference(source, output, model="patchcore")
    assert destination.is_dir()


def test_staging_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="Unsupported MVTec"):
        stage_reference(tmp_path, tmp_path / "private-out", model="unknown")


def test_staging_does_not_replace_dangling_destination_symlink(tmp_path):
    source = tmp_path / "saved"
    prepare_source(source, model="patchcore")
    destination_root = tmp_path / "out"
    destination_root.mkdir()
    link = destination_root / "patchcore-bottle-seed42"
    link.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage_reference(source, destination_root, model="patchcore")
    assert link.is_symlink()
