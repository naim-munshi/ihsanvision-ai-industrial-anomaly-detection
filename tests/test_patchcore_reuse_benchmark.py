"""Reuse-comparison contract: preserved PaDiM is mandatory and never retrained."""
import json
from pathlib import Path

import pytest

from src import benchmark
from src.config import verify_reference_splits


def splits():
    return {
        "train": ["/kaggle/input/dataset/bottle/train/good/001.png"],
        "val": ["/kaggle/input/dataset/bottle/test/broken/001.png"],
        "test": ["/kaggle/input/dataset/bottle/test/good/003.png"],
    }


def write_run(folder: Path, model: str, seed: int = 42, category: str = "bottle"):
    folder.mkdir(parents=True)
    (folder / "splits.json").write_text(json.dumps(splits()))
    (folder / "model.ckpt").write_bytes(b"trusted fixture only, never deserialized")
    report = {
        "model": model, "category": category, "config": {"seed": seed},
        "split_counts": {"train": 1, "val": 1, "test": 1},
        "train_seconds": 29.5,
        "evaluation": {"metrics": {
            "image_AUROC": 0.91, "image_F1Score": 0.82,
            "pixel_AUROC": 0.93, "pixel_F1Score": 0.64,
        }},
    }
    (folder / "report.json").write_text(json.dumps(report))
    return report


def test_existing_padim_required_before_training(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(benchmark, "run", lambda conf: called.append(conf))
    with pytest.raises(FileNotFoundError, match="no automatic PaDiM retraining"):
        benchmark.main(["--dataset-root", str(tmp_path),
                        "--output-dir", str(tmp_path), "--reuse-existing"])
    assert called == []


def test_reuses_existing_padim_and_trains_patchcore_only(tmp_path, monkeypatch):
    output = tmp_path / "runs"
    write_run(output / "padim-bottle-seed42", "padim")
    called = []

    def fake_train(conf):
        called.append(conf)
        assert conf.model == "patchcore"
        assert conf.reference_splits == output / "padim-bottle-seed42/splits.json"
        conf.validate() if False else None  # Real datasets are not present in CI.
        folder = output / "patchcore-bottle-seed42"
        folder.mkdir()
        (folder / "splits.json").write_text(json.dumps(splits()))
        return {"train_seconds": 55.0, "evaluation": {"metrics": {"image_AUROC": 0.92}}}

    monkeypatch.setattr(benchmark, "run", fake_train)
    benchmark.main(["--dataset-root", str(tmp_path), "--output-dir", str(output),
                    "--reuse-existing", "--accelerator", "cpu"])
    assert len(called) == 1
    saved = json.loads((output / "comparison.json").read_text())
    assert saved["model_sources"]["bottle"] == {"padim": "reused", "patchcore": "trained"}
    assert saved["split_equivalence"] == "verified_path_by_path"


def test_reuses_both_when_two_completed_runs_present(tmp_path, monkeypatch):
    output = tmp_path / "runs"
    write_run(output / "padim-bottle-seed42", "padim")
    write_run(output / "patchcore-bottle-seed42", "patchcore")
    monkeypatch.setattr(benchmark, "run", lambda conf: pytest.fail("Should not retrain"))
    benchmark.main(["--dataset-root", str(tmp_path), "--output-dir", str(output),
                    "--reuse-existing"])
    report = json.loads((output / "comparison.json").read_text())
    assert report["model_sources"]["bottle"] == {"padim": "reused", "patchcore": "reused"}


def test_invalid_saved_report_rejected(tmp_path):
    folder = tmp_path / "padim-bottle-seed42"
    write_run(folder, "padim")
    report = json.loads((folder / "report.json").read_text())
    report["evaluation"]["metrics"]["pixel_F1Score"] = float("nan")
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="pixel_F1Score"):
        benchmark.read_completed_run(folder, "padim", "bottle", 42)


def test_reuse_refuses_partial_existing_patchcore_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "runs"
    write_run(output / "padim-bottle-seed42", "padim")
    folder = output / "patchcore-bottle-seed42"
    folder.mkdir()
    (folder / "splits.json").write_text(json.dumps(splits()))
    monkeypatch.setattr(benchmark, "run", lambda conf: pytest.fail("Should not retrain partial"))
    with pytest.raises(FileNotFoundError, match="Cannot reuse patchcore"):
        benchmark.main(["--dataset-root", str(tmp_path), "--output-dir", str(output),
                        "--reuse-existing"])
    assert not (output / "comparison.csv").exists()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_reference_split_mismatch_rejected_before_training(tmp_path, split):
    reference = tmp_path / "splits.json"
    data = splits()
    reference.write_text(json.dumps(data))
    changed = splits()
    changed[split] = ["/wrong-folder/other-image.png"]
    with pytest.raises(RuntimeError, match=f"Reference {split} split differs"):
        verify_reference_splits(changed, reference)


def test_reference_split_success(tmp_path):
    reference = tmp_path / "splits.json"
    reference.write_text(json.dumps(splits()))
    assert verify_reference_splits(splits(), reference) == {
        "train": 1, "val": 1, "test": 1,
    }


def test_reference_split_leakage_rejected(tmp_path):
    reference = tmp_path / "splits.json"
    contaminated = splits()
    contaminated["val"] = contaminated["test"]
    reference.write_text(json.dumps(contaminated))
    with pytest.raises(ValueError, match="Overlapping"):
        verify_reference_splits(splits(), reference)
