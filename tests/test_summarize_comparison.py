"""Offline tests for verification-first, checkpoint-free benchmark reporting."""
import json

import pytest

from scripts.summarize_comparison import main, summarize


def manifest(test="/mvtec/bottle/test/broken/2.png"):
    return {
        "train": ["/mvtec/bottle/train/good/1.png"],
        "val": ["/mvtec/bottle/test/broken/1.png"],
        "test": [test],
    }


def make_complete_benchmark(root):
    rows = []
    for model, score in (("padim", 0.93), ("patchcore", 0.95)):
        folder = root / f"{model}-bottle-seed42"
        folder.mkdir(parents=True)
        (folder / "model.ckpt").write_bytes(b"trusted placeholder; not loaded")
        (folder / "splits.json").write_text(json.dumps(manifest()))
        metrics = {
            "image_AUROC": score,
            "image_F1Score": 0.8,
            "pixel_AUROC": 0.9,
            "pixel_F1Score": 0.6,
        }
        report = {
            "model": model, "category": "bottle", "config": {"seed": 42},
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "train_seconds": 12.3, "evaluation": {"metrics": metrics},
        }
        (folder / "report.json").write_text(json.dumps(report))
        rows.append({"category": "bottle", "model": model, "train_seconds": 12.3, **metrics})
    comparison = {
        "rows": rows, "seed": 42,
        "split_counts": {"bottle": {"train": 1, "val": 1, "test": 1}},
        "split_equivalence": "verified_path_by_path",
        "model_sources": {"bottle": {"padim": "reused", "patchcore": "trained"}},
    }
    (root / "comparison.json").write_text(json.dumps(comparison))
    return comparison


def test_summary_verifies_two_model_reports_without_loading_checkpoints(tmp_path, capsys):
    make_complete_benchmark(tmp_path)
    summary = summarize(tmp_path)
    assert "| bottle | padim | 0.9300" in summary
    assert "| bottle | patchcore | 0.9500" in summary
    assert "not** directly comparable" in summary
    assert "different sessions" in summary
    main(["--runs-dir", str(tmp_path)])
    assert capsys.readouterr().out == summary


def test_rejects_changed_result_row(tmp_path):
    obj = make_complete_benchmark(tmp_path)
    obj["rows"][0]["image_AUROC"] = 0.99
    (tmp_path / "comparison.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="differs from saved report"):
        summarize(tmp_path)


def test_rejects_mismatched_splits(tmp_path):
    make_complete_benchmark(tmp_path)
    target = tmp_path / "patchcore-bottle-seed42/splits.json"
    target.write_text(json.dumps(manifest("/other.png")))
    with pytest.raises(RuntimeError, match="Unfair benchmark"):
        summarize(tmp_path)


def test_refuses_incomplete_checkpoint_without_writing(tmp_path):
    make_complete_benchmark(tmp_path)
    (tmp_path / "patchcore-bottle-seed42/model.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="model.ckpt"):
        main(["--runs-dir", str(tmp_path), "--output", str(tmp_path / "summary.md")])
    assert not (tmp_path / "summary.md").exists()


def test_refuses_unverified_split_claim(tmp_path):
    obj = make_complete_benchmark(tmp_path)
    obj["split_equivalence"] = "not_verified"
    (tmp_path / "comparison.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="split equivalence"):
        summarize(tmp_path)


def test_refuses_overwriting_existing_summary(tmp_path):
    make_complete_benchmark(tmp_path)
    output = tmp_path / "summary.md"
    output.write_text("keep original")
    with pytest.raises(FileExistsError):
        main(["--runs-dir", str(tmp_path), "--output", str(output)])
    assert output.read_text() == "keep original"


def test_refuses_partial_rows(tmp_path):
    obj = make_complete_benchmark(tmp_path)
    obj["rows"].pop()
    (tmp_path / "comparison.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Both PaDiM and PatchCore"):
        summarize(tmp_path)
