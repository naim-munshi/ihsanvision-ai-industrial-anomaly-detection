"""Build a portfolio comparison from actual saved report triples, never fake scores."""
import json

import pytest

from scripts.summarize_comparison import main, summarize_saved_runs


CATEGORIES = ("bottle", "cable", "carpet")


def populate(tmp_path):
    for category in CATEGORIES:
        for model, score in (("padim", 0.82), ("patchcore", 0.91)):
            folder = tmp_path / f"{model}-{category}-seed42"
            folder.mkdir()
            (folder / "model.ckpt").write_bytes(b"not-deserialized-test-placeholder")
            manifest = {
                "train": [f"/private/{category}/train/good/0.png"],
                "val": [f"/private/{category}/test/scratch/0.png"],
                "test": [f"/private/{category}/test/scratch/1.png"],
            }
            (folder / "splits.json").write_text(json.dumps(manifest))
            (folder / "report.json").write_text(json.dumps({
                "model": model, "category": category, "config": {"seed": 42},
                "split_counts": {"train": 1, "val": 1, "test": 1},
                "train_seconds": 125.25,
                "evaluation": {"metrics": {
                    "image_AUROC": score, "image_F1Score": 0.75,
                    "pixel_AUROC": 0.86, "pixel_F1Score": 0.65,
                }},
            }))


def test_three_category_portfolio_table_needs_no_comparison_json_or_training(tmp_path):
    populate(tmp_path)
    assert not (tmp_path / "comparison.json").exists()
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    summary = summarize_saved_runs(tmp_path, list(CATEGORIES), 42)
    assert summary.count("| bottle |") == 2
    assert summary.count("| cable |") == 2
    assert summary.count("| carpet |") == 2
    assert "| carpet | patchcore | 1 | 1 | 1 | 0.9100" in summary
    assert "not directly comparable" in summary
    assert "/private/" not in summary
    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_cli_exports_non_overwriting_verified_markdown(tmp_path):
    populate(tmp_path)
    out = tmp_path / "portfolio-3-category.md"
    main(["--runs-dir", str(tmp_path), "--from-saved-runs",
          "--categories", *CATEGORIES, "--seed", "42", "--output", str(out)])
    assert "Validated Saved-Run Portfolio Comparison" in out.read_text()
    with pytest.raises(FileExistsError):
        main(["--runs-dir", str(tmp_path), "--from-saved-runs",
              "--output", str(out)])


def test_missing_patchcore_checkpoint_refuses_to_fabricate_results(tmp_path):
    populate(tmp_path)
    (tmp_path / "patchcore-cable-seed42/model.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="model.ckpt"):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)


def test_mismatched_model_splits_refused(tmp_path):
    populate(tmp_path)
    target = tmp_path / "patchcore-carpet-seed42/splits.json"
    data = json.loads(target.read_text())
    data["test"] = ["/different/image.png"]
    target.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Unfair benchmark"):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)


def test_reject_nonfinite_or_missing_metrics(tmp_path):
    populate(tmp_path)
    report = tmp_path / "padim-bottle-seed42/report.json"
    data = json.loads(report.read_text())
    data["evaluation"]["metrics"]["image_AUROC"] = 1.2
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="image_AUROC"):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)


@pytest.mark.parametrize("categories,seed", [
    (["bottle", "bottle"], 42),
    ([], 42),
    (["bottle", "invalid"], 42),
    (["bottle"], -1),
])
def test_refuse_bad_portfolio_scope(tmp_path, categories, seed):
    with pytest.raises(ValueError):
        summarize_saved_runs(tmp_path, categories, seed)


@pytest.mark.parametrize("invalid,match", [
    (True, "image_AUROC"),
    (False, "image_AUROC"),
    ("0.99", "image_AUROC"),
    (None, "image_AUROC"),
])
def test_saved_report_rejects_boolean_or_non_numeric_auc(tmp_path, invalid, match):
    # JSON true would otherwise be accepted as AUROC=1.0000 by Python's
    # isinstance(True, int). Never publish fabricated perfection.
    populate(tmp_path)
    path = tmp_path / "padim-bottle-seed42/report.json"
    payload = json.loads(path.read_text())
    payload["evaluation"]["metrics"]["image_AUROC"] = invalid
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)


@pytest.mark.parametrize("evaluation", [None, [], {}, {"metrics": None}, {"metrics": []}])
def test_saved_report_rejects_malformed_evaluation_without_attribute_error(
    tmp_path, evaluation
):
    populate(tmp_path)
    path = tmp_path / "padim-bottle-seed42/report.json"
    payload = json.loads(path.read_text())
    payload["evaluation"] = evaluation
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="evaluation metrics"):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)


@pytest.mark.parametrize("invalid", [True, False, "125.25", None])
def test_saved_report_rejects_invalid_training_duration(tmp_path, invalid):
    populate(tmp_path)
    path = tmp_path / "padim-bottle-seed42/report.json"
    payload = json.loads(path.read_text())
    payload["train_seconds"] = invalid
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="train_seconds"):
        summarize_saved_runs(tmp_path, list(CATEGORIES), 42)
