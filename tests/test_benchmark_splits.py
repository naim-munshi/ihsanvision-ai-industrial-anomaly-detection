"""Offline comparison contract: identical model-specific held-out data is required."""
import json

import pytest

from src import benchmark


def sample_manifest():
    return {
        "train": ["/dataset/bottle/train/good/001.png", "/dataset/bottle/train/good/002.png"],
        "val": ["/dataset/bottle/test/defect/001.png"],
        "test": ["/dataset/bottle/test/defect/002.png"],
    }


def test_matching_train_val_test_manifests():
    first = sample_manifest()
    second = {name: list(reversed(items)) for name, items in first.items()}
    first = {name: sorted(items) for name, items in first.items()}
    second = {name: sorted(items) for name, items in second.items()}
    assert benchmark.require_matching_splits(
        "bottle", {"padim": first, "patchcore": second}
    ) == {"train": 2, "val": 1, "test": 1}


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_mismatched_split_is_rejected(split):
    a, b = sample_manifest(), sample_manifest()
    b[split] = ["/different.png"]
    with pytest.raises(RuntimeError, match=split):
        benchmark.require_matching_splits("bottle", {"padim": a, "patchcore": b})


def test_read_manifest_rejects_overlapping_paths(tmp_path):
    file = tmp_path / "splits.json"
    values = sample_manifest()
    values["val"] = list(values["test"])
    file.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="leakage"):
        benchmark.read_split_manifest(file)


def test_read_manifest_normalizes_order(tmp_path):
    file = tmp_path / "splits.json"
    values = sample_manifest()
    values["train"] = list(reversed(values["train"]))
    file.write_text(json.dumps(values))
    assert benchmark.read_split_manifest(file)["train"] == sorted(values["train"])


def test_benchmark_writes_comparison_only_for_equal_splits(monkeypatch, tmp_path):
    data = tmp_path / "dataset"
    output = tmp_path / "output"
    data.mkdir()
    visited = []

    def fake_training(config):
        visited.append((config.model, config.category))
        folder = output / f"{config.model}-{config.category}-seed{config.seed}"
        folder.mkdir(parents=True)
        (folder / "splits.json").write_text(json.dumps(sample_manifest()))
        return {
            "train_seconds": 5.0 if config.model == "padim" else 9.0,
            "evaluation": {"metrics": {"image_AUROC": 0.9}},
        }

    monkeypatch.setattr(benchmark, "run", fake_training)
    benchmark.main(["--dataset-root", str(data), "--output-dir", str(output)])
    result = json.loads((output / "comparison.json").read_text())
    assert visited == [("padim", "bottle"), ("patchcore", "bottle")]
    assert result["split_equivalence"] == "verified_path_by_path"
    assert result["split_counts"]["bottle"] == {"train": 2, "val": 1, "test": 1}
    assert len(result["rows"]) == 2
    assert (output / "comparison.csv").exists()


def test_comparison_is_not_published_when_splits_differ(monkeypatch, tmp_path):
    output = tmp_path / "output"

    def fake_training(config):
        folder = output / f"{config.model}-{config.category}-seed{config.seed}"
        folder.mkdir(parents=True)
        paths = sample_manifest()
        if config.model == "patchcore":
            paths["test"] = ["/different_final_test_sample.png"]
        (folder / "splits.json").write_text(json.dumps(paths))
        return {"train_seconds": 5.0, "evaluation": {"metrics": {"image_AUROC": 0.9}}}

    monkeypatch.setattr(benchmark, "run", fake_training)
    with pytest.raises(RuntimeError, match="Unfair benchmark"):
        benchmark.main(["--dataset-root", str(tmp_path), "--output-dir", str(output)])
    assert not (output / "comparison.json").exists()
    assert not (output / "comparison.csv").exists()
