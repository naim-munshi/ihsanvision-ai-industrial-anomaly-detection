from pathlib import Path
import pytest

from scripts.kaggle_preflight import find_dataset_roots, scan_dataset


def make_dataset(root: Path) -> Path:
    for relative in (
        "bottle/train/good/001.png",
        "bottle/test/good/002.png",
        "bottle/test/broken_large/003.png",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake png")
    return root


def test_scan_counts(tmp_path):
    root = make_dataset(tmp_path / "MVTecAD")
    info = scan_dataset(root)
    assert info["train_good_images"] == 1
    assert info["test_good_images"] == 1
    assert info["test_anomaly_images"] == 1
    assert info["ground_truth_directory"] is False


def test_nested_kaggle_directory_discovery(tmp_path):
    root = make_dataset(tmp_path / "datasets" / "owner" / "mvtec-ad")
    assert find_dataset_roots(tmp_path) == [root.resolve()]


def test_input_dir_itself_is_dataset(tmp_path):
    root = make_dataset(tmp_path / "mvtec-ad")
    assert find_dataset_roots(root) == [root.resolve()]


def test_reject_incomplete_dataset(tmp_path):
    incomplete = tmp_path / "incomplete"
    (incomplete / "bottle" / "train" / "good").mkdir(parents=True)
    (incomplete / "bottle" / "test").mkdir(parents=True)
    with pytest.raises(ValueError, match="required"):
        scan_dataset(incomplete)
