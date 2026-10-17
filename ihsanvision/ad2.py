"""Non-downloading MVTec AD 2 preflight and split checks.

MVTec AD 2 is distinct from classic MVTec AD:
  train/good, validation/good (both normal), test_public/good,
  test_public/bad, test_public/ground_truth/bad/<stem>_mask.png.
Private test folders have no public labels and MUST NOT be scored locally.

Dataset images are neither opened nor uploaded. Only directory structure,
nonempty files and matching public anomaly-mask filenames are inspected.
"""
from __future__ import annotations

from pathlib import Path
from .registry import AD2_CATEGORIES

_EXTENSIONS = {".png"}  # MVTec AD 2's Anomalib loader accepts PNG


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Required private dataset directory is missing/unsafe: {directory.name}")
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in _EXTENSIONS)
    if not files or any(p.is_symlink() or p.stat().st_size <= 0 for p in files):
        raise ValueError(f"Required image directory empty or invalid: {directory.name}")
    return files


def preflight_ad2(dataset_root: Path, category: str) -> dict:
    """Validate only owned local, per-category files; never auto-download."""
    if category not in AD2_CATEGORIES:
        raise ValueError("Unsupported MVTec AD 2 category")
    root = Path(dataset_root).expanduser()
    if not root.is_dir() or root.is_symlink() or (root / category).is_symlink():
        raise ValueError("MVTec AD 2 dataset root/category is missing or unsafe")
    folder = root / category
    if not folder.is_dir():
        raise ValueError("Selected MVTec AD 2 category is not installed locally")

    train = _images(folder / "train" / "good")
    validation = _images(folder / "validation" / "good")
    public_good = _images(folder / "test_public" / "good")
    public_bad = _images(folder / "test_public" / "bad")
    masks = _images(folder / "test_public" / "ground_truth" / "bad")
    mask_names = {file.name for file in masks}
    expected_masks = {
        image.stem + "_mask" + image.suffix for image in public_bad
    }
    if mask_names != expected_masks or len(mask_names) != len(masks):
        raise ValueError(
            "Public anomaly image/mask correspondence invalid: "
            "each bad PNG needs exactly one <stem>_mask.png"
        )
    splits = {"train": train, "val": validation, "test": public_good + public_bad}
    resolved = [{str(p.resolve()) for p in files} for files in splits.values()]
    if (resolved[0] & resolved[1] or resolved[0] & resolved[2]
            or resolved[1] & resolved[2]):
        raise ValueError("AD2 train/validation/public test split images overlap")

    return {
        "dataset": "mvtec_ad_2",
        "category": category,
        "train_normal": len(train),
        "validation_normal": len(validation),
        "public_test_normal": len(public_good),
        "public_test_anomalous": len(public_bad),
        "public_test_masks": len(masks),
        "private_test_evaluated": False,
        "test_protocol": "public_only",
    }


def verify_ad2_splits(manifest: dict[str, list[str]], dataset_root: Path,
                      category: str) -> None:
    """Require Anomalib split paths to stay within original, labeled folders."""
    root = (Path(dataset_root).expanduser() / category).resolve()
    for split, allowed in (
        ("train", root / "train" / "good"),
        ("val", root / "validation" / "good"),
    ):
        images = manifest.get(split, [])
        if not images or any(Path(image).resolve().parent != allowed for image in images):
            raise ValueError(f"AD2 {split} must use its dedicated normal-only split")
    test = manifest.get("test", [])
    if not test or any(
        Path(image).resolve().parent not in {
            root / "test_public" / "good", root / "test_public" / "bad"
        }
        for image in test
    ):
        raise ValueError("AD2 local evaluation must use test_public exclusively")
