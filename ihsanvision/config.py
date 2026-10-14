"""Configuration, model factory, and test split auditing."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import random

CATEGORIES = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
MODELS = ("padim", "patchcore")

@dataclass(frozen=True)
class TrainConfig:
    dataset_root: Path
    category: str = "bottle"
    model: str = "padim"
    output_dir: Path = Path("runs")
    seed: int = 42
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    validation_ratio: float = 0.2
    accelerator: str = "auto"
    reference_splits: Path | None = None

    def validate(self) -> None:
        if self.category not in CATEGORIES or self.model not in MODELS:
            raise ValueError("Unsupported MVTec category or model")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("Validation ratio must be between zero and one")
        if self.train_batch_size < 1 or self.eval_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid data loader settings")
        if self.reference_splits is not None and not self.reference_splits.is_file():
            raise FileNotFoundError(f"Reference splits missing: {self.reference_splits}")
        if not (self.dataset_root / self.category / "train" / "good").is_dir():
            raise FileNotFoundError("Dataset missing <root>/<category>/train/good")
        if not (self.dataset_root / self.category / "test").is_dir():
            raise FileNotFoundError("Dataset missing <root>/<category>/test")

    def serializable(self) -> dict:
        values = asdict(self)
        values["dataset_root"] = str(self.dataset_root.resolve())
        values["output_dir"] = str(self.output_dir)
        values["reference_splits"] = str(self.reference_splits.resolve()) if self.reference_splits is not None else None
        return values

def make_model(name: str, *, pre_trained: bool = True):
    """Construct a training model, or checkpoint-restored inference model.

    Training requires ImageNet backbone initialization (pre_trained=True).
    Inference restores backbone, thresholds and normal statistics from the
    trusted Lightning checkpoint, so requesting pretrained weights again is
    unnecessary and can fail on networks that inspect HTTPS traffic.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    from anomalib.models import Padim, Patchcore
    return {"padim": Padim, "patchcore": Patchcore}[name](pre_trained=pre_trained)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def audit_splits(datamodule) -> dict[str, list[str]]:
    """Fail closed when files overlap or when splits cannot be inspected."""
    manifest = {}
    for name in ("train", "val", "test"):
        dataset = getattr(datamodule, f"{name}_data", None)
        samples = getattr(dataset, "samples", None)
        if samples is None or "image_path" not in samples.columns:
            raise RuntimeError(f"Cannot inspect {name} samples")
        paths = [str(Path(p).resolve()) for p in samples["image_path"].tolist()]
        if not paths or len(set(paths)) != len(paths):
            raise RuntimeError(f"Empty or duplicate {name} sample paths")
        manifest[name] = sorted(paths)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(manifest[a]) & set(manifest[b]):
            raise RuntimeError(f"Data leakage: overlapping {a}/{b} files")
    return manifest

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

def scalar_metrics(test_results) -> dict:
    metrics = {}
    for result in test_results or []:
        for key, val in result.items():
            try:
                if hasattr(val, "numel") and val.numel() != 1:
                    continue
                number = float(val)
                if math.isfinite(number):
                    metrics[str(key)] = number
            except (TypeError, ValueError):
                pass
    return metrics


def verify_reference_splits(manifest: dict[str, list[str]], path: Path) -> dict[str, int]:
    """Refuse training when a new run would use different evaluation images.

    This check runs *after datamodule setup* but *before fitting* so a mismatched
    Kaggle mount or seed does not waste compute or produce invalid comparison.
    """
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError("Reference split manifest must be an object")
    keys = ("train", "val", "test")
    for split in keys:
        paths = reference.get(split)
        if (
            not isinstance(paths, list) or not paths
            or not all(isinstance(p, str) and p for p in paths)
            or len(paths) != len(set(paths))
        ):
            raise ValueError(f"Invalid reference {split} split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(reference[a]) & set(reference[b]):
            raise ValueError(f"Overlapping reference {a}/{b} split")
    for split in keys:
        if sorted(manifest[split]) != sorted(reference[split]):
            raise RuntimeError(
                f"Reference {split} split differs from current dataset: refusing to train. "
                "Check Kaggle dataset mount, category, and seed."
            )
    return {split: len(manifest[split]) for split in keys}
