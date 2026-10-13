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

    def validate(self) -> None:
        if self.category not in CATEGORIES or self.model not in MODELS:
            raise ValueError("Unsupported MVTec category or model")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("Validation ratio must be between zero and one")
        if self.train_batch_size < 1 or self.eval_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid data loader settings")
        if not (self.dataset_root / self.category / "train" / "good").is_dir():
            raise FileNotFoundError("Dataset missing <root>/<category>/train/good")
        if not (self.dataset_root / self.category / "test").is_dir():
            raise FileNotFoundError("Dataset missing <root>/<category>/test")

    def serializable(self) -> dict:
        values = asdict(self)
        values["dataset_root"] = str(self.dataset_root.resolve())
        values["output_dir"] = str(self.output_dir)
        return values

def make_model(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    from anomalib.models import Padim, Patchcore
    return {"padim": Padim, "patchcore": Patchcore}[name]()

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
