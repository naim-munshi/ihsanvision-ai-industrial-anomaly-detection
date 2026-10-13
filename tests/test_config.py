from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import TrainConfig, audit_splits, scalar_metrics


def datamodule(train, val, test):
    def dataset(paths):
        return SimpleNamespace(samples=pd.DataFrame({"image_path": paths}))
    return SimpleNamespace(train_data=dataset(train), val_data=dataset(val),
                           test_data=dataset(test))


def test_disjoint_manifest(tmp_path):
    paths = [str(tmp_path / f"{name}.png") for name in ("train", "val", "test")]
    manifest = audit_splits(datamodule([paths[0]], [paths[1]], [paths[2]]))
    assert all(len(values) == 1 for values in manifest.values())


def test_leakage_is_rejected(tmp_path):
    p = str(tmp_path / "shared.png")
    with pytest.raises(RuntimeError, match="leakage"):
        audit_splits(datamodule([p], [p], [str(tmp_path / "test.png")]))


def test_duplicate_is_rejected(tmp_path):
    p = str(tmp_path / "duplicate.png")
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_splits(datamodule([p, p], [str(tmp_path / "val.png")],
                                [str(tmp_path / "test.png")]))


def test_invalid_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainConfig(dataset_root=tmp_path).validate()


def test_valid_dataset(tmp_path):
    (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)
    (tmp_path / "bottle" / "test").mkdir(parents=True)
    TrainConfig(dataset_root=tmp_path).validate()


def test_bad_ratio(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(dataset_root=tmp_path, validation_ratio=1.0).validate()


def test_metrics_filter_non_finite_values():
    assert scalar_metrics([{"auroc": 0.9, "nan": float("nan"),
                            "nonscalar": [0.1, 0.2]}]) == {"auroc": 0.9}


def test_model_factory_passes_pretrained_flag(monkeypatch):
    """Checkpoint inference must not try to fetch ImageNet backbone weights."""
    import sys
    from types import ModuleType
    from src.config import make_model

    fake_anomalib = ModuleType("anomalib")
    fake_anomalib.__path__ = []
    fake_models = ModuleType("anomalib.models")
    recorded = []

    class FakePadim:
        def __init__(self, *, pre_trained=True):
            recorded.append(("padim", pre_trained))

    class FakePatchcore:
        def __init__(self, *, pre_trained=True):
            recorded.append(("patchcore", pre_trained))

    fake_models.Padim = FakePadim
    fake_models.Patchcore = FakePatchcore
    monkeypatch.setitem(sys.modules, "anomalib", fake_anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.models", fake_models)

    make_model("padim")
    make_model("padim", pre_trained=False)
    make_model("patchcore", pre_trained=False)
    assert recorded == [
        ("padim", True),
        ("padim", False),
        ("patchcore", False),
    ]
