"""Dependency-isolated integration contracts; no pretrained weights required."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from src import predict, train
from src.config import TrainConfig


def fake_anomalib(monkeypatch, model_output=None):
    anomalib = ModuleType("anomalib")
    anomalib.__version__ = "2.5.0"
    data = ModuleType("anomalib.data")
    engine_module = ModuleType("anomalib.engine")

    class FakeData:
        def __init__(self, **kwargs):
            assert kwargs["val_split_mode"] == "from_test"
            assert kwargs["test_split_mode"] == "from_dir"
            self.train_data = SimpleNamespace(samples=pd.DataFrame(
                {"image_path": ["/mvtec/bottle/train/good/001.png"]}))
            self.val_data = SimpleNamespace(samples=pd.DataFrame(
                {"image_path": ["/mvtec/bottle/test/broken/001.png"]}))
            self.test_data = SimpleNamespace(samples=pd.DataFrame(
                {"image_path": ["/mvtec/bottle/test/broken/002.png"]}))

        def setup(self):
            return None

    class FakeTrainer:
        @staticmethod
        def save_checkpoint(path):
            Path(path).write_bytes(b"test-checkpoint")

    class FakeEngine:
        def __init__(self, **kwargs):
            self.trainer = FakeTrainer()

        def fit(self, *, model, datamodule):
            assert model is not None and datamodule is not None

        def test(self, *, model, datamodule, ckpt_path):
            assert Path(ckpt_path).is_file()
            return [{"image_AUROC": 0.86, "pixel_AUROC": 0.91}]

        def predict(self, *, model, data_path, ckpt_path, return_predictions):
            assert Path(data_path).is_file()
            assert Path(ckpt_path).is_file()
            assert return_predictions is True
            return [model_output]

    data.MVTecAD = FakeData
    engine_module.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "anomalib", anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.data", data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", engine_module)
    monkeypatch.setattr(train, "make_model", lambda _: object())
    monkeypatch.setattr(train, "seed_everything", lambda _: None)
    monkeypatch.setattr(predict, "make_model", lambda _, **kwargs: object())

    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_train_creates_report_and_checkpoint(monkeypatch, tmp_path):
    fake_anomalib(monkeypatch)
    root = tmp_path / "dataset"
    (root / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "bottle" / "test").mkdir(parents=True)
    report = train.run(TrainConfig(dataset_root=root, output_dir=tmp_path / "runs"))
    folder = tmp_path / "runs" / "padim-bottle-seed42"
    assert (folder / "model.ckpt").is_file()
    assert report["evaluation"]["metrics"]["image_AUROC"] == 0.86
    recorded = json.loads((folder / "splits.json").read_text())
    assert set(recorded) == {"train", "val", "test"}
    assert (folder / "report.json").is_file()


def test_predict_contract_with_checkpoint(monkeypatch, tmp_path):
    output = SimpleNamespace(pred_score=np.array([0.87]),
                             pred_label=np.array([1]),
                             anomaly_map=np.arange(64).reshape(1, 8, 8))
    fake_anomalib(monkeypatch, model_output=output)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (18, 14), color="white").save(image_path)
    ckpt = tmp_path / "trusted.ckpt"
    ckpt.write_bytes(b"fake")
    result = predict.predict_image(image_path, ckpt)
    assert result["anomaly_score"] == 0.87
    assert result["predicted_anomaly"] is True
    assert result["heatmap_png_base64"]
    assert result["latency_ms"] >= 0
