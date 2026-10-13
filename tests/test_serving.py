"""Offline persistent model serving contracts; no Anomalib weights downloaded."""
import base64
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.serving import PersistentModelPredictor


def setup_fake_engine(monkeypatch):
    engines = []
    anomalib = ModuleType("anomalib")
    anomalib.__path__ = []
    engine_module = ModuleType("anomalib.engine")
    anomalib.engine = engine_module

    class FakeEngine:
        def __init__(self, **kwargs):
            self.calls = []
            engines.append(self)

        def predict(self, *, model, data_path, ckpt_path, return_predictions):
            assert Path(data_path).is_file()
            assert return_predictions
            self.calls.append((model, ckpt_path))
            return [SimpleNamespace(
                pred_score=np.array([0.81]),
                pred_label=np.array([1]),
                anomaly_map=np.arange(16).reshape(1, 4, 4),
            )]

    engine_module.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "anomalib", anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.engine", engine_module)
    created = []
    def fake_model(name):
        obj = object()
        created.append((name, obj))
        return obj
    monkeypatch.setattr("src.serving.make_model", fake_model)
    return engines, created


def make_inputs(tmp_path):
    image = tmp_path / "bottle.png"
    Image.new("RGB", (12, 9), "white").save(image)
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"trusted checkpoint fixture")
    return image, checkpoint


def test_checkpoint_loaded_once_and_model_reused(monkeypatch, tmp_path):
    engines, created = setup_fake_engine(monkeypatch)
    image, checkpoint = make_inputs(tmp_path)
    predictor = PersistentModelPredictor()

    first = predictor(image, checkpoint, "padim")
    second = predictor(image, checkpoint, "padim")

    assert len(engines) == 1
    assert len(created) == 1
    assert engines[0].calls[0] == (created[0][1], str(checkpoint))
    assert engines[0].calls[1] == (created[0][1], None)
    assert first["model_reused"] is False
    assert second["model_reused"] is True
    assert first["anomaly_score"] == second["anomaly_score"] == 0.81
    assert first["predicted_anomaly"] is True
    assert base64.b64decode(second["heatmap_png_base64"]).startswith(b"\x89PNG")


def test_changed_checkpoint_invalidates_cache(monkeypatch, tmp_path):
    engines, _ = setup_fake_engine(monkeypatch)
    image, checkpoint = make_inputs(tmp_path)
    predictor = PersistentModelPredictor()
    predictor(image, checkpoint, "padim")
    checkpoint.write_bytes(b"different trusted checkpoint contents")
    out = predictor(image, checkpoint, "padim")
    assert len(engines) == 2
    assert out["model_reused"] is False
    assert engines[1].calls[0][1] == str(checkpoint)


def test_switched_model_invalidates_cache(monkeypatch, tmp_path):
    engines, created = setup_fake_engine(monkeypatch)
    image, checkpoint = make_inputs(tmp_path)
    predictor = PersistentModelPredictor()
    predictor(image, checkpoint, "padim")
    predictor(image, checkpoint, "patchcore")
    assert len(engines) == 2
    assert [x[0] for x in created] == ["padim", "patchcore"]


def test_reject_invalid_checkpoint(monkeypatch, tmp_path):
    engines, created = setup_fake_engine(monkeypatch)
    image, checkpoint = make_inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        PersistentModelPredictor()(image, tmp_path / "missing.ckpt", "padim")
    with pytest.raises(ValueError):
        PersistentModelPredictor()(image, checkpoint, "unknown")
    assert engines == []
    assert created == []


def test_failed_first_prediction_does_not_mark_cached(monkeypatch, tmp_path):
    engines, _ = setup_fake_engine(monkeypatch)
    image, checkpoint = make_inputs(tmp_path)
    predictor = PersistentModelPredictor()
    original = engines
    # First attempt fails before successful prediction; next attempt must reload weights.
    from anomalib.engine import Engine
    def faulty_engine(**kwargs):
        obj = Engine(**kwargs)
        old_predict = obj.predict
        called = False
        def fail_once(**args):
            nonlocal called
            if not called:
                called = True
                raise RuntimeError("transient")
            return old_predict(**args)
        obj.predict = fail_once
        return obj
    monkeypatch.setattr("anomalib.engine.Engine", faulty_engine)
    with pytest.raises(RuntimeError, match="transient"):
        predictor(image, checkpoint, "padim")
    out = predictor(image, checkpoint, "padim")
    assert out["model_reused"] is False
    assert len(original) == 1
    assert original[0].calls[0][1] == str(checkpoint)
