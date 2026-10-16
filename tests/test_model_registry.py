"""Contract tests: category-aware trained-model discovery and inference routing."""
import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from ihsanvision.registry import configured_registry, model_catalog, public_catalog


def example_png():
    payload = BytesIO()
    Image.new("RGB", (14, 14), "blue").save(payload, format="PNG")
    return payload.getvalue()


def checkpoint(tmp_path, name):
    path = tmp_path / f"{name}.ckpt"
    path.write_bytes(b"locally trusted testing stub")
    return path


def registry(tmp_path, monkeypatch, entries):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 1, "models": entries}))
    monkeypatch.setenv("IHSANVISION_MODEL_REGISTRY", str(path))
    return path


def setup_catalog(tmp_path, monkeypatch):
    bottle = checkpoint(tmp_path, "bottle-padim")
    cable = checkpoint(tmp_path, "cable-patchcore")
    metal = checkpoint(tmp_path, "ad2-sheet")
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(bottle))
    monkeypatch.delenv("IHSANVISION_PATCHCORE_CHECKPOINT", raising=False)
    monkeypatch.delenv("IHSANVISION_CHECKPOINT", raising=False)
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    registry(tmp_path, monkeypatch, [
        {"category": "cable", "model": "patchcore", "checkpoint": str(cable)},
        {"category": "sheet_metal", "model": "padim", "checkpoint": str(metal)},
    ])
    return bottle, cable, metal


def test_discover_categories_and_never_disclose_paths(tmp_path, monkeypatch):
    bottle, cable, metal = setup_catalog(tmp_path, monkeypatch)
    client = TestClient(create_app())
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available_models"] == ["padim"]
    assert data["default_category"] == "bottle"
    assert data["default_model"] == "padim"
    assert data["models_by_category"] == {
        "bottle": ["padim"], "cable": ["patchcore"], "sheet_metal": ["padim"],
    }
    assert data["available_categories"] == ["bottle", "cable", "sheet_metal"]
    for path in (bottle, cable, metal):
        assert str(path) not in resp.text
    assert client.get("/health").json()["model_configured"] is True


def test_predict_routes_to_registered_category_and_model(tmp_path, monkeypatch):
    bottle, cable, metal = setup_catalog(tmp_path, monkeypatch)
    calls = []

    def predict(path, checkpoint, model_name):
        calls.append((checkpoint, model_name))
        return {"model": model_name, "anomaly_score": 0.79,
                "predicted_anomaly": True, "latency_ms": 2.5,
                "heatmap_png_base64": None}

    client = TestClient(create_app(predict_fn=predict))
    def request(extra=""):
        return client.post("/predict"+extra, files={"file": ("img.png", example_png(), "image/png")})

    assert request("?model=patchcore&category=cable").json()["category"] == "cable"
    assert calls[-1] == (cable.resolve(), "patchcore")
    assert request("?category=sheet_metal").json()["category"] == "sheet_metal"
    assert calls[-1] == (metal.resolve(), "padim")
    assert request("?model=padim").json()["category"] == "bottle"
    assert calls[-1] == (bottle, "padim")

    assert request("?category=../../private&model=padim").status_code == 400
    assert request("?category=cable&model=padim").status_code == 503
    assert request("?category=cable&model=unknown").status_code == 400
    assert len(calls) == 3


def test_category_defaults_to_first_registered_if_no_bottle(tmp_path, monkeypatch):
    monkeypatch.delenv("IHSANVISION_CHECKPOINT", raising=False)
    for name in ("PADIM", "PATCHCORE"):
        monkeypatch.delenv("IHSANVISION_"+name+"_CHECKPOINT", raising=False)
    ckpt = checkpoint(tmp_path, "can")
    registry(tmp_path, monkeypatch, [
        {"category": "can", "model": "patchcore", "checkpoint": str(ckpt)},
    ])
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    models = TestClient(create_app()).get("/models").json()
    assert models["default_category"] == "can"
    assert models["default_model"] == "patchcore"
    assert models["available_categories"] == ["can"]


@pytest.mark.parametrize("entries,error", [
    ([{"category": "private", "model": "padim", "checkpoint": "/tmp/model.ckpt"}],
     "Unsupported"),
    ([{"category": "bottle", "model": "x", "checkpoint": "/tmp/model.ckpt"}],
     "Unsupported"),
    ([{"category": "bottle", "model": "padim", "checkpoint": "model.ckpt"}],
     "absolute"),
    ([{"category": "bottle", "model": "padim", "checkpoint": "/tmp/model.ckpt",
       "unsafe": True}], "Every registry entry"),
])
def test_reject_invalid_registry_entries(tmp_path, monkeypatch, entries, error):
    registry(tmp_path, monkeypatch, entries)
    with pytest.raises(ValueError, match=error):
        configured_registry()


def test_reject_duplicate_models(tmp_path, monkeypatch):
    p = checkpoint(tmp_path, "both")
    row = {"category": "bottle", "model": "padim", "checkpoint": str(p)}
    registry(tmp_path, monkeypatch, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        configured_registry()


def test_registry_requires_absolute_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("IHSANVISION_MODEL_REGISTRY", "settings/model-registry.json")
    with pytest.raises(ValueError, match="absolute"):
        configured_registry()


def test_private_registry_only_not_client_checkpoint_path(tmp_path, monkeypatch):
    ckpt = checkpoint(tmp_path, "good")
    registry(tmp_path, monkeypatch, [{"category": "bottle", "model": "padim",
                                     "checkpoint": str(ckpt)}])
    result = model_catalog({})
    assert result["bottle"]["padim"] == ckpt.resolve()
    assert "checkpoint" not in public_catalog(result, "padim")
