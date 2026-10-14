"""Typed FastAPI output contract: no NaN/Inf scores or negative latency."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
import pytest

from src.api import create_app
from src.schemas import PredictionResponse


def test_valid_inference_output():
    result = PredictionResponse(
        model="padim",
        anomaly_score=0.824,
        predicted_anomaly=True,
        latency_ms=123.54,
        model_reused=True,
        heatmap_png_base64=None,
    )
    assert result.model == "padim"
    assert result.anomaly_score == 0.824
    assert result.model_reused is True


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_reject_non_finite_anomaly_score(score):
    with pytest.raises(ValidationError):
        PredictionResponse(model="padim", anomaly_score=score)


def test_reject_negative_latency_and_invalid_model():
    with pytest.raises(ValidationError):
        PredictionResponse(model="padim", anomaly_score=0.3, latency_ms=-5)
    with pytest.raises(ValidationError):
        PredictionResponse(model="unsupported", anomaly_score=0.3)


def test_openapi_declares_predict_response():
    openapi = TestClient(create_app()).get("/openapi.json")
    assert openapi.status_code == 200
    path = openapi.json()["paths"]["/predict"]["post"]
    response = path["responses"]["200"]
    assert "PredictionResponse" in str(response)
    schema = openapi.json()["components"]["schemas"]["PredictionResponse"]
    assert "anomaly_score" in schema["required"]
    assert "model" in schema["required"]
    assert "model_reused" in schema["properties"]


def test_api_serializes_optional_cache_field(monkeypatch, tmp_path):
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))

    def fake_predict(image, ckpt, model):
        return {
            "model": model,
            "anomaly_score": 0.71,
            "latency_ms": 44,
            "model_reused": True,
        }

    client = TestClient(create_app(predict_fn=fake_predict))
    image = BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")
    result = client.post(
        "/predict", files={"file": ("fake.png", image.getvalue(), "image/png")}
    )
    assert result.status_code == 200
    assert result.json()["model_reused"] is True
    assert result.json()["predicted_anomaly"] is None
