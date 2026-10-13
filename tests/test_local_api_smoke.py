"""Offline smoke-test adapter tests: no network sockets or model weights."""
from io import BytesIO
import json

from PIL import Image
import pytest

from scripts import local_api_smoke as smoke


def test_synthetic_image_is_valid():
    raw = smoke.make_png()
    assert Image.open(BytesIO(raw)).size == (256, 256)


def test_multipart_contains_expected_file_field():
    payload = smoke.make_png()
    encoded, mime = smoke.multipart_png(payload)
    assert mime.startswith("multipart/form-data; boundary=ihsanvision-")
    assert b'name="file"; filename="smoke.png"' in encoded
    assert payload in encoded


def test_localhost_only():
    with pytest.raises(ValueError, match="localhost"):
        smoke.smoke("https://example.com", b"fake")


def test_smoke_contract(monkeypatch):
    payload = smoke.make_png()
    response_data = {
        "model": "padim", "anomaly_score": 0.55,
        "predicted_anomaly": False, "latency_ms": 350,
        "heatmap_png_base64": None,
    }

    class FakeResponse:
        status = 200
        def __init__(self, content):
            self.content = content
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self.content

    calls = []
    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            calls.append(request)
            if request.endswith("/health"):
                return FakeResponse(json.dumps({"model_configured": True}).encode())
            if request.endswith("/"):
                return FakeResponse(b"IhsanVision")
        else:
            calls.append(request.full_url)
            return FakeResponse(json.dumps(response_data).encode())
        raise AssertionError("Unexpected HTTP request")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)
    result = smoke.smoke("http://127.0.0.1:8001", payload)
    assert result["status"] == "api-smoke-passed"
    assert result["overlay_png"] is False
    assert len(calls) == 3
