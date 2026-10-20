"""Offline contract coverage for the synthetic five-category localhost smoke test."""
import base64
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import multicategory_smoke as smoke
from scripts.local_api_smoke import make_png


def response(category="cable", model="padim", overlay=None, inspection_id=None):
    result = {
        "category": category,
        "model": model,
        "anomaly_score": 0.73,
        "predicted_anomaly": False,
        "latency_ms": 14.5,
        "model_reused": False,
        "heatmap_png_base64": base64.b64encode(
            overlay if overlay is not None else make_png()
        ).decode("ascii"),
    }
    if inspection_id is not None:
        result["inspection_id"] = inspection_id
    return result


def test_response_checks_model_routing_and_valid_overlay():
    evidence = smoke.validate_response(
        response(), category="cable", model="padim", dimensions=(256, 256))
    assert evidence["overlay_png_valid"] is True
    assert evidence["label_present"] is True


@pytest.mark.parametrize("changes", [
    {"model": "patchcore"},
    {"category": "metal_nut"},
    {"anomaly_score": float("nan")},
    {"heatmap_png_base64": "not a base64 encoded png"},
    {"predicted_anomaly": "yes"},
    {"latency_ms": -1},
])
def test_malformed_prediction_is_rejected(changes):
    payload = response()
    payload.update(changes)
    with pytest.raises(ValueError):
        smoke.validate_response(payload, category="cable", model="padim",
                                dimensions=(256, 256))


def test_verify_pairs_exercises_both_models_and_cleans_only_new_records(monkeypatch):
    posted = []
    deleted = []

    def stub_request(url, *, body=None, headers=None, method="GET", timeout=30):
        path = urlsplit(url).path
        if path == "/models":
            return {"models_by_category": {"cable": ["padim", "patchcore"]}}
        if path == "/ready":
            return {"status": "config_ready", "approved_sha256_verified": True,
                    "model_pairs": 10}
        if path == "/predict":
            assert method == "POST"
            assert headers["X-IhsanVision-Key"] == "a" * 32
            assert body
            qs = parse_qs(urlsplit(url).query)
            model = qs["model"][0]
            posted.append(model)
            return response(category="cable", model=model,
                            inspection_id="a" * 31 + str(len(posted)))
        if path.startswith("/inspections/"):
            assert method == "DELETE"
            identifier = path.rsplit("/", 1)[1]
            deleted.append(identifier)
            return {"deleted": True, "inspection_id": identifier}
        raise AssertionError("Unexpected endpoint")

    monkeypatch.setattr(smoke, "request_json", stub_request)
    got = smoke.verify_pairs("http://127.0.0.1:8004", ["cable"], "a" * 32)
    assert len(got) == 2
    assert posted == ["padim", "patchcore"]
    assert deleted == ["a" * 31 + "1", "a" * 31 + "2"]


def test_cleanup_still_runs_on_invalid_prediction(monkeypatch):
    deleted = []

    def stub_request(url, *, body=None, headers=None, method="GET", timeout=30):
        path = urlsplit(url).path
        if path == "/models":
            return {"models_by_category": {"capsule": ["padim", "patchcore"]}}
        if path == "/ready":
            return {"status": "config_ready", "approved_sha256_verified": True,
                    "model_pairs": 10}
        if path == "/predict":
            result = response(category="capsule", model="padim",
                              inspection_id="b" * 32)
            result["anomaly_score"] = "invalid"
            return result
        if path == "/inspections/" + "b" * 32:
            deleted.append(True)
            return {"deleted": True, "inspection_id": "b" * 32}
        raise AssertionError("Unexpected endpoint")

    monkeypatch.setattr(smoke, "request_json", stub_request)
    with pytest.raises(ValueError):
        smoke.verify_pairs("http://127.0.0.1:8004", ["capsule"], "a" * 32)
    assert deleted == [True]


def test_rejects_bad_auth_value_before_requests(monkeypatch):
    monkeypatch.setattr(smoke, "request_json", lambda *a, **kw: pytest.fail("No HTTP expected"))
    with pytest.raises(ValueError):
        smoke.verify_pairs("http://127.0.0.1:8004", ["bottle"], "pasted\ncommand")
