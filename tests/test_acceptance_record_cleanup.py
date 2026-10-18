"""Private E2E acceptance must not leave its own inspection records in history."""
import json

from PIL import Image
import pytest

from scripts import dual_model_acceptance as dual
from scripts import local_stack_acceptance as stack
from scripts import operator_workflow_acceptance as operator


def sample_files(tmp_path):
    a = tmp_path / "normal.png"
    b = tmp_path / "defective.png"
    Image.new("RGB", (8, 8), "blue").save(a)
    Image.new("RGB", (8, 8), "red").save(b)
    return a, b


def test_dual_model_collects_private_ids_but_never_outputs_them(monkeypatch, tmp_path):
    normal, defect = sample_files(tmp_path)
    bad_payload = dual.image_sample(defect)["png_bytes"]
    counter = iter(range(1, 5))
    monkeypatch.setattr(dual, "discover_models", lambda _: ("padim", "patchcore"))

    def predict(_origin, model, payload, timeout):
        return {
            "model": model,
            "predicted_anomaly": payload == bad_payload,
            "inspection_id": f"{next(counter):032x}",
        }

    monkeypatch.setattr(dual, "predict_png", predict)
    monkeypatch.setattr(dual, "validate_prediction",
                        lambda data, _model, _size: {
                            "predicted_anomaly": data["predicted_anomaly"],
                            "overlay_png_valid": True,
                        })
    ids = []
    report = dual.accept("http://127.0.0.1:8003", normal, defect,
                         inspection_sink=ids)
    assert report["status"] == "passed"
    assert ids == [f"{number:032x}" for number in range(1, 5)]
    assert "inspection_id" not in json.dumps(report)
    assert f"{1:032x}" not in json.dumps(report)


def test_capture_id_before_invalid_prediction_allows_cleanup(monkeypatch, tmp_path):
    normal, defect = sample_files(tmp_path)
    monkeypatch.setattr(dual, "discover_models", lambda _: ("padim", "patchcore"))
    monkeypatch.setattr(dual, "predict_png", lambda *_args, **_kw: {
        "inspection_id": "a" * 32,
    })
    monkeypatch.setattr(dual, "validate_prediction",
                        lambda *_: pytest.fail("Must fail before validation"))
    ids = []
    with pytest.raises(RuntimeError, match="unique inspection"):
        # First pass: captured ID, second prediction ID collides.
        dual.accept("http://127.0.0.1:8003", normal, defect, inspection_sink=ids)
    assert ids == ["a" * 32]


def test_cleanup_attempts_each_id_and_refuses_partial_success(monkeypatch):
    known = ["a" * 32, "b" * 32, "c" * 32]
    attempted = []

    def fake_call(_origin, path, method):
        attempted.append(path)
        if path.endswith("b" * 32):
            raise RuntimeError("temporary failure")
        return {"deleted": True, "inspection_id": path.rsplit("/", 1)[1]}

    monkeypatch.setattr(operator, "_json_request", fake_call)
    with pytest.raises(RuntimeError, match="1 of 3"):
        operator.cleanup_accepted_inspections("http://127.0.0.1:8003", known)
    assert len(attempted) == 3
    assert attempted == ["/inspections/" + i for i in known]


def test_cleanup_rejects_duplicate_or_unsafe_ids_without_deleting(monkeypatch):
    calls = []
    monkeypatch.setattr(operator, "_json_request",
                        lambda *_args, **_kw: calls.append(1))
    with pytest.raises(ValueError, match="duplicate"):
        operator.cleanup_accepted_inspections(
            "http://127.0.0.1:8003", ["a" * 32, "a" * 32]
        )
    with pytest.raises(ValueError, match="invalid"):
        operator.cleanup_accepted_inspections(
            "http://127.0.0.1:8003", ["../danger"]
        )
    assert calls == []


def setup_mock_stack(monkeypatch):
    data = {
        "/health": {"status": "ok", "model_configured": True},
        "/ready": {"status": "config_ready", "model_pairs": 2},
        "/models": {"models_by_category": {"bottle": ["padim", "patchcore"]}},
    }
    monkeypatch.setattr(stack, "_get", lambda _, path: (
        json.dumps(data[path]).encode() if path in data
        else {
            "/": b"IhsanVision", "/batch": b"Inspect a product batch",
            "/history": b"Private inspection history",
        }[path]
    ))
    monkeypatch.setattr(stack, "assert_unauthenticated_prediction_denied",
                        lambda _: None)
    monkeypatch.setenv("IHSANVISION_CLIENT_API_KEY", "test-private-key")


def test_local_stack_cleans_four_records_even_if_operator_flow_fails(
    monkeypatch, tmp_path,
):
    setup_mock_stack(monkeypatch)
    ids = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]
    cleaned = []

    def fake_accept(_origin, _normal, _defect, *, timeout, inspection_sink):
        inspection_sink.extend(ids)
        return {"status": "passed", "results": []}

    monkeypatch.setattr(stack, "accept", fake_accept)
    monkeypatch.setattr(stack, "image_sample",
                        lambda _: {"png_bytes": b"mock-png"})
    monkeypatch.setattr(stack, "verify_operator_workflow",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("operator test failed")))
    monkeypatch.setattr(stack, "cleanup_accepted_inspections",
                        lambda _, values: cleaned.extend(values))
    with pytest.raises(RuntimeError, match="operator test failed"):
        stack.verify_stack("http://127.0.0.1:8003",
                           tmp_path / "normal", tmp_path / "defect",
                           require_auth=True, check_operator=True)
    assert cleaned == ids


def test_local_stack_cleans_all_four_on_success(monkeypatch, tmp_path):
    setup_mock_stack(monkeypatch)
    ids = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]
    deleted = []

    def fake_accept(_origin, _normal, _defect, *, timeout, inspection_sink):
        inspection_sink.extend(ids)
        return {"status": "passed", "results": []}

    monkeypatch.setattr(stack, "accept", fake_accept)
    monkeypatch.setattr(stack, "image_sample", lambda _: {"png_bytes": b"mock"})
    monkeypatch.setattr(stack, "verify_operator_workflow",
                        lambda *_: {"status": "operator_workflow_passed"})
    monkeypatch.setattr(stack, "cleanup_accepted_inspections",
                        lambda _, values: deleted.extend(values))
    result = stack.verify_stack("http://127.0.0.1:8003",
                                tmp_path / "normal", tmp_path / "defect",
                                require_auth=True, check_operator=True)
    assert result["status"] == "local_stack_acceptance_passed"
    assert deleted == ids
    assert "a" * 32 not in json.dumps(result)
