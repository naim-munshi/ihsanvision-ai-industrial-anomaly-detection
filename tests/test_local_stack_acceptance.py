"""Offline harness tests; real API/model validation runs only on operator Mac."""
import json
from pathlib import Path

import pytest

from scripts import local_stack_acceptance as stack


def mock_stack(monkeypatch):
    payloads = {
        "/health": json.dumps({"status": "ok", "model_configured": True}).encode(),
        "/ready": json.dumps({"status": "config_ready", "model_pairs": 2,
                              "inference_verified": False}).encode(),
        "/models": json.dumps({"models_by_category": {
            "bottle": ["padim", "patchcore"]}}).encode(),
        "/": b"<h1>IhsanVision</h1>",
        "/batch": b"<h1>Inspect a product batch</h1>",
        "/history": b"<h1>Private inspection history</h1>",
    }
    monkeypatch.setattr(stack, "_get", lambda _origin, route: payloads[route])
    monkeypatch.setattr(stack, "accept", lambda *_a, **_k: {
        "status": "passed", "results": [
            {"model": "padim", "sample": "normal", "overlay_png_valid": True},
            {"model": "padim", "sample": "defective", "overlay_png_valid": True},
            {"model": "patchcore", "sample": "normal", "overlay_png_valid": True},
            {"model": "patchcore", "sample": "defective", "overlay_png_valid": True},
        ],
    })
    return payloads


def test_local_end_to_end_orchestrator(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    report = stack.verify_stack(
        "http://127.0.0.1:8003", tmp_path / "normal.png",
        tmp_path / "defect.png",
    )
    assert report["status"] == "local_stack_acceptance_passed"
    assert len(report["inference"]["results"]) == 4
    assert report["checks"]["missing_api_key_refused"] == "not_checked"


def test_remote_origins_are_refused_before_io(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    with pytest.raises(ValueError, match="localhost"):
        stack.verify_stack("https://example.com", tmp_path / "n", tmp_path / "d")


def test_missing_models_or_ui_fail_closed(monkeypatch, tmp_path):
    payloads = mock_stack(monkeypatch)
    payloads["/models"] = b'{"models_by_category":{"bottle":["padim"]}}'
    with pytest.raises(RuntimeError, match="Both bottle"):
        stack.verify_stack("http://localhost:8003", tmp_path / "n", tmp_path / "d")
    payloads["/models"] = b'{"models_by_category":{"bottle":["padim","patchcore"]}}'
    payloads["/batch"] = b"missing"
    with pytest.raises(RuntimeError, match="dashboard"):
        stack.verify_stack("http://localhost:8003", tmp_path / "n", tmp_path / "d")


def test_failed_prediction_labels_cannot_be_passed(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    monkeypatch.setattr(stack, "accept", lambda *_a, **_k: {
        "status": "label_mismatch", "results": [],
    })
    report = stack.verify_stack("http://localhost:8003", tmp_path / "n", tmp_path / "d")
    assert report["status"] == "prediction_label_mismatch"


def test_cli_writes_private_nonoverwriting_evidence(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    output = tmp_path / "acceptance.json"
    args = ["--normal", str(tmp_path / "n"), "--defective", str(tmp_path / "d"),
            "--output", str(output)]
    stack.main(args)
    assert json.loads(output.read_text())["status"] == "local_stack_acceptance_passed"
    assert output.stat().st_mode & 0o077 == 0
    with pytest.raises(SystemExit):
        stack.main(args)


def test_operator_mode_rejects_missing_auth_before_io(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    with pytest.raises(ValueError, match="requires --require-auth"):
        stack.verify_stack(
            "http://127.0.0.1:8003", tmp_path / "normal.png",
            tmp_path / "defect.png", check_operator=True,
        )


def test_operator_mode_verifies_and_cleans_all_temporary_inspections(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    monkeypatch.setenv("IHSANVISION_CLIENT_API_KEY", "test-only-key")
    monkeypatch.setattr(stack, "assert_unauthenticated_prediction_denied",
                        lambda origin: None)
    recorded = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]
    cleaned = []

    def fake_accept(*args, inspection_sink=None, **kwargs):
        assert inspection_sink is not None
        inspection_sink.extend(recorded)
        return {"status": "passed", "results": []}

    monkeypatch.setattr(stack, "accept", fake_accept)
    monkeypatch.setattr(stack, "image_sample", lambda _path: {"png_bytes": b"png"})
    monkeypatch.setattr(stack, "verify_operator_workflow",
                        lambda origin, image: {"status": "operator_workflow_passed"})
    monkeypatch.setattr(stack, "cleanup_accepted_inspections",
                        lambda origin, ids: cleaned.extend(ids))
    report = stack.verify_stack(
        "http://127.0.0.1:8003", tmp_path / "normal.png",
        tmp_path / "defect.png", require_auth=True, check_operator=True,
    )
    assert report["status"] == "local_stack_acceptance_passed"
    assert report["operator_workflow"]["status"] == "operator_workflow_passed"
    assert report["checks"]["missing_api_key_refused"] == "passed"
    assert cleaned == recorded


def test_operator_mode_cleans_temporary_rows_if_inference_fails(monkeypatch, tmp_path):
    mock_stack(monkeypatch)
    monkeypatch.setenv("IHSANVISION_CLIENT_API_KEY", "test-only-key")
    monkeypatch.setattr(stack, "assert_unauthenticated_prediction_denied",
                        lambda origin: None)
    cleaned = []

    def failing_accept(*args, inspection_sink=None, **kwargs):
        inspection_sink.append("f" * 32)
        raise RuntimeError("inference interrupted after first inspection")

    monkeypatch.setattr(stack, "accept", failing_accept)
    monkeypatch.setattr(stack, "cleanup_accepted_inspections",
                        lambda origin, ids: cleaned.extend(ids))
    with pytest.raises(RuntimeError, match="inference interrupted"):
        stack.verify_stack(
            "http://127.0.0.1:8003", tmp_path / "normal.png",
            tmp_path / "defect.png", require_auth=True, check_operator=True,
        )
    assert cleaned == ["f" * 32]
