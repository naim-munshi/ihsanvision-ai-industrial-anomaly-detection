"""End-to-end operator acceptance contract, using the real FastAPI app and private SQLite.

No external network, trained weights or dataset images are needed in CI.
"""
from io import BytesIO
import json
from urllib import error, parse

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from scripts import operator_workflow_acceptance as operator
from scripts import local_stack_acceptance as stack


def sample_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 12), "orange").save(output, format="PNG")
    return output.getvalue()


def client_and_adapter(tmp_path, monkeypatch, *, fail_route=None):
    for name in (
        "IHSANVISION_PADIM_CHECKPOINT", "IHSANVISION_PATCHCORE_CHECKPOINT",
        "IHSANVISION_MODEL_REGISTRY", "IHSANVISION_APPROVED_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("IHSANVISION_MODEL", "padim")
    monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(tmp_path / "private.sqlite3"))
    monkeypatch.setenv("IHSANVISION_API_KEY", "test-only-long-private-key")
    monkeypatch.setenv("IHSANVISION_CLIENT_API_KEY", "test-only-long-private-key")

    def predictor(image_path, saved_checkpoint, name):
        assert name == "padim" and saved_checkpoint == checkpoint
        assert image_path.read_bytes() == sample_png()
        return {
            "model": name, "anomaly_score": 0.15,
            "predicted_anomaly": False, "latency_ms": 3.0,
            "heatmap_png_base64": None,
        }

    client = TestClient(create_app(predict_fn=predictor))

    class Response:
        def __init__(self, body: bytes, status: int):
            self.body = body
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit: int) -> bytes:
            return self.body[:limit]

    def urlopen(req, timeout=180):
        assert req.full_url.startswith("http://127.0.0.1:8003/")
        path = req.full_url.split(":8003", 1)[1]
        if fail_route and path.startswith(fail_route):
            raise error.HTTPError(req.full_url, 503, "simulated failure", {}, None)
        response = client.request(
            req.get_method(), path,
            content=req.data, headers=dict(req.header_items()),
        )
        if response.status_code >= 400:
            raise error.HTTPError(req.full_url, response.status_code,
                                  "private request rejected", {}, None)
        return Response(response.content, response.status_code)

    monkeypatch.setattr(operator.request, "urlopen", urlopen)
    return client


def test_real_api_private_operator_workflow_creates_and_cleans_one_record(
    monkeypatch, tmp_path,
):
    client = client_and_adapter(tmp_path, monkeypatch)
    report = operator.verify_operator_workflow(
        "http://127.0.0.1:8003", sample_png()
    )
    assert report["status"] == "operator_workflow_passed"
    assert len(report["observed_workflow"]) == 9
    assert report["persisted_after_service_restart_verified"] is False
    assert report["browser_interaction_verified"] is False
    for forbidden in ("checkpoint_sha256", "lot_id", "inspection_id",
                      "test-only-long-private-key"):
        assert forbidden not in json.dumps(report)
    key = {"X-IhsanVision-Key": "test-only-long-private-key"}
    assert client.get("/inspections", headers=key).json()["total"] == 0
    assert client.get("/operator-metrics", headers=key).json()["inference_successful"] == 1


def test_private_failure_cleans_only_its_own_record(monkeypatch, tmp_path):
    client = client_and_adapter(
        tmp_path, monkeypatch, fail_route="/inspection-model-versions"
    )
    with pytest.raises(error.HTTPError):
        operator.verify_operator_workflow("http://127.0.0.1:8003", sample_png())
    key = {"X-IhsanVision-Key": "test-only-long-private-key"}
    assert client.get("/inspections", headers=key).json()["total"] == 0


def test_no_operator_key_or_remote_host_is_refused_before_network(monkeypatch):
    monkeypatch.delenv("IHSANVISION_CLIENT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CLIENT_API_KEY"):
        operator.verify_operator_workflow("http://127.0.0.1:8003", sample_png())
    monkeypatch.setenv("IHSANVISION_CLIENT_API_KEY", "private")
    with pytest.raises(ValueError, match="localhost"):
        operator.verify_operator_workflow("https://example.com", sample_png())


def test_invalid_csv_fails_and_still_cleans_test_row(monkeypatch, tmp_path):
    client = client_and_adapter(tmp_path, monkeypatch)
    original = operator._request_local

    def corrupt_csv(origin, route, **kwargs):
        if route.startswith("/inspections/export.csv"):
            return b"id,operator_verdict\nwrong,needs_followup\n"
        return original(origin, route, **kwargs)

    monkeypatch.setattr(operator, "_request_local", corrupt_csv)
    with pytest.raises(RuntimeError, match="CSV"):
        operator.verify_operator_workflow("http://127.0.0.1:8003", sample_png())
    key = {"X-IhsanVision-Key": "test-only-long-private-key"}
    assert client.get("/inspections", headers=key).json()["total"] == 0


def test_operator_mode_requires_explicit_auth_and_is_opt_in(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires --require-auth"):
        stack.verify_stack(
            "http://127.0.0.1:8003", tmp_path / "normal",
            tmp_path / "defect", check_operator=True, require_auth=False,
        )
    with pytest.raises(SystemExit):
        stack.main([
            "--normal", str(tmp_path / "normal"),
            "--defective", str(tmp_path / "defect"),
            "--output", str(tmp_path / "private.json"), "--operator-workflow",
        ])
