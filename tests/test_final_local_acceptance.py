"""Offline contract tests for final localhost portfolio acceptance."""
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image
import pytest

from scripts import final_local_acceptance as final


class Response:
    def __init__(self, data, status=200):
        self.status = status
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return self._data

    def __iter__(self):
        return iter(self._data.splitlines())


def mock_local_api(monkeypatch, *, bad_status=None):
    calls = []

    def fake_urlopen(req, timeout):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/health"):
            return Response(b'{"status":"ok","model_configured":true}')
        if url.endswith("/"):
            return Response(b"<html>IhsanVision dashboard</html>")
        if "/predict" in url:
            calls.append(url)
            expected = 400 if "unknown" in url else 415 if req.data.startswith(b"--") else 422
            code = bad_status if bad_status is not None else expected
            raise HTTPError(url, code, "rejected", {}, BytesIO(b'{"detail":"rejected"}'))
        raise AssertionError("Unexpected network request")

    monkeypatch.setattr(final.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(final, "discover_models", lambda origin: ("padim", "patchcore"))
    return calls


def test_service_and_negative_api_contract(monkeypatch):
    calls = mock_local_api(monkeypatch)
    result = final.check_service_and_errors("http://127.0.0.1:8003")
    assert result["health"] == "ok"
    assert result["dashboard"] == "served"
    assert result["available_models"] == ["padim", "patchcore"]
    assert [case["actual_status"] for case in result["rejected_request_checks"]] == [400, 415, 422]
    assert len(calls) == 3


def test_wrong_rejection_code_fails_closed(monkeypatch):
    mock_local_api(monkeypatch, bad_status=500)
    with pytest.raises(RuntimeError, match="expected HTTP 400, received 500"):
        final.check_service_and_errors("http://127.0.0.1:8003")


def test_api_key_auth_failure_has_actionable_private_diagnostic(monkeypatch):
    mock_local_api(monkeypatch, bad_status=401)
    with pytest.raises(RuntimeError, match="IHSANVISION_CLIENT_API_KEY") as exc:
        final.check_service_and_errors("http://127.0.0.1:8003")
    assert "HTTP 401" in str(exc.value)
    assert "private-secret-example" not in str(exc.value)


def test_reject_remote_host_before_network(monkeypatch):
    monkeypatch.setattr(final.request, "urlopen", lambda *_args, **_kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError, match="localhost"):
        final.check_service_and_errors("https://external.example:8003")


def test_end_to_end_report_does_not_claim_release(monkeypatch, tmp_path):
    mock_local_api(monkeypatch)
    monkeypatch.setattr(final, "accept", lambda *_args, **_kwargs: {
        "status": "passed",
        "results": [{"model": "padim", "predicted_as_expected": True}],
        "sample_evidence": {"normal": {"filename": "normal.png", "sha256": "a" * 64}},
    })
    report = final.run_acceptance(
        "http://localhost:8003", tmp_path / "normal.png", tmp_path / "defective.png"
    )
    assert report["status"] == "local_mvp_acceptance_passed"
    assert report["release_authorized"] is False
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "heatmap_png_base64" not in serialized


def test_model_mismatch_preserved_as_failure(monkeypatch, tmp_path):
    mock_local_api(monkeypatch)
    monkeypatch.setattr(final, "accept", lambda *_args, **_kwargs: {
        "status": "label_mismatch", "results": [{"predicted_as_expected": False}],
    })
    report = final.run_acceptance(
        "http://localhost:8003", tmp_path / "normal.png", tmp_path / "defective.png"
    )
    assert report["status"] == "local_mvp_acceptance_failed"


def test_existing_evidence_file_not_overwritten(tmp_path):
    out = tmp_path / "old.json"
    out.write_text('{"preserved":true}')
    with pytest.raises(SystemExit) as exc:
        final.main(["--normal", "normal.png", "--defective", "defect.png", "--output", str(out)])
    assert exc.value.code == 2
    assert out.read_text() == '{"preserved":true}'


def test_missing_upload_rejection_uses_no_private_input(monkeypatch):
    mock_local_api(monkeypatch)
    result = final.check_service_and_errors("http://localhost:8003")
    assert result["rejected_request_checks"][-1]["case"] == "missing_upload"
    assert "API_KEY" not in json.dumps(result)
