"""Fast offline load-profile contract tests without a running inference server."""
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import benchmark_concurrent as benchmark


def fake_server(monkeypatch):
    monkeypatch.setattr(benchmark, "discover_models", lambda _: ("padim", "patchcore"))


def test_success_profile_records_all_requests_and_samples(monkeypatch):
    fake_server(monkeypatch)
    def mock_measure(origin, model, payload, timeout):
        assert model == "patchcore" and payload
        return {"round_trip_ms": 10.0, "model_predict_ms": 8.0, "model_reused": True}
    monkeypatch.setattr(benchmark, "measure_once", mock_measure)
    result = benchmark.run_load("http://127.0.0.1:8003", "patchcore", concurrency=3, requests=7, warmup=2)
    assert result["status"] == "all_requests_succeeded"
    assert result["results_successful"] == 7
    assert result["results_failed_or_rejected"] == 0
    assert result["response_status_counts"] == {"200": 7}
    assert result["successful_client_round_trip"]["p95_nearest_rank_ms"] == 10
    assert result["successful_cache_reuse_counts"] == {"true": 7, "false": 0, "unknown": 0}


def test_rejection_and_failure_count_not_hidden(monkeypatch):
    fake_server(monkeypatch)
    def mocked_attempt(origin, model, payload, timeout):
        with mocked_attempt_lock:
            x = calls[0]
            calls[0] += 1
        if x == 0:
            return {"status": "200", "round_trip_ms": 25.0, "model_predict_ms": 15.0, "model_reused": False}
        return {"status": "503", "round_trip_ms": 4.0}
    from threading import Lock
    mocked_attempt_lock = Lock()
    calls = [0]
    monkeypatch.setattr(benchmark, "attempt", mocked_attempt)
    result = benchmark.run_load("http://localhost:8003", "padim", concurrency=2, requests=4, warmup=0)
    assert result["status"] == "completed_with_rejections_or_errors"
    assert result["results_successful"] == 1
    assert result["results_failed_or_rejected"] == 3
    assert result["response_status_counts"] == {"200": 1, "503": 3}
    assert result["successful_client_round_trip"]["samples"] == 1


def test_no_success_is_reported_not_divided_by_zero(monkeypatch):
    fake_server(monkeypatch)
    monkeypatch.setattr(benchmark, "attempt", lambda *args: {"status": "429", "round_trip_ms": 1.0})
    result = benchmark.run_load("http://localhost:8003", "padim", concurrency=2, requests=3, warmup=0)
    assert result["results_successful"] == 0
    assert result["successful_client_round_trip"] is None
    assert result["successful_server_prediction"] is None
    assert result["response_status_counts"] == {"429": 3}


def test_warmup_failure_prevents_misleading_run(monkeypatch):
    fake_server(monkeypatch)
    monkeypatch.setattr(benchmark, "attempt", lambda *args: {"status": "503", "round_trip_ms": 1.0})
    with pytest.raises(RuntimeError, match="Warmup failed"):
        benchmark.run_load("http://localhost:8003", "padim", concurrency=2, requests=4, warmup=1)


@pytest.mark.parametrize("settings", [
    {"concurrency": 0, "requests": 5}, {"concurrency": 17, "requests": 20},
    {"concurrency": 3, "requests": 2}, {"concurrency": 2, "requests": 501},
    {"concurrency": 2, "requests": 4, "warmup": -1},
    {"concurrency": 2, "requests": 4, "timeout": 601},
])
def test_invalid_load_limits_refused(settings):
    with pytest.raises(ValueError):
        benchmark.run_load("http://127.0.0.1:8003", "padim", **settings)


def test_attempt_http_error_redacts_response(monkeypatch):
    def fail(origin, model, payload, timeout):
        raise HTTPError(origin, 503, "private error path", {}, None)
    monkeypatch.setattr(benchmark, "measure_once", fail)
    result = benchmark.attempt("http://127.0.0.1:8003", "padim", b"image", 5)
    assert result["status"] == "503"
    assert "private" not in str(result)


def test_existing_output_not_overwritten(tmp_path):
    out = tmp_path / "old.json"
    out.write_text('{"old":true}')
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--model", "padim", "--output", str(out)])
    assert exc.value.code == 2
    assert out.read_text() == '{"old":true}'
