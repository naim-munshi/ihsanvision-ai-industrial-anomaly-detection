"""Offline regression tests for controlled sequential serving profile."""
import json
from pathlib import Path

import pytest

from scripts import benchmark_serving as b


def test_nearest_rank_quantiles():
    assert b.nearest_rank([1, 4, 8, 2, 6], 50) == 4
    assert b.nearest_rank([1, 4, 8, 2, 6], 95) == 8
    assert b.summary([20, 10, 30, 40])["p95_nearest_rank_ms"] == 40
    assert b.summary([20, 10, 30, 40])["median_ms"] == 25


def test_bad_timings_refused():
    with pytest.raises(ValueError):
        b.summary([])
    for item in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            b.summary([1, item])


def test_profile_separates_warmup_and_measured(monkeypatch):
    monkeypatch.setattr(b, "discover_models", lambda _: ("padim", "patchcore"))
    sequence = iter([
        {"round_trip_ms": 500, "model_predict_ms": 490, "model_reused": False},
        {"round_trip_ms": 400, "model_predict_ms": 390, "model_reused": True},
        {"round_trip_ms": 10, "model_predict_ms": 8, "model_reused": True},
        {"round_trip_ms": 20, "model_predict_ms": 17, "model_reused": True},
        {"round_trip_ms": 30, "model_predict_ms": 27, "model_reused": True},
        {"round_trip_ms": 40, "model_predict_ms": 36, "model_reused": True},
    ])
    calls = []
    def fake_measure(origin, model, payload, timeout):
        calls.append((origin, model, len(payload), timeout))
        return next(sequence)
    monkeypatch.setattr(b, "measure_once", fake_measure)
    clock = iter([100.0, 101.0])
    monkeypatch.setattr(b, "perf_counter", lambda: next(clock))
    report = b.profile("http://127.0.0.1:8003", "patchcore", warmup=2, requests=4)
    assert len(calls) == 6
    assert report["status"] == "sequential_profile_complete"
    assert report["warmup_requests"] == 2
    assert report["measured_requests"] == 4
    assert report["client_round_trip"]["median_ms"] == 25
    assert report["client_round_trip"]["p95_nearest_rank_ms"] == 40
    assert report["server_reported_prediction"]["p95_nearest_rank_ms"] == 36
    assert report["model_cache_reuse_counts"] == {"true": 4, "false": 0, "unknown": 0}
    assert report["warmup_cache_reuse_counts"] == {"true": 1, "false": 1, "unknown": 0}
    assert report["sequential_completed_requests_per_second"] == 4.0
    assert report["sample_type"] == "synthetic"
    assert len(report["sample_sha256_of_transmitted_png"]) == 64
    assert report["input_image_dimensions"] == [256, 256]
    assert "http://127" not in json.dumps(report)
    assert "heatmap" not in json.dumps(report)


def test_bad_url_model_counts_fail_before_requests(monkeypatch):
    monkeypatch.setattr(b, "discover_models", lambda _: ("padim",))
    with pytest.raises(RuntimeError, match="not configured"):
        b.profile("http://localhost:8003", "patchcore", requests=2)
    with pytest.raises(ValueError):
        b.profile("http://remote.example:8003", "padim", requests=2)
    with pytest.raises(ValueError):
        b.profile("http://localhost:8003", "padim", requests=1)
    with pytest.raises(ValueError):
        b.profile("http://localhost:8003", "padim", requests=501)
    with pytest.raises(ValueError):
        b.profile("http://localhost:8003", "padim", warmup=-1, requests=2)


def test_api_response_validation(monkeypatch):
    import io
    from urllib import request
    content = b'{"model":"padim","anomaly_score":0.6,"latency_ms":20,"model_reused":true}'
    class FakeResponse(io.BytesIO):
        pass
    monkeypatch.setattr(request, "urlopen", lambda req, timeout: FakeResponse(content))
    result = b.measure_once("http://localhost:8003", "padim", b"PNG", 3)
    assert result["model_reused"] is True
    assert result["model_predict_ms"] == 20
    assert result["round_trip_ms"] >= 0


def test_bad_score_rejected(monkeypatch):
    import io
    from urllib import request
    class FakeResponse(io.BytesIO):
        pass
    monkeypatch.setattr(
        request,
        "urlopen",
        lambda req, timeout: FakeResponse(b'{"model":"padim","anomaly_score":"bad","latency_ms":4}'),
    )
    with pytest.raises(RuntimeError, match="Invalid anomaly score"):
        b.measure_once("http://localhost:8003", "padim", b"PNG", 3)


def test_output_never_overwritten(tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"original": true}')
    with pytest.raises(SystemExit) as exc:
        b.main(["--model", "padim", "--output", str(output)])
    assert exc.value.code == 2
    assert json.loads(output.read_text()) == {"original": True}
