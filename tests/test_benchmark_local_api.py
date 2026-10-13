"""Offline latency aggregation and request consistency tests."""
import pytest

from scripts import benchmark_local_api as benchmark


def result(latency, *, reused, score=0.8242901564, label=True):
    return {
        "status": "api-smoke-passed",
        "model": "padim",
        "anomaly_score": score,
        "predicted_anomaly": label,
        "latency_ms_including_load": latency,
        "model_reused": reused,
        "overlay_png": True,
    }


def test_known_mac_pair_speed_and_warm_latency():
    reported = benchmark.summarize([
        result(721.99, reused=False),
        result(123.54, reused=True),
    ])
    assert reported["first_request_ms"] == 721.99
    assert reported["warm_reused_samples"] == 1
    assert reported["warm_inference_p50_ms"] == 123.54
    assert reported["warm_inference_p95_ms"] == 123.54
    assert reported["consistent_anomaly_score"] == pytest.approx(0.8242901564)


def test_nearest_rank_and_median_warm():
    payload = [
        result(900, reused=False),
        result(100, reused=True),
        result(150, reused=True),
        result(200, reused=True),
        result(300, reused=True),
        result(400, reused=True),
    ]
    report = benchmark.summarize(payload)
    assert report["warm_inference_p50_ms"] == 200
    assert report["warm_inference_p95_ms"] == 400


def test_already_warm_first_sample_is_reported_not_misclassified():
    results = [result(120, reused=True), result(119, reused=True)]
    summary = benchmark.summarize(results)
    assert summary["first_request_model_reused"] is True
    assert summary["warm_reused_samples"] == 2


def test_no_cache_does_not_pretend_warm():
    summary = benchmark.summarize([result(800, reused=False), result(700, reused=False)])
    assert summary["warm_reused_samples"] == 0
    assert summary["warm_inference_p50_ms"] is None
    assert summary["warm_inference_p95_ms"] is None


@pytest.mark.parametrize(
    "bad",
    [
        [result(100, reused=True, label=False), result(100, reused=True)],
        [result(100, reused=False), result(100, reused=True, score=0.6)],
        [result(100, reused=False), result(-1, reused=True)],
        [result(100, reused=False)],
    ],
)
def test_detect_bad_or_incomplete_results(bad):
    with pytest.raises(ValueError):
        benchmark.summarize(bad)


def test_run_makes_repeated_calls_with_same_image(monkeypatch):
    calls = []
    def fake_smoke(url, payload, *, timeout):
        calls.append((url, payload, timeout))
        return result(100 if len(calls) > 1 else 700, reused=len(calls) > 1)
    monkeypatch.setattr(benchmark, "smoke", fake_smoke)
    payload = b"PNG fixture"
    report = benchmark.run("http://127.0.0.1:8002", payload, runs=3)
    assert report["requests"] == 3
    assert report["warm_reused_samples"] == 2
    assert len(calls) == 3
    assert all(x == ("http://127.0.0.1:8002", payload, 180) for x in calls)


def test_run_rejects_excessive_requests():
    with pytest.raises(ValueError, match="runs"):
        benchmark.run("http://127.0.0.1:8002", b"PNG", runs=101)
