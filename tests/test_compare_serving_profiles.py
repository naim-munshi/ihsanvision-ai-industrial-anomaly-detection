"""Offline regression tests for read-only paired local serving comparison."""
import json
from pathlib import Path

import pytest

from scripts import compare_serving_profiles as compare


SHA = "a" * 64


def profile(model: str, *, requests=20, warmup=3):
    return {
        "schema_version": 1,
        "status": "sequential_profile_complete",
        "model": model,
        "sample_type": "user_supplied",
        "sample_sha256_of_transmitted_png": SHA,
        "input_image_dimensions": [900, 900],
        "system": {"os": "Darwin", "machine": "arm64", "python_version": "3.12.1"},
        "warmup_requests": warmup,
        "measured_requests": requests,
        "model_cache_reuse_counts": {"true": requests, "false": 0, "unknown": 0},
        "client_round_trip": {
            "samples": requests,
            "mean_ms": 450.123,
            "p50_nearest_rank_ms": 390.234,
            "p95_nearest_rank_ms": 740.345,
        },
        "server_reported_prediction": {
            "samples": requests,
            "mean_ms": 300.123,
            "p50_nearest_rank_ms": 250.234,
            "p95_nearest_rank_ms": 520.345,
        },
    }


def write_pair(tmp_path, padim=None, patchcore=None):
    a = tmp_path / "padim.json"
    b = tmp_path / "patchcore.json"
    a.write_text(json.dumps(padim or profile("padim")))
    b.write_text(json.dumps(patchcore or profile("patchcore")))
    return a, b


def test_valid_pair_generates_grounded_report(tmp_path):
    a, b = write_pair(tmp_path)
    report = compare.compare(a, b)
    assert "20 measured requests per model" in report
    assert "3 warmup requests" in report
    assert "900 × 900" in report
    assert SHA in report
    assert "450.123" in report
    assert "390.234" in report
    assert "740.345" in report
    assert "100% reported cache reuse" in report
    assert "not a concurrent throughput" in report
    assert str(tmp_path) not in report


@pytest.mark.parametrize("field,updated", [
    ("sample_type", "synthetic"),
    ("sample_sha256_of_transmitted_png", "b" * 64),
    ("input_image_dimensions", [256, 256]),
    ("warmup_requests", 2),
    ("measured_requests", 30),
])
def test_reject_mismatched_workloads(tmp_path, field, updated):
    other = profile("patchcore")
    other[field] = updated
    if field == "measured_requests":
        other["model_cache_reuse_counts"]["true"] = 30
        other["client_round_trip"]["samples"] = 30
        other["server_reported_prediction"]["samples"] = 30
    a, b = write_pair(tmp_path, patchcore=other)
    with pytest.raises(ValueError, match="differing"):
        compare.compare(a, b)


def test_reject_different_runtime(tmp_path):
    other = profile("patchcore")
    other["system"]["machine"] = "x86_64"
    a, b = write_pair(tmp_path, patchcore=other)
    with pytest.raises(ValueError, match="system.machine"):
        compare.compare(a, b)


def test_reject_invalid_numbers(tmp_path):
    other = profile("patchcore")
    other["server_reported_prediction"]["mean_ms"] = float("nan")
    a, b = write_pair(tmp_path, patchcore=other)
    with pytest.raises(ValueError, match="Invalid numeric"):
        compare.compare(a, b)


def test_reject_cache_miss_and_unknown(tmp_path):
    other = profile("patchcore")
    other["model_cache_reuse_counts"] = {"true": 19, "false": 1, "unknown": 0}
    a, b = write_pair(tmp_path, patchcore=other)
    with pytest.raises(ValueError, match="cold/unknown"):
        compare.compare(a, b)


def test_reject_wrong_model_name(tmp_path):
    other = profile("padim")
    a, b = write_pair(tmp_path, patchcore=other)
    with pytest.raises(ValueError, match="wrong model"):
        compare.compare(a, b)


def test_refuse_overwriting_previous_report(tmp_path):
    a, b = write_pair(tmp_path)
    out = tmp_path / "paired.md"
    out.write_text("existing audit record")
    with pytest.raises(SystemExit) as exc:
        compare.main(["--padim", str(a), "--patchcore", str(b), "--output", str(out)])
    assert exc.value.code == 2
    assert out.read_text() == "existing audit record"


def test_successful_cli_writes_valid_markdown(tmp_path, capsys):
    a, b = write_pair(tmp_path)
    out = tmp_path / "paired.md"
    compare.main(["--padim", str(a), "--patchcore", str(b), "--output", str(out)])
    assert out.read_text().startswith("# PaDiM vs PatchCore")
    assert "Saved paired profile" in capsys.readouterr().out
