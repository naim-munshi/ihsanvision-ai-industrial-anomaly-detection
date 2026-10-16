# 2026-10-16 — real Mac PaDiM finite concurrency-2 probe

## Evidence and setup

**Evidence:** User-provided Terminal output from `python -m scripts.benchmark_concurrent` run against a locally running IhsanVision API at `127.0.0.1:8003`. This report transcribes observed CLI output; the original JSON remains on the user's Mac at `~/ihsanvision-artifacts/padim_concurrent_probe_2026-10-16.json`.

- Model: `padim`; `test_type: finite_single_client_local_concurrent_probe`.
- Selected sample: `user_supplied` PNG; SHA-256 of transmitted encoded PNG `4d58bce0bf8242420879a4222ad9efe933784684a5eadc4d3f42b3594b0cd76d` (matches the prior sequential paired-report image hash).
- **Concurrency 2** (bounded local simultaneous HTTP submissions), **2 successful warmup requests**, **10 measured requests**.
- `status: all_requests_succeeded`; HTTP response status `200` for **10/10** requests; **zero rejected or failed** requests.
- All measured requests reported **`model_reused=true`**, 10/10.
- Measured wall-clock loop: **3.541 seconds**; observed completed successful requests per second **2.824** (finite-run statistic, not sustained capacity).

## Successful-only latency (milliseconds)

| Metric | HTTP client round-trip | Server-reported prediction span |
| --- | ---: | ---: |
| Minimum | 351.564 | 166.300 |
| Mean | 637.527 | 226.035 |
| Median | 635.084 | 190.335 |
| P50 (nearest rank) | 633.590 | 190.120 |
| P95 (nearest rank) | 1,019.311 | 553.210 |
| Maximum | 1,019.311 | 553.210 |
| Successful samples | 10 | 10 |

The observed HTTP p95 is about **1.02 seconds**, which includes processing and concurrent queuing. The API serializes Anomalib inference per process, so concurrent arrivals **are not evidence of parallel model execution**. Nearest-rank p95 with N=10 is effectively the maximum, and is highly variable across small samples. Do not compare this probe as a controlled head-to-head speed benchmark against previously captured sequential profiling: scheduling, request order and sample size differ.

## What this proves and what remains open

The user has observed the current local server accept and complete all 10 measured PaDiM requests under a bounded two-client-thread probe with cached checkpoint reuse. The instrument captured HTTP status counts and successful-only response latency. It **does not** establish server maximum throughput, production reliability, admission/rate-limit behavior under overload, or concurrency scale-out; there were **no** rejected requests in this run to exercise failure-path evidence.

Next repeat **the same** `--concurrency 2 --warmup 2 --requests 10` test with `--model patchcore` and the same input PNG. Then perform carefully bounded repeats, error/admission testing, and macOS memory/thermal profiling. Keep artifacts, input images and checkpoint binaries outside the repository.
