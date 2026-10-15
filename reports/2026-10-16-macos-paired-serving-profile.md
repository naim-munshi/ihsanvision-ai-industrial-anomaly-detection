# 2026-10-16 — macOS PaDiM vs PatchCore cached serving profile

## Source and validation scope

**Evidence:** User-provided macOS Terminal output showing existing, saved `scripts.benchmark_serving` JSON profiles for PaDiM and PatchCore. These two JSON files are held privately under `~/ihsanvision-artifacts/` and were **not copied to GitHub or directly inspected by the documentation author**. The user ran a local cross-check that returned:

- `Both completed: True`; each profile status `sequential_profile_complete`.
- `Same image: True` — the two **transmitted PNG SHA-256** hashes match.
- `Same dimensions: True` — both images `900 × 900`.
- Both use `sample_type: user_supplied`, 3 warmup requests and **20 measured** requests.
- Both report `model_cache_reuse_counts: {"true": 20, "false": 0, "unknown": 0}`, i.e. all measured requests reused the checkpoint.

No raw checkpoint, source image, base64 overlay, token, endpoint credential or private report is committed.

## Sequential HTTP round-trip results (milliseconds; lower is better)

| Statistic | PaDiM | PatchCore |
| --- | ---: | ---: |
| Mean | 458.533 | 493.983 |
| P50, nearest rank | 388.868 | 493.050 |
| P95, nearest rank | 766.458 | 514.819 |

## Server-reported model prediction span (milliseconds)

| Statistic | PaDiM | PatchCore |
| --- | ---: | ---: |
| Mean | 314.173 | 351.587 |
| P50, nearest rank | 244.270 | 351.420 |
| P95, nearest rank | 607.460 | 371.660 |

The PaDiM HTTP p95 is **251.639 ms higher** than PatchCore's p95 in this run; PatchCore's p95 is approximately **32.8% lower** relative to PaDiM. Conversely, PaDiM's HTTP mean is **35.450 ms lower** and its p50 is **104.182 ms lower**. The direction therefore differs between central latency and tail latency. The sample is too small to infer intrinsic model performance or statistical significance.

## Interpretation boundaries

- **Real locally measured sequential serving**, not an offline/mock profiler result, but the metrics below are transcribed from the user's reported profiles rather than independently regenerated from raw per-request timings.
- Reported `p50` and `p95` use **nearest rank**; for 20 requests, `p95` is effectively the 19th sorted request. The separate reported median uses a different definition for even sample sizes.
- The matched SHA-256 proves both profiler runs transmitted the *same encoded PNG bytes*, assuming the local verifier outputs are authentic; it does not prove the image is in the saved Kaggle held-out test manifest.
- All 40 measured requests reported `model_reused: true`; thus the experiment samples **warm/reused-serving behavior**, not cold-start speed. A shared in-process model cache is switched between families, and request order can bias timing.
- No claim about concurrency, high-percentile production latency, sustained QPS, CPU utilization, peak RSS, GPU utilization, scaling, raw anomaly score calibration or defect accuracy is supported.
- The available terminal output for this comparison did not include exact Python package/OS/accelerator version provenance; capture full per-run `system` metadata and reproduce under controlled conditions before attributing differences.
- A lower one-run p95 for PatchCore does **not** demonstrate that PatchCore is generally more stable or faster. PaDiM's lower mean/p50 do **not** demonstrate a universal advantage.

## Next steps

1. Run the paired read-only verifier `scripts.compare_serving_profiles` on the two preserved local JSON reports and archive its Markdown output outside Git; it checks SHA/workload/runtime consistency before comparing.
2. Repeat controlled alternating PaDiM/PatchCore trials on the same sample with more requests and consistent macOS power/load/thermal conditions.
3. Run `scripts.benchmark_concurrent` with a **small finite concurrency budget** and record both successes and rejections, plus measure memory footprint separately.
4. Continue M6 source test-split provenance and error-state checks, M9 real Docker builds and M10 release acceptance. Do not claim a public production SLA.
