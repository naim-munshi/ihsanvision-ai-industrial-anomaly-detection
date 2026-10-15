# M7 — Sequential localhost serving performance protocol

## Scope

The profiler (scripts/benchmark_serving.py) runs HTTP POST /predict requests against a **running localhost** IhsanVision API. Warmup calls are excluded from measured distributions. It records client round-trip and the server's reported prediction span separately.

The report includes min, mean, median, nearest-rank p50/p95 and max, model reuse counts, and completed requests per second for a strictly **sequential single-client** loop. This is **not** concurrent load testing or server capacity measurement.

## Mac commands

In one terminal, launch the API with the two trusted checkpoint paths configured. In another, use a checkout containing this feature and execute:

~~~bash
python -m scripts.benchmark_serving \
  --url http://127.0.0.1:8003 \
  --model padim --warmup 3 --requests 30 \
  --output "$HOME/ihsanvision-artifacts/padim_serving_profile.json"

python -m scripts.benchmark_serving \
  --url http://127.0.0.1:8003 \
  --model patchcore --warmup 3 --requests 30 \
  --output "$HOME/ihsanvision-artifacts/patchcore_serving_profile.json"
~~~

The default input is a synthetic, solid-color 256x256 PNG, intended only for systems benchmarking. Optionally add --image /path/to/trusted-sample.png to use a controlled real example. **Never compare run speed based on different image dimensions.** Outputs are exclusive: choose a new name for repeat runs. Do not switch model types in the browser while profiling.

## Interpretation

- Client round trip includes HTTP transfer, queuing, processing, serialization and transfer back to client.
- Model prediction span is reported by the inference backend and may include checkpoint restoration on cache misses.
- Cache counts distinguish server-reported reuse true, false or unknown for both measured and warmup requests.
- Nearest-rank p95 is the sorted observation at ceil(0.95*N). With ten samples this is the maximum, so p95 is unstable at small N.
- Sequential completed requests per second = number of measured completions divided by total measured-loop wall-clock duration. **It is not max QPS, concurrent throughput or scalability.**
- Provenance includes OS/CPU architecture, Python version, input dimensions and a SHA-256 of transmitted PNG bytes. No model weights, source image bytes, overlays or API raw response are saved.

For credible comparisons use the **same** source image, accelerator, hardware, library versions, thermal/power conditions and count. A model switch currently evicts the one-entry checkpoint cache. A warm-only claim requires inspection of all measured reuse flags.

## Optional concurrent admission probe

The companion tool makes **simultaneous** localhost requests to one running API process. Keep its finite workload small enough that you do not inadvertently overload your workstation.

~~~bash
python -m scripts.benchmark_concurrent \
  --url http://127.0.0.1:8003 \
  --model patchcore --concurrency 2 --warmup 2 --requests 20 \
  --output "$HOME/ihsanvision-artifacts/patchcore_concurrent_probe.json"
~~~

A two-thread client submits at most two inference HTTP calls at once, but the current server intentionally serializes its Anomalib engine. The report captures **all** submitted responses, including 429/503 and transport failures. p50/p95 metrics only cover successful responses, while `response_status_counts` and `results_failed_or_rejected` show the full outcome. If any request fails, the CLI records the result then exits nonzero. It does not claim sustainable throughput, and a high success rate for only one 20-request run does not establish production scaling.

Results are specific to the selected model, sample, Python and hardware environment. A rate limit of 120 accepted requests per minute is enabled by default per worker; repeated experiments may hit this boundary. A valid shared API key is sent through IHSANVISION_CLIENT_API_KEY when configured, not written in a report. Do not run multiple benchmarks concurrently, or use these tools against an external host.

## Pairwise audit of saved Mac profiles

Two 2026-10-16 saved Mac profiles have been reported for the **same 900×900 transmitted PNG**, with 3 warmups and 20 measured cached requests per model: [recorded evidence](../reports/2026-10-16-macos-paired-serving-profile.md). The reported HTTP round-trip P50/P95 values are **388.868 / 766.458 ms for PaDiM** versus **493.050 / 514.819 ms for PatchCore**.

The read-only script below checks the original JSON files before generating a new Markdown report (and does not load checkpoints or send HTTP requests):

~~~bash
python -m scripts.compare_serving_profiles \
  --padim "$HOME/ihsanvision-artifacts/padim_serving_profile_2026-10-16.json" \
  --patchcore "$HOME/ihsanvision-artifacts/patchcore_serving_profile_2026-10-16.json" \
  --output "$HOME/ihsanvision-artifacts/paired_serving_profile_2026-10-16.md"
~~~

The verifier requires matching transmitted PNG SHA-256, dimensions, workload, warmup count, OS/architecture/Python versions, both successful run statuses, and **all measured predictions reporting cache reuse**. It refuses existing output files; choose a new name to repeat. Until the user runs it and shares the output, the consolidated report is grounded in the pasted Terminal result rather than independently regenerated from the raw JSON by GitHub.

## Acceptance gates

The code has offline mocked regression tests. Real-checkpoint Mac **sequential profiling has now been observed** for both models on the same user-supplied image. Paired JSON content-level audit, full cold-start measurement, memory measurement, controlled repeatability, concurrent load and error admission tests are **still pending**. A completed script execution is not proof of production capacity.
