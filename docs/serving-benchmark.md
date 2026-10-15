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

## Acceptance gates

The code has offline mocked regression tests. Real-checkpoint Mac profiling, full cold-start measurement, memory measurement, concurrent load and error admission tests are **still pending**. A completed script execution is not proof of production capacity.
