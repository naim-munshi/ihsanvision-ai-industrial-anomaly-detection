# macOS 10-request PaDiM inference latency sample

**Date:** 2026-10-14 JST. **Evidence:** User-provided Mac Terminal transcript. This report records the observed output; it was not independently rerun in GitHub Actions.

## Environment and scope

- API base: `http://127.0.0.1:8002`, served from macOS localhost using the saved PaDiM bottle checkpoint.
- The benchmark was invoked from a virtual environment with `python -m scripts.benchmark_local_api --url http://127.0.0.1:8002 --runs 10 --output ~/ihsanvision-artifacts/padim-bottle-seed42/serving-benchmark.json`.
- The synthetic 256×256 RGB gray PNG has **no known real defect label**.
- `latency_ms_including_load` is the **model-reported prediction span** from the FastAPI service, not elapsed client network time, user-perceived UI time, or requests per second.
- All ten response objects claimed `model_reused=true`, including the first measurement. The server was already warm before this benchmark started. The server appeared to have been started before the user's subsequent `git pull`, so this sample **does not validate new code pulled while Uvicorn was running**.

## Raw observed timings

| Request | Model-reported inference time (ms) | Model reused |
| ---: | ---: | --- |
| 1 | 1244.62 | true |
| 2 | 148.92 | true |
| 3 | 118.86 | true |
| 4 | 139.60 | true |
| 5 | 139.00 | true |
| 6 | 120.61 | true |
| 7 | 132.55 | true |
| 8 | 118.91 | true |
| 9 | 164.19 | true |
| 10 | 129.15 | true |

The script reported:

```json
{
  "status": "api-benchmark-passed",
  "model": "padim",
  "requests": 10,
  "first_request_ms": 1244.62,
  "first_request_model_reused": true,
  "warm_reused_samples": 10,
  "warm_inference_p50_ms": 135.78,
  "warm_inference_p95_ms": 1244.62,
  "consistent_anomaly_score": 0.8242901563644409,
  "predicted_anomaly": true,
  "overlay_png": true
}
```

**Interpretation:** The model returned an identical numeric anomaly score and generated an overlay for each request. All ten API smoke checks passed. The unusually slow first measured call (1244.62 ms) was still marked as reused. With ten samples, the empirical **nearest-rank p95 is the largest value** (`ceil(0.95*10)=10`), so the single long tail appears in p95. This does not justify trimming the outlier or claiming p95 is typically ~140 ms. p50 = 135.78 ms; nine individual requests are between 118.86 and 164.19 ms. This is one short sequential same-image sample, not a realistic workload, cold-start distribution, GPU comparison, or load test.

## Subsequent code update and port issue

In the same user transcript, `git pull --ff-only` fast-forwarded the feature branch from `c0a5f40` to `bc88ffd` and confirmed Python 3.12.4. Attempting another `uvicorn` on port 8002 yielded `[Errno 48] address already in use`; the previous process had already bound that port. Restart the server after updating the working tree to exercise the new image-validation and Pydantic response-schema changes. Do not assume an already-running Uvicorn process has reloaded its Python modules.

## Remaining evidence

1. Restart Uvicorn cleanly (or select a different free port), check the latest OpenAPI response schema and run a local smoke call.
2. From the real MVTec bottle data, test separate normal and defective images in the browser; preserve screenshots/results without redistributing restricted data.
3. Collect a longer measured sample to investigate the observed 1.24-second spike and characterize latency variability.
4. Run PatchCore with verified identical saved partitions before reporting comparisons.

The checkpoint, original images and generated local benchmark JSON are **not** committed to the source repository.
