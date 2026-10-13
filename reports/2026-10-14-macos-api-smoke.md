# macOS checkpoint-backed FastAPI smoke verification

**Observed:** 2026-10-14 JST (user-provided Terminal output from Mac, not independently rerun by CI).

## Scope

A real Mac venv, downloaded checkpoint artifact, Anomalib 2.5.0, FastAPI/Uvicorn on localhost port 8001, and a generated **synthetic** PNG were used to verify a complete request/response path. This was performed **before** the experimental model-cache optimization was added.

## Environment and saved artifacts

```text
Python 3.12.4 / macOS Darwin
torch 2.14.1
anomalib 2.5.0
numpy 2.0.2
fastapi 0.143.0
uvicorn 0.54.0
```

The `scripts.local_preflight` output shows all five expected files exist outside the repo:

- `model.ckpt` 176,679,475 bytes
- `report.json`
- `splits.json`
- `smoke_report.json`
- `smoke_overlay.png`

The manifest has **209 train / 16 validation / 67 test** samples and **no overlapping paths**. The report includes image_AUROC, image_F1Score, pixel_AUROC and pixel_F1Score. This checks recorded artifact consistency; it does not re-evaluate model accuracy.

## Real HTTP integration result

From the output of `python -m scripts.local_api_smoke --url http://127.0.0.1:8001`:

```json
{
  "api": "http://127.0.0.1:8001",
  "model": "padim",
  "anomaly_score": 0.8242901563644409,
  "predicted_anomaly": true,
  "latency_ms_including_load": 66687.78,
  "overlay_png": true,
  "status": "api-smoke-passed"
}
```

This establishes that /health, the root HTML page, HTTP /predict, real model checkpoint inference, and a PNG overlay all worked in the user environment. A **synthetic** input has no defect-label ground truth, so neither the anomaly label nor score measures true defect detection accuracy.

The **66.7 seconds** refers to one run including model initialization and checkpoint loading. It is not a performance measurement across repeated requests, a p50/p95 benchmark, or evidence of production scalability. Separate visual user interaction with the browser UI has not been demonstrated.

## Subsequent engineering change (awaiting real runtime verification)

`src/serving.py` adds an **experimental lazy persistent model/Engine session**. The FastAPI default now attempts to load a trusted checkpoint only for the first request of a given model/checkpoint version, and reuse weights on later serialized requests in the same process. Offline fake-engine tests verify cache key, reuse and invalidation.

This optimization **has not yet been exercised with the real saved checkpoint** in the user's Mac environment. The previously tested cold-reload behavior remains available:

```bash
export IHSANVISION_MODEL_CACHE=0
```

A new server process is required when changing this setting. Before asserting performance improvements, compare two successive calls to `python -m scripts.local_api_smoke` against the same running Uvicorn process, record `model_reused` and actual timings, and validate the numerical result/overlay.

## Next acceptance gates

1. Test cached repeated requests on the actual Mac checkpoint. Confirm response consistency and timing improvement (if any).
2. Upload a representative actual normal and defect bottle image through the browser.
3. Benchmark PatchCore using the same dataset and evaluation splits.
4. Consider a warm model lifecycle, queued request bounds, observability and security before public deployment.


## Subsequent macOS operational incident (2026-10-14 JST)

A later user-provided server log showed successful `GET /` and `GET /health` responses, followed by one `POST /predict` returning **500** while the default PaDiM backbone attempted to fetch pretrained ResNet18 files from Hugging Face. TLS rejected the network certificate ("certificate is not trusted") through repeated retries. After the model weights eventually downloaded, a subsequent `POST /predict` returned **200**.

The server was then stopped with `Ctrl+C`. Two later smoke commands returned **`Errno 61: Connection refused`**, because no service was running. After updating the working tree, starting another server on port 8001 failed with **`Errno 48: address already in use`**; this indicates a separate listening process existed at that moment, not a model-specific failure.

**Engineering response:** checkpoint-restored inference now constructs PaDiM/PatchCore with `pre_trained=False` and relies on the trusted checkpoint to supply trained backbone weights, while training still constructs models with `pre_trained=True`. Added offline unit checks and CI. This resolves the model-factory dependency on new pretrained downloads by design, **but no Mac runtime run of that change has yet been supplied**. Use a free port (e.g. 8002), leave Uvicorn running in one terminal, and run two smoke calls from another. Never disable TLS validation to bypass intercepted certificates.

This follow-up does **not** establish that model caching has worked; the log does not include two confirmed successful requests against the same newly updated cache-enabled server.
