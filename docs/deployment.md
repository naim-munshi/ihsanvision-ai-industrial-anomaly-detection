# Service Operation, API and Security

## Current status

The FastAPI service and responsive browser dashboard are implemented. Automated API tests pass with simulated predictions, and standalone **real** PaDiM checkpoint inference has succeeded on Kaggle. **Real Mac HTTP → FastAPI → checkpoint inference has now passed** using a synthetic PNG. Real browser upload interaction with actual normal/defect images is still pending.

The current server offloads synchronous Anomalib execution to a worker thread, so the event loop can answer health requests. A semaphore allows **one inference at a time per worker process**. A **new experimental lazy in-memory cache** loads the operator's trusted model checkpoint on the first request and attempts to reuse that same Engine/model instance for subsequent requests in the same process.

**Caching is not yet verified with a real checkpoint.** The previous cold-inference route is still available by setting `IHSANVISION_MODEL_CACHE=0` **before starting Uvicorn**. Changing the environment variable in another shell does not update an existing server process. No bounded request queue, authentication, rate limiting or scale-out capacity exists.

Only bind the service to `127.0.0.1` for initial validation. Do **not** expose this prototype directly to the public internet.

## Local startup

Requirements: Python 3.11/3.12, installed dependencies from `requirements.txt` and a checkpoint **you trust** from the matching model.

~~~bash
export IHSANVISION_CHECKPOINT="/absolute/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL="padim"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

UI: `http://127.0.0.1:8000/`. Swagger: `http://127.0.0.1:8000/docs`. If port 8000 is already occupied (common on macOS), use `--port 8001` instead.

### Local verification commands

Check dependencies and the copied checkpoint/report JSONs without deserializing the model:

~~~bash
python -m scripts.local_preflight
~~~

With the server started and its port known, use **another terminal**:

~~~bash
python -m scripts.local_api_smoke --url http://127.0.0.1:8001
~~~

Or pass an actual product image:

~~~bash
python -m scripts.local_api_smoke --url http://127.0.0.1:8001 --image /path/to/bottle.png
~~~

This runs an actual HTTP integration test and validates score/label/overlay fields. The user already observed `api-smoke-passed` on macOS with PaDiM and a ~66.69-second cold result (see [run evidence](../reports/2026-10-14-macos-api-smoke.md)).

For caching validation, use **two calls without restarting the same server**. The new smoke script prints `model_reused`: it should be `false` on the first call and `true` on the second. Measure latency and numerical consistency before claiming any improvement. If errors occur, restart with `IHSANVISION_MODEL_CACHE=0` as above. When no image is provided, it sends a synthetic PNG: its score is **not** meaningful for defect detection. Neither script provides authentication or load testing.

## API contract

| Method | Endpoint | Input | Output |
| --- | --- | --- | --- |
| GET | `/` | None | Static inspection UI |
| GET | `/health` | None | JSON `status` and `model_configured` |
| POST | `/predict` | Multipart field `file` | JSON score, optional label, time and PNG overlay |

Example from a **local trusted environment**:

~~~bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@/absolute/path/to/bottle.png"
~~~

The response includes fields similar to:

~~~json
{
  "model": "padim",
  "anomaly_score": 0.8633,
  "predicted_anomaly": true,
  "latency_ms": 1396.49,
  "heatmap_png_base64": "<base64 omitted>",
  "notes": "Score is NOT a probability; heatmap is relative visualization."
}
~~~

**Example only:** numeric values illustrate the previously observed Kaggle smoke result; they are not a live service invocation. Do not interpret `anomaly_score` as a calibrated probability, `latency_ms` as warm-service throughput, or the overlay as a binary defect mask.

### Upload validation

The service validates common image formats (PNG, JPEG, WEBP, BMP) by actual content, limits uploads to 10 MiB and 25 million pixels, and writes a temporary file for prediction. Unsupported/oversized payloads are rejected. Errors are surfaced without a model traceback; operators should inspect internal server logs separately.

## Security posture

**Not suitable for public deployment as is**:

- No API authentication, account management, tenant isolation or rate limiting.
- Experimental per-process model cache without real-model compatibility verification, model registry or safe hot-reload lifecycle.
- Only per-process serialized inference; no bounded queue, streaming limits, monitoring dashboards or service-level objectives.
- No structured audit logs or user privacy retention policy.
- No hosted deployment smoke test or disaster recovery procedure.

**Never accept arbitrary model/checkpoint files from clients.** A serialized Lightning/PyTorch checkpoint can run code during deserialization. The server must load only a model artifact from a trusted location selected by the operator; it must not be configured by an untrusted HTTP request.

For a later public service implement authenticated authorization, reverse proxy/TLS, bounded request size at gateway, rate limiting, process isolation, trusted model registry, warm model loading, observability and log scrubbing. Evaluate licensing of any training data and pretrained weights before commercial release.

## Deployment acceptance gate

Before a public endpoint is announced:

1. Re-verify two successive requests with the **real Kaggle-trained checkpoint** after enabling the experimental cache and compare with the proven cold result.
2. Confirm proper health, missing-checkpoint errors, supported and rejected file inputs.
3. Measure cold/warm p50/p95 latency and memory using a repeatable workload.
4. Introduce authentication and rate limiting, plus a proper checkpoint/model version lifecycle.
5. Establish privacy/logging policy, security review, TLS and deployment rollback.
6. Verify UI functions on desktop and mobile; record evidence in an issue or report.

Further work is tracked in [roadmap.md](roadmap.md).
