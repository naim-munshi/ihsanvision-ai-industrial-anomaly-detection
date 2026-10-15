# Service Operation, API and Security

## Current status

The FastAPI service and responsive browser dashboard are implemented. Automated API tests pass with simulated predictions, and standalone **real** PaDiM checkpoint inference has succeeded on Kaggle. **Real Mac HTTP → FastAPI → checkpoint inference has now passed** using a synthetic PNG. Real-browser interaction with a normal and a visibly defective bottle image has now been observed in [the UI inspection report](../reports/2026-10-14-browser-bottle-inspection.md); separate validation of error states, mobile behavior and sample split provenance is still pending.

The current server offloads synchronous Anomalib execution to a worker thread, so the event loop can answer health requests. A semaphore allows **one inference at a time per worker process**. A **new experimental lazy in-memory cache** loads the operator's trusted model checkpoint on the first request and attempts to reuse that same Engine/model instance for subsequent requests in the same process.

**PaDiM caching has now been verified with the user's real Mac checkpoint** (see the recorded first/reused pair), although PatchCore, multi-worker reuse and load-test behavior are still unverified. The previous cold-inference route is still available by setting `IHSANVISION_MODEL_CACHE=0` **before starting Uvicorn**. Changing the environment variable in another shell does not update an existing server process. A local-only security baseline now adds optional shared-key authentication, a process-global 60-second rate cap, a finite admission budget, and scrubbed request ID audit logs. These controls do not establish internet-safe deployment, account authorization, distributed quotas or scale-out capacity.

Only bind the service to `127.0.0.1` for initial validation. Do **not** expose this prototype directly to the public internet. A [localhost Docker Compose configuration](container-deployment.md) now exists with external readonly model mounts, a nonroot runtime and a required shared key; **real image build and container inference remain unverified**.

## Local startup

For optional API-key authentication, configurable per-process rate limits/admission, client header handling and the remaining cloud-security gaps, see [the security baseline](security-baseline.md). The browser sends a key only in an HTTP header, and the localhost CLI tools can read IHSANVISION_CLIENT_API_KEY. Public exposure requires gateway TLS, pre-multipart authentication/body limits and additional infrastructure.

Requirements: Python 3.11/3.12, installed dependencies from `requirements.txt` and a checkpoint **you trust** from the matching model.

~~~bash
export IHSANVISION_CHECKPOINT="/absolute/path/to/trusted/model.ckpt"
export IHSANVISION_MODEL="padim"
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8000
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

This runs an actual HTTP integration test and validates score/label/overlay fields. The user observed `api-smoke-passed` on macOS with PaDiM and a ~66.69-second earlier download-affected cold result. With the updated factory and cache, a later two-request run reported 721.99 ms initial inference and 123.54 ms reuse with matching score and overlay (see [run evidence](../reports/2026-10-14-macos-api-smoke.md)).

For caching validation, use **two calls without restarting the same server**. The new smoke script prints `model_reused`: it should be `false` on the first call and `true` on the second. Measure latency and numerical consistency before claiming any improvement. If errors occur, restart with `IHSANVISION_MODEL_CACHE=0` as above. When no image is provided, it sends a synthetic PNG: its score is **not** meaningful for defect detection. Neither script provides authentication or load testing.

### Repeatable local inference latency sample

With Uvicorn running on port 8002, use a separate Terminal:

~~~bash
python -m scripts.benchmark_local_api --url http://127.0.0.1:8002 --runs 10 --output results/local-padim-serving-benchmark.json
~~~

The script measures Anomalib's reported inference span per request, checks that identical test inputs produce consistent scores, and reports the warm-sample p50 and empirical nearest-rank p95. It only classifies samples as warm if `model_reused=true`. The first request is **not necessarily cold**, because the server may already have the checkpoint in memory. These numbers exclude total HTTP exchange and UI rendering; no throughput or concurrency claims should be made.

### macOS networking and port troubleshooting

When the port is occupied (`[Errno 48] address already in use`), use another available local port such as `8002` and run the smoke test from a **separate Terminal while Uvicorn remains running**. A `Ctrl+C` in the server Terminal stops Uvicorn and the smoke test then reports `Connection refused` (`Errno 61`). You can inspect listeners using `lsof -nP -iTCP:8001 -sTCP:LISTEN`; do not terminate an unknown process blindly.

A previous Mac run failed to retrieve the ResNet18 backbone from Hugging Face because TLS certificate verification failed. Since checkpoint-backed inference restores the backbone weights, the factory now sets `pre_trained=False` for both the cold and cached inference paths, without changing the training initialization. This removes an unnecessary download during model construction. **The subsequent Mac run succeeded on both initial and reused requests without Hugging Face TLS failures.** Do not disable certificate verification to work around a network's HTTPS interception.

## API contract

| Method | Endpoint | Input | Output |
| --- | --- | --- | --- |
| GET | `/` | None | Static inspection UI |
| GET | `/health` | None | JSON `status` and `model_configured` |
| POST | `/predict` | Multipart field `file` | Typed `PredictionResponse` JSON: raw finite score, model, optional label/latency/PNG overlay/model_reused |

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

The endpoint declares a Pydantic `PredictionResponse` schema in `/openapi.json`. It requires a recognized `model` and finite `anomaly_score`, refuses negative `latency_ms`, and documents `model_reused` (optional for cold/CLI-style output). Invalid model output is a server-side error and must be investigated; no anomaly probability is asserted.

**Example only:** numeric values illustrate the previously observed Kaggle smoke result; they are not a live service invocation. Do not interpret `anomaly_score` as a calibrated probability, `latency_ms` as warm-service throughput, or the overlay as a binary defect mask.

### Upload validation

The service validates common image formats (PNG, JPEG, WEBP, BMP) by actual content, limits uploads to 10 MiB and 25 million pixels, and writes a temporary file whose extension is derived from inspected content, not the original user-supplied filename. Pillow validation executes in a worker thread to avoid blocking the FastAPI event loop. Unsupported/oversized payloads are rejected. Errors are surfaced without a model traceback; operators should inspect internal server logs separately.

## Security posture

**Not suitable for public deployment as is**:

- An optional shared API key, per-process global rate limit, finite admission and audit request IDs are implemented. They are **not enabled by default for localhost**, and do not provide account management, tenant isolation, distributed quotas, external TLS or public gateway security.
- Real-Mac-verified PaDiM per-process cache, but without PatchCore, multi-worker, high-load verification, model registry or safe hot-reload lifecycle.
- Per-process serialized inference and finite application admission exist; the FastAPI multipart parser still processes request bodies before endpoint authentication/admission. No reverse-proxy upload limits, distributed queue, monitoring dashboards or service-level objectives.
- Minimal scrubbed request-ID audit lines exist; no comprehensive user privacy retention policy or SIEM integration has been reviewed.
- No hosted deployment smoke test or disaster recovery procedure.

**Never accept arbitrary model/checkpoint files from clients.** A serialized Lightning/PyTorch checkpoint can run code during deserialization. The server must load only a model artifact from a trusted location selected by the operator; it must not be configured by an untrusted HTTP request.

For a later public service implement authenticated authorization, reverse proxy/TLS, bounded request size at gateway, rate limiting, process isolation, trusted model registry, warm model loading, observability and log scrubbing. Evaluate licensing of any training data and pretrained weights before commercial release.

## Deployment acceptance gate

Before a public endpoint is announced:

1. Collect repeated warm latency samples with the **real Kaggle-trained checkpoint** and compare reproducibly against the first request; the initial two-request cache smoke already passed.
2. Confirm proper health, missing-checkpoint errors, supported and rejected file inputs.
3. Measure cold/warm p50/p95 latency and memory using a repeatable workload.
4. Introduce authentication and rate limiting, plus a proper checkpoint/model version lifecycle.
5. Establish privacy/logging policy, security review, TLS and deployment rollback.
6. The desktop browser's normal/defective inspection path has been observed working. Verify error-state behavior and mobile layout; retain sample provenance and screenshots in an approved portfolio location.

Further work is tracked in [roadmap.md](roadmap.md).
