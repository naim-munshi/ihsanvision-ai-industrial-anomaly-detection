# M8 — Local inference security baseline

## Design and limitations

The service offers **optional** server-configured API-key protection for POST /predict, a process-global sliding-window rate cap, finite inference admission, and short opaque request IDs in HTTP response headers and sanitized server logs. These are first-stage defense-in-depth controls, **not sufficient for public internet hosting**. The default no-key mode exists solely for backwards-compatible localhost development.

The key is a shared application secret, not account authorization or tenant separation. With multiple Uvicorn workers, each worker has separate rate/admission budgets. A per-process counter is not a distributed limiter. An additional pure-ASGI pre-parser guard now checks the shared API key and bounds the total /predict HTTP request body **before** FastAPI parses multipart data, including uploads without Content-Length. This guard now also uses a process-local finite pre-body admission semaphore (default 2 in-flight requests, configurable 1–100) with a 1-second timeout and HTTP 503/Retry-After on saturation. This bounds aggregate multipart buffering per process. It complements (rather than replaces) proxy-level request limits and gateway authentication. Optional [local operator accounts](operator-accounts.md) now add named roles, revocable sessions and browser CSRF checks. Distributed inference rate limits, proxy TLS and high-load validation remain pending. A real internet service also needs HTTPS, managed secrets, origin/CSRF review, monitoring, dependency scanning, container isolation, incident response and a user/data retention policy.

## Configure localhost optional authentication

Generate a random secret in a suitable secret manager or your own terminal. Do not paste it into the chat or commit it. In the **server** shell, use environment settings supplied privately to Uvicorn:

~~~bash
export IHSANVISION_API_KEY="use-a-private-long-random-key-from-a-secret-store"
export IHSANVISION_RATE_LIMIT_PER_MINUTE=120
export IHSANVISION_MAX_PENDING=2
export IHSANVISION_QUEUE_TIMEOUT_SECONDS=1
~~~

The demo UI accepts the key in a password-type field and sends it via the **X-IhsanVision-Key HTTP header** on inference only. It never places the value in the URL or browser storage. Avoid remote HTTP connections: for non-local deployments, browsers must use HTTPS.

Local CLI smoke, acceptance and profiler scripts optionally read the separate client-side environment variable **IHSANVISION_CLIENT_API_KEY**. Assign its value privately from your secret manager or, if running both server/client in the same secure local shell:

~~~bash
export IHSANVISION_CLIENT_API_KEY="$IHSANVISION_API_KEY"
~~~

Never include a key in a command-line URL, test fixture, report, screenshot, public CI variables or Git commit.

## Admission behavior

| Condition | Endpoint behavior |
| --- | --- |
| Correct configured header / no key required in local mode | Continue normal input validation |
| Missing/incorrect key when server key set | HTTP 401; never invokes the model |
| More than configured requests in 60 seconds | HTTP 429 with Retry-After |
| More than permitted concurrent POST uploads before multipart parsing | HTTP 503 with Retry-After |
| Inference admission unavailable beyond configured wait | HTTP 503 with Retry-After |
| HTTP multipart body over 10 MiB + 64 KiB (before parsing), image over 10 MiB (route), or >25 million decoded pixels | HTTP 413 |
| Every normal HTTP response | Opaque X-Request-ID header; no client-chosen ID trusted |

The rate limit defaults to **120 accepted inference requests per minute per process**, adjustable to 1–10000. The admission limit defaults to **2 concurrent admitted requests** with a **1-second** deadline. The existing one-at-a-time Anomalib inference semaphore still protects the model instance. Admission permits a small amount of waiting; no claim about gateway-scale queue-bounding is made.

A structured audit line logs only generated request ID, HTTP method, endpoint **path without query**, status and elapsed time. It does not deliberately record request body, filename, key, raw score or base64 overlay. Nevertheless, upstream proxy logs, Python exception logs or third-party instrumentation can contain sensitive data and need separate privacy/security review.

## Remaining acceptance gates

- Automated offline 401/429/audit/privacy/configuration tests are included; real concurrency/load tests remain pending.
- Pre-parser key and bounded body checks are implemented with offline regression tests; real reverse-proxy TLS, distributed admission and gateway authentication/limits remain pending.
- Secret rotation, per-user quotas, distributed inference limiter, incident monitoring and deployment privacy policy remain pending.
- Do not change Uvicorn from 127.0.0.1 to 0.0.0.0 based on this PR alone.
