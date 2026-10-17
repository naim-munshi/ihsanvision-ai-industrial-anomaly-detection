# Local operator metrics (no training)

When a private `IHSANVISION_API_KEY` is configured, a local authenticated
endpoint exposes bounded per-process inference telemetry:

```http
GET /operator-metrics
X-IhsanVision-Key: <your private operator key>
```

Use the normal browser private key field or a localhost HTTP client. Never put
the key into URL parameters, committed configs or screenshots. An unset server
key makes the metrics endpoint unavailable (HTTP 503); an incorrect key
receives 401. Do not expose the service publicly.

The output includes admitted prediction request successes/failures,
p50/p95 post-admission elapsed milliseconds (last 256 completed attempts),
sample count and process uptime. Request bodies, filenames, images, anomaly
scores, lot identifiers, checkpoints, raw SHA-256 identities and API keys
are **never stored by this telemetry object**.

These numbers include image validation and serialized inference time but
**exclude request body upload and pre-admission rejections**. They are not
full HTTP round-trip latency, service throughput, GPU utilization, multiworker
metrics or a certified SLO. The counters reset on app restart and require
external observability infrastructure for durable alerts and dashboards.

Use the separately documented [real local stack acceptance](operational-acceptance.md)
and [sequential serving profiler](serving-benchmark.md) to validate actual
checkpoint-backed outputs and defined performance workloads.
