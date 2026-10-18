# M6/M8/M9 — Real localhost stack acceptance (no training)

A consolidated acceptance runner checks the **already running** IhsanVision
service without doing any training or checkpoint deserialization itself.

It checks liveness, /ready configuration integrity, actual PaDiM and PatchCore
Bottle model discovery, single/batch/history dashboard HTTP routes, and the
existing four **real** checkpoint-backed predictions (normal/defect samples
through both models with validated heatmap PNGs). These checks cannot be run
in lightweight GitHub Actions: the actual model weights live privately on the
operator machine.

## On your Mac (one command after starting the server)

Start `python -m scripts.run_local_demo --no-browser` with both trusted Bottle
weights installed, or start the local container described in
[container-deployment.md](container-deployment.md).

Set `IHSANVISION_CLIENT_API_KEY` privately to your server key if using
authenticated inference; never commit or paste it into issue text.

```bash
python -m scripts.local_stack_acceptance \
  --url http://127.0.0.1:8003 \
  --normal /absolute/private/normal_bottle.png \
  --defective /absolute/private/defective_bottle.png \
  --output "$HOME/ihsanvision-artifacts/local-stack-acceptance.json" \
  --require-auth
```

Remove `--require-auth` only for the unkeyed **localhost developer** setup.
The authenticated mode checks an unauthenticated upload is rejected with 401
and that the client has its own key for successful inference.

## Optional private operator workflow acceptance

If you enabled **both** a private `IHSANVISION_HISTORY_DB` and a strong
`IHSANVISION_API_KEY` when starting the server, you may additionally run the
local-stack test with **`--require-auth --operator-workflow`**. Keep the key in
the private client environment variable, never in a command line argument.
This mode is **explicit opt-in** because the four normal/defect × model
acceptance requests **also create four private inspection rows** when history
is enabled. It collects their IDs only in process. It then creates a fifth,
randomly identified test inspection, marks it `confirmed_normal`, verifies
private history/checkpoint SHA/lot groups/CSV/telemetry and **cleans all five
specific test records**, never pre-existing operator rows. If the
network or deletion fails, the command refuses to pass and reports the
synthetic lot identifier for manual cleanup. Use a disposable private
inspection DB for initial acceptance. Existing records are not deleted.

```bash
# Configure server-side IHSANVISION_HISTORY_DB and IHSANVISION_API_KEY privately.
# Configure client-side IHSANVISION_CLIENT_API_KEY with the same private key.
python -m scripts.local_stack_acceptance \
  --url http://127.0.0.1:8003 \
  --normal /absolute/private/normal_bottle.png \
  --defective /absolute/private/defective_bottle.png \
  --output "$HOME/ihsanvision-artifacts/operator-stack-acceptance.json" \
  --require-auth --operator-workflow
```

This opt-in check performs **real checkpoint-backed predictions** in the
current server, while CI integration tests use a fake predictor and a private
test SQLite DB. Browser clicks and successful persistence *after a real
service restart* remain **separate manual checks**. The new report deliberately
excludes raw input images, operator API key, checkpoint hashes, record IDs and
lot identifiers.

## Docker validation

First successfully run `docker compose -f compose.yaml build` and
`docker compose -f compose.yaml up -d` with your private read-only checkpoint
mount. Run the **same** command with `--url http://127.0.0.1:8080`. A
passing report then demonstrates inference through that port, but you must
separately confirm `docker compose ps`, user 10001, readonly mounts,
container restart, memory and cold-start behavior. A successful lightweight
GitHub CI test cannot replace a real Docker build.

The JSON report is private, created exclusively, POSIX 0600, and contains
the two sample **filenames and SHA-256 hashes**, no image bytes,
base64 heatmaps, weights, absolute checkpoint paths or API keys.
Existing reports are never overwritten. Unexpected predictions exit nonzero
and stay recorded as a failure.

**Not established:** held-out split membership, robustness, calibration,
long-running camera throughput, browser UI interaction, actual Docker
internals, accessibility or production internet security. Manual review
of overlays and image provenance is still required.
