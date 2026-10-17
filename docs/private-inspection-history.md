# Private persistent inspection history (local operator opt-in)

## What it does

IhsanVision can now persist **successful** inspected-image prediction summaries to a
private SQLite database across server restarts. This is an opt-in, **single-operator
localhost** feature for internal industrial inspection workflows, **not** a production
audit ledger or a multi-tenant SaaS database.

Each row contains an opaque random inspection ID, UTC timestamp, configured
product category/model, raw uncalibrated anomaly score, saved-threshold
normal/anomaly/unknown AI decision, prediction duration and model-cache reuse.
A human operator may independently mark `confirmed_normal`,
`confirmed_anomaly`, or `needs_followup` after reviewing the inspected
item; a separate `reviewed_at` timestamp is recorded. The original model
prediction and score remain unchanged. Clearing a review returns it to null.
No raw image bytes, heatmaps, source filenames, SHA-256 fingerprints,
checkpoint files/paths, browser credentials or API keys are stored.

By default **history is OFF**; ordinary inference and batch inspection remain
fully functional and no SQLite file is created.

## Secure local enablement

Use a checkout with this feature, with Anomalib runtime dependencies and
trusted checkpoints installed. Choose a **private local directory outside Git**
(e.g. `~/ihsanvision-artifacts`; it must already exist) for the database.

Configure in your local Terminal before starting Uvicorn / the one-command
launcher:

~~~bash
export IHSANVISION_HISTORY_DB="$HOME/ihsanvision-artifacts/private-inspections.sqlite3"
# Supply a secret containing at least 16 characters. Keep the secret private:
export IHSANVISION_API_KEY="REPLACE_WITH_YOUR_OWN_LONG_PRIVATE_SECRET"
python -m scripts.run_local_demo
~~~

The example key is a placeholder **not suitable for real use**. Generate your
own strong random key using a trusted password manager or `secrets.token_urlsafe(32)`
and retain it securely outside the repo. **Never commit secrets or share them
in screenshots, Terminal logs or chats.** When enabled, the same key is needed
to submit predictions and retrieve/delete private history. All requests must
stay on `http://127.0.0.1:8003/`; even with this key, the local API is **not
safe to expose publicly** without TLS, proxy upload controls and independent
identity/authorization. Note: changing the key between runs prevents access
until the new key is entered, but existing data remains on disk.

The server refuses to enable history without a nonempty API key at least
16 characters long; it refuses relative DB paths, symlinks, unsafe nonregular
files and existing DB files readable by other users. Newly created DB files
have owner-only `0600` permissions on macOS/Linux. History storage startup
is intentionally explicit and does not overwrite existing data.

## Operator workflows

Open **http://127.0.0.1:8003/history** while the private localhost service
runs, enter your operator key and click **Load private history**. You can review
25 recent records at a time, assign or clear a human review verdict, and
permanently delete an individual metadata record with confirmation. The key remains only in the
current browser tab's memory; the page does **not** use cookies, localStorage,
URLs or embedded API responses to persist it. Record values are rendered
with `textContent`, not HTML injection.

The same operations are available as HTTP:

- `GET /inspections?limit=50&offset=0`, with a valid
  `X-IhsanVision-Key` header, returns up to 100 records per page plus
  `total`, `limit`, `offset`.
- `PATCH /inspections/{inspection_id}/review` with the same key and
  JSON `{"verdict":"confirmed_normal"}` marks the human assessment.
  Allowed values: `confirmed_normal`, `confirmed_anomaly`,
  `needs_followup`, or `null` (clear review). Extraneous request fields
  are refused. This update never modifies the saved AI result.
- `DELETE /inspections/{inspection_id}` with the same key deletes **one**
  record and returns `{"deleted": true, "inspection_id": "..."}`.
- `POST /predict` returns `inspection_id` only when a successful inference
  was recorded. Disabled history returns `null`; `GET/DELETE` returns 503
  while disabled.

Deleting a record is irreversible; make a **private** database backup first
if it is needed. Errors, model exceptions and invalid inference outputs do
**not** create successful records. The client should not interpret missing
threshold labels as normal; an unknown decision stays `null`.

## Limitations and next platform milestones

- This local database is not tamper-proof, encrypted-at-rest beyond whatever
  host disk encryption provides, or a regulatory-compliant electronic record.
- The shared API key is single-operator authentication, **not** account-aware
  authorization, SSO, per-site tenancy or role-based access.
- Human review verdicts are supported, but per-image records do not yet
  have operator identity, lot/run IDs, full checkpoint provenance/versions,
  failure-event audit trails, retention policies, scheduled backups or
  general schema migration governance. This update migrates existing PR #28
  private SQLite databases non-destructively by adding nullable review columns.
- SQLite in-process local operation is not a replacement for a managed
  PostgreSQL multiworker service with encrypted storage and tenant RLS.
- Full Mac validation requires restarting a real checkpoint-backed app with
  explicit opt-in and verifying a normal/anomalous upload followed by history
  review **without** leaking images, paths or secrets.
- Other MVTec AD categories and AD2 experiments are not automatically
  available; only operator-verified checkpoints should appear in the UI.
