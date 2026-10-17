# Exact checkpoint identity in private inspection records

IhsanVision's operator-enabled inspection history records which **checkpoint bytes**
were used for each successful inspection. The model name `padim` or
`patchcore` is **not** a sufficient version identifier: a trained checkpoint
may be replaced or re-trained with a different dataset or threshold.

## How it works

With both `IHSANVISION_HISTORY_DB` and `IHSANVISION_API_KEY` configured:

- Immediately before a model prediction, the server streams the selected
  **trusted local** `.ckpt` file through SHA-256. This happens in a worker
  thread and never deserializes the checkpoint for hashing.
- The file's device, inode, size, modification time and change time are
  checked before and after hashing. A matching, unchanged file snapshot
  reuses its cached hash in later predictions so large checkpoint files are
  not read again for every image.
- Once prediction completes, the file metadata is checked again. If it
  changed during inference, the request **fails closed**; no output is
  attributed to the old checkpoint and no successful private inspection
  history is added.
- On a valid prediction, the API returns optional
  `checkpoint_sha256` and the private SQLite record stores that exact
  64-character digest. Older records migrate non-destructively with
  `checkpoint_sha256: null`; their missing historical identity is **not
  guessed or backfilled**.
- The local `/history` operator dashboard displays the first twelve hash
  characters for compact viewing and exposes the complete hash as a tooltip.
  Records remain protected by the existing API key; source images, private
  file paths, checkpoint content and API keys are not added to this ledger.

When history is disabled, predictions follow the existing lightweight
inference path and return `checkpoint_sha256: null`. No database is created
and no checkpoint hashing overhead is introduced.

### Run locally

Configure the existing opt-in history as described in
[private inspection history](private-inspection-history.md), then run the
normal `python -m scripts.run_local_demo` command from an updated checkout.
The existing private checkpoint files remain under operator ownership outside
Git. Open `http://127.0.0.1:8003/history` and compare fingerprints after
inspecting an image using each verified model.

To inspect a checkpoint *without loading it* using your local Terminal:

~~~bash
shasum -a 256 /absolute/private/path/to/trusted/model.ckpt
~~~

The complete digest should equal the API's `checkpoint_sha256` for the
corresponding model. **Do not upload checkpoint files to a public verifier.**

## Operator inspection analytics by version

The authenticated `GET /inspection-model-versions?limit=50&offset=0`
endpoint groups only successfully saved history records by **category,
model family, and exact checkpoint SHA-256**. Optionally use
`&lot_id=LOT-2026-A01` to limit the groups to a production lot.
This avoids mixing observations from differently trained checkpoints
with a reused `padim` / `patchcore` label.

Each group reports counts of AI normal/flagged/unknown, independent human
normal/anomaly/follow-up/unreviewed and the number of known human/AI
disagreements. These are **workflow counts**, not defect sensitivity or
model-accuracy estimates, and raw model scores are not averaged.

The operator-only `/history` dashboard has a **Show model versions** button.
It shows at most the first 100 groups for quick operator review; the
authenticated API offers bounded pagination for further pages. It may
also use the existing lot-filter input. Null SHA-256 is a separate
**Unknown (legacy)** group and cannot be equated to an existing model.

No source images, checkpoint paths or weights are returned through the
aggregation. Operator API key authentication and local-only deployment
limitations continue to apply.

## Limits and next safeguards

SHA-256 identifies **file contents**, not whether the checkpoint was trained
on an approved dataset, securely sourced, scientifically validated or safe
to deserialize. File timestamp/inode caching is a practical per-process
optimisation, **not** an adversarial/tamper-resistant content monitor; a
privileged operator could modify files and manipulate metadata. For
production attestation use a signed model manifest and read-only, immutable
model artifacts with verified chain-of-custody.

This feature does **not** yet validate actual new model training outcomes,
guarantee a consistent GPU/Mac runtime, implement a model promotion/rollback
workflow, or establish an industrial quality SLO. Offline Python/SQLite/FastAPI
regression tests are not a substitute for real Mac checkpoint acceptance.
