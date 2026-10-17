# Operator-reviewed model releases and rollback (local/offline)

The application already supports a private checkpoint registry and exact
operator-approved SHA-256 pins. This utility bridges the **unapproved**
candidate inventory to an explicitly reviewed, versioned pair of compatible
configuration files. It **does not** train, deserialize, certify, or activate a
model. A human operator must independently confirm model evaluation, dataset
rights, threshold provenance, actual image QA, and checkpoint trust.

## Prepare a release

Generate a candidate inventory from actual completed locally trusted experiments
using [registry-candidates.md](registry-candidates.md), then independently
review its dataset/report evidence. On your own computer:

```bash
python -m scripts.stage_model_release stage \
  --candidates /absolute/private/candidate-models.json \
  --output-dir /absolute/private/releases/v1 \
  --approve bottle:padim bottle:patchcore \
  --operator-reviewed

python -m scripts.stage_model_release verify \
  --release-dir /absolute/private/releases/v1
```

The operator acknowledgement is **not** proof of a safety audit; it prevents
an accidental automated promotion path. Staging refuses a checksum mismatch,
missing or symlinked checkpoint, duplicate pairs, malformed inventory,
existing release directory or inconsistent registry/approval mappings. It
creates an exclusive local release directory, mode 0700 on POSIX, with
exclusive private files (0600); no secrets, image bytes or weights are copied.
The checks do not make a signed attestation or render arbitrary Lightning
checkpoints safe to deserialize.

## Manual activation

Review the stage output, then set **both** variables to the same verified
release before restarting your private localhost service. Set an operator API
key privately; do not paste it into terminals/screenshots shared publicly.

```bash
export IHSANVISION_MODEL_REGISTRY=/absolute/private/releases/v1/model-registry.json
export IHSANVISION_APPROVED_MODELS=/absolute/private/releases/v1/approved-models.json
# export IHSANVISION_API_KEY from your private secret manager
python -m uvicorn ihsanvision.api:app --host 127.0.0.1 --port 8003
```

A release is not activated by the stage command. Verify `GET /models`, then
real checkpoint-backed `/predict` on normal and anomalous samples before
accepting the version. API-key enforcement and approved checkpoint pinning
prevent unlisted model versions from running; the server snapshots approval
pins on startup. A change to either file requires a controlled restart.

## Manual rollback

Keep the previously verified release untouched. Stop the service. Run
`verify --release-dir /absolute/private/releases/v0` and restore **both**
private environment paths to the previous release v0; then restart and run
health, model and real inference checks again. If checkpoint bytes were
changed, verification refuses the old release: restore independently trusted
artifacts first, never blindly bypass a checksum mismatch.

A real zero-downtime deployment, automatic blue/green switchover,
rollback orchestration, shared identity and public TLS gateway remain
separate infrastructure milestones. Model approval metadata here records
the operator's explicit action, **not** independent scientific verification.
