# Opt-in checkpoint approval and version pinning

IhsanVision.AI can run a safer **operator-pinned local inspection mode** so
an accidentally replaced, unreviewed, or wrongly selected model checkpoint
is **not used for inference**. This opt-in mode is separate from the
existing per-inspection SHA-256 traceability.

## Approval policy

The operator is responsible for independently reviewing a real trained
model's validation protocol, test results, threshold, category, dataset
rights, security and checkpoint provenance. After that review, create
a **private** JSON manifest outside Git:

~~~json
{
  "version": 1,
  "approvals": [
    {
      "category": "bottle",
      "model": "padim",
      "checkpoint_sha256": "REPLACE_WITH_EXACT_LOWERCASE_64_CHARACTER_SHA256"
    }
  ]
}
~~~

The placeholder above is NOT valid until replaced with a real 64-character
lowercase hex digest. Inspect your own trusted checkpoint without running
or deserializing it:

~~~bash
shasum -a 256 "/absolute/private/path/to/your/trusted/model.ckpt"
~~~

**Do not interpret the command or auto-generated digest as model approval.**
Review the actual dataset, experiment report and test evidence first.
The manifest format deliberately does not allow arbitrary checkpoint
paths or remote URLs. The allowlist identifies expected **bytes for a
specific category and model family**. Only categories with trusted
checkpoints configured by the server and appearing in this manifest
are offered through `/models`.

## Enable separately from private inspection history

From a local checkout with the normal runtime and privately stored
checkpoints, set both environment variables before launching:

~~~bash
export IHSANVISION_APPROVED_MODELS="/absolute/private/path/approved-models.json"
# Use a real, private, strong secret instead of this placeholder:
export IHSANVISION_API_KEY="REPLACE_WITH_A_PRIVATE_SECRET_LONGER_THAN_16_CHARACTERS"
python -m scripts.run_local_demo
~~~

The startup command above is **illustrative**, not a request to paste a
literal placeholder as a credential. Approval mode requires a configured
`IHSANVISION_API_KEY` containing **at least 16 characters**. This is a
single-operator local API key, not enterprise identity and not suitable for
unprotected public deployment. Do not commit or share a manifest, secret,
checkpoint or training image. Keep the service on localhost.

For additional registered categories, the existing private
`IHSANVISION_MODEL_REGISTRY` supplies checkpoint paths. The approval
manifest pins the category/model pair to the reviewed file digest.
Multiple approved pairs are permitted, including two different model
families for the same category. A service restart is required after
modifying the approval manifest; the manifest is intentionally loaded
once at startup so permissions do not change in the middle of a request.

## Behavior and failure handling

- **Default/legacy mode**: when `IHSANVISION_APPROVED_MODELS` is not
  provided, the existing local demo works as before. No model allowlist
  is enforced and no new hashing overhead is incurred unless private
  inspection history is enabled.
- **Pinned mode**: only configured category/model pairs appearing in
  the approval manifest are advertised through `/models`. An attempt
  to request an unapproved pair returns HTTP 503.
- **Exact content check**: immediately before inference under the
  single-model process semaphore, the service computes/reuses the
  SHA-256 identity of the selected trusted local checkpoint and compares
  it with the operator's approval pin. Mismatch returns **HTTP 503**
  with a generic message. The model is **not invoked**, and no successful
  history record is created for that request.
- **Change during inference**: checkpoint metadata is verified again
  after the prediction. A changed snapshot results in a failure instead
  of publishing an output attributed to an old checkpoint identity.
- **Traceability**: successful pinned predictions return
  `checkpoint_sha256` even when private history is disabled. When history
  is separately enabled, the digest is also stored in private SQLite
  metadata. Do not confuse a valid SHA with approval of source data or
  demonstrated model performance.
- **Explicitly empty approval setting**: an environment variable that is set
  but blank is treated as a startup error, **not** a request to disable
  checkpoint approval. Unset the variable entirely to return to legacy mode.
- **Invalid manifest**: missing/relative/symlink file, unknown category,
  duplicate approval, non-lowercase SHA-256, wrong schema and oversized
  manifest **prevent app startup** instead of silently disabling the policy.
- `/models` discovery checks the configured pair and approval entry,
  **not the live checkpoint bytes**. A stale approval can still be
  listed until a prediction attempts to use it; the inference itself
  is blocked with HTTP 503.

Approval mode does not implement a signed artifact chain, immutable
filesystem storage, external identity provider, multiuser RBAC, policy
governance, model evaluation approval workflow, public HTTPS security
or production change-control certification. Its SHA cache assumes a
cooperating single-host operator; timestamps and inode checks are **not
cryptographic protection against an adversary who can edit both the
checkpoint and filesystem metadata**. A real protected deployment needs
signed approvals, immutable/safely mounted checkpoint artifacts, independent
operator roles and audited promotion/rollback.

Offline regressions cover accidental checkpoint replacement, incorrect
hashes and credentials, category isolation, history integration and the
legacy unpinned path. Real-checkpoint Mac end-to-end acceptance remains
outstanding.
