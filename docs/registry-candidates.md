# Offline model-registry candidates (classic MVTec AD)

This tool does **not** train, load, approve, or serve models. It reads previously
completed, locally trusted classic MVTec AD run artifacts and verifies each
selected category has BOTH PaDiM and PatchCore reports, splits and nonempty
checkpoint files. It hashes checkpoint bytes (SHA-256), and produces a private
**candidate-only** JSON document for human review.

Run on your own machine only when you have authorized local artifacts:

```bash
python -m scripts.build_registry_candidates \
  --runs-dir /absolute/private/classic-runs \
  --categories bottle cable \
  --output /absolute/private/candidate-models.json
```

Output is exclusive-create, mode 0600 on POSIX, and cannot be inside the runs
directory. Never commit or share it: it contains local checkpoint paths.
Symlinked checkpoints and changed files during hashing are refused.
Reports/splits are structurally validated, not cryptographically attested.
The candidate JSON is **NOT** the serving registry schema and is deliberately
**NOT** an operator-approved SHA allowlist. An operator must separately inspect
scientific metrics, provenance, licensing, thresholds, checkpoint safety, and
real local inference before manually promoting an approved version under the
existing private registry and approval mechanisms.

No dataset is downloaded; no training or model performance is claimed.
