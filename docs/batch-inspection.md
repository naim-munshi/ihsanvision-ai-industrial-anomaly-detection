# Local multi-image industrial inspection workflow

The `scripts.inspect_batch` command processes a **finite folder of inspection
images through the real localhost FastAPI API**, using a single trained
category/model pair. Unlike the benchmark profiler, this workflow produces an
individual defect/normal outcome for **each image** and two structured operator
reports (JSON and spreadsheet-compatible CSV). It is not a model accuracy test
because the input images have no independently verified ground-truth labels.

## Local usage

First run the current local IhsanVision API with your trusted checkpoints,
for example `python -m scripts.run_local_demo --no-browser` on port 8003.
Then, in another terminal:

~~~bash
python -m scripts.inspect_batch \
  --url http://127.0.0.1:8003 \
  --directory "$HOME/Downloads/bottle-ui-test" \
  --category bottle \
  --model padim \
  --max-images 100 \
  --output-prefix "$HOME/ihsanvision-artifacts/bottle-inspection"
~~~

This demonstrates how the user-owned `bottle-ui-test` folder can be inspected,
but it is **not a claim that the script has run against the user's Mac**.
For non-bottle products choose a category/model pair whose trusted checkpoint
is [registered server-side](multicategory-model-registry.md). The endpoint
must not be exposed publicly. Set `IHSANVISION_CLIENT_API_KEY` privately
if server-side API-key protection is enabled; never put the key on the command
line, in reports or screenshots.

The outputs are:
- `bottle-inspection.json` — full local per-image inspection status, SHA-256 of
  source image, raw score, predicted anomaly, inference latency, cache reuse,
  overlay validity and bounded error codes, plus summary counts.
- `bottle-inspection.csv` — Excel-compatible tabular outcomes with suspicious
  formula-like user-supplied filenames neutralized.

Source images, base64 overlays, model weights, raw HTTP payloads, stack traces
and credentials are **never written** into the reports. The output still
contains filenames, scores and file hashes, which might be sensitive. Keep it
in your private artifact storage. The tool refuses to overwrite an existing
JSON/CSV pair. Give a different `--output-prefix` for each inspection lot.

## Operational controls

- Maximum 100 images per run by default, never more than 200; top-level files
  only (no recursive discovery). Supports PNG/JPEG/WEBP/BMP.
- Input symlinks are refused. Each source image is bounded to 10 MiB and
  25 million decoded pixels, and must remain within 10 MiB after PNG encoding.
- Strict local-only URL validation rejects remote endpoints.
- Per-image errors are accounted for, and do not silently count as normal
  products. HTTP 429/503 rejections appear as error status rows. The command
  exits nonzero when any image fails.
- The script checks model identity, category identity for category-specific
  inference, valid raw score, boolean saved operating-threshold decision and
  full-size PNG localization. Invalid responses become errors, **not normal**.
- Flagged anomalies are **operational inspection outcomes**, not process
  failures. They do not automatically block exporting the report; the next
  action is operator review and appropriate quality control.

## Limits and next engineering milestones

This is a **local CLI workflow**, not a networked factory management system.
It does not store user accounts, inspection history in PostgreSQL, image
annotations, work orders, model monitoring or training lifecycle. It does not
establish correctness without calibrated thresholds and independent ground
truth. Operators must review defect heatmaps before manufacturing decisions.

Next steps toward the full vision: web batch-upload/viewer with bounded
background jobs, per-lot history and review, dataset/model-version provenance,
pixel-mask evaluation, audit storage, controlled deployment and security
integration. The current checkpoint set has real evaluation evidence **only
for MVTec AD bottle**. Additional categories require training and licensing
review before deployment.
