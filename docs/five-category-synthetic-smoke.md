# Five-category real-checkpoint API smoke without dataset images

**Purpose:** The Mac owner has trained and privately archived five classic MVTec AD categories (bottle, cable, carpet, metal_nut, capsule) with PaDiM and PatchCore. The operator has independently established a ten-entry SHA-256-approved registry and received `config_ready` from the local server. This command exercises **actual model inference** for the selected pairs without downloading or redistributing any MVTec photographs.

The code sends a **generated 256 × 256 grayscale PNG** to every chosen category/model pair through authenticated `POST /predict`, validates model/category routing, finite anomaly scores and a real PNG overlay of the same size. **Synthetic results are not defect-detection accuracy, MVTec sample performance, calibration, validation/test protocol evidence, or browser UI acceptance.** The generated image may be reported as an anomaly. Do not label it "correct" or "incorrect".

## Owner Mac procedure

Keep the existing operator server running at `http://127.0.0.1:8004`, using your private `IHSANVISION_MODEL_REGISTRY`, `IHSANVISION_APPROVED_MODELS`, `IHSANVISION_API_KEY`, and `IHSANVISION_HISTORY_DB` configurations. Do not start a second competing server at the same port.

In a **second terminal**:

~~~bash
cd "$HOME/ihsanvision-ai-industrial-anomaly-detection"
source "$HOME/ihsanvision-artifacts/venv-py312/bin/activate"
git pull --ff-only origin main

# Start with the two Cable checkpoints (no external images required).
python -m scripts.multicategory_smoke \
  --url http://127.0.0.1:8004 \
  --categories cable \
  --key-file "$HOME/ihsanvision-artifacts/portfolio-operator-key.txt" \
  --timeout 300
~~~

If Cable passes, repeat one category at a time or run `--categories bottle cable carpet metal_nut capsule` for all ten. This can be resource-intensive on a Mac: only one prediction runs at once; wait for real checkpoint loading and observe host memory. Do not treat timeouts or memory failures as model-quality failures.

The `--key-file` must be a regular non-symlink file readable only by the owner (`chmod 600`); its contents are never printed by the script. Alternatively set a private `IHSANVISION_CLIENT_API_KEY` environment variable **in the calling terminal**, without copying it into GitHub or chat.

**Private history side effects:** Enabling the history database causes real successful test requests to add synthetic inspection rows. This command uses the returned *new* inspection ID to delete **only that specific synthetic row**, requiring authenticated `DELETE /inspections/{inspection_id}`. It never deletes pre-existing records. If an HTTP response is lost, execution is interrupted, or cleanup fails, a synthetic row may remain; inspect your private history manually instead of bulk-deleting unrelated records. For important real inspection data, make a private database backup before tests.

## What counts as passed

- `GET /models`: Both trained and approved models are available for every requested category.
- `GET /ready`: Operator-approved checkpoint SHA-256 checks pass. This alone is not inference.
- Each real authenticated `POST /predict?category=...&model=...`: The requested category and model are echoed correctly, anomaly score is finite, optional label/latency/cache flags have valid types, and a genuine PNG overlay of the input's size is returned.
- The tester confirms the newly generated private history record is individually deleted if an `inspection_id` is returned. Report a failure if cleanup is not confirmed.
- The final command prints `"status": "synthetic_inference_passed"`, with `tested_pairs` equal to the number exercised; it never stores images, checkpoint bytes, API keys, overlay base64 or inspection IDs in its output.

## Separate required real-image acceptance

To test **defect decisions** and the **actual browser dashboard**, the operator still needs permission-cleared representative *normal and anomalous* samples for each category. Keep sample files privately on the Mac. If a Kaggle session with authorized MVTec AD assets is still available, export one normal and one defect image per category to a private archive; do **not** upload MVTec photographs or model checkpoints to the public source repository. These manually chosen examples do not establish accuracy/recall and must not be mislabeled as unseen independent holdout data.

Open the actual operator UI at `http://127.0.0.1:8004/`, pick a category and model, enter the private key in the browser if requested, upload a **correct-category** image, and check visually that the defect heatmap corresponds to the input. Also inspect `/batch` and authenticated `/history`, including operator review, lot/CSV exports and persistence after an actual server restart. Do not mark the browser acceptance gate complete until these are observed with real images.

See [portfolio scope](portfolio-five-category-extension.md), [model registry](multicategory-model-registry.md), [private history](private-inspection-history.md), and [operator acceptance](operational-acceptance.md).