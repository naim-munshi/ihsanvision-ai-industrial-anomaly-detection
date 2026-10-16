# Operator-configured multi-category industrial inspection

## Capability and limitations

The original IhsanVision.AI vision covers all **15 MVTec AD product categories**, plus
selected **MVTec AD 2** categories (`sheet_metal`, `can`, and optionally
`wall_plugs`). The FastAPI application originally supported a single `bottle`
checkpoint per model. It now supports **selecting from operator-registered,
per-category, per-model trusted checkpoints**, without retraining or ever
accepting model paths from an HTTP client.

**This change does not supply pretrained weights for the additional
categories**, and it does not implement an MVTec AD 2 dataloader or assert
that AD2 experiments were performed. An additional category appears in the
dashboard only after a real trusted checkpoint is locally configured.
Existing bottle users do not need to make any changes.

## Configure trusted checkpoints

Create a private JSON file **outside the source repository**, for example
`~/ihsanvision-artifacts/model-registry.json`:

~~~json
{
  "version": 1,
  "models": [
    {
      "category": "cable",
      "model": "padim",
      "checkpoint": "/absolute/private/path/to/cable-padim.ckpt"
    },
    {
      "category": "cable",
      "model": "patchcore",
      "checkpoint": "/absolute/private/path/to/cable-patchcore.ckpt"
    }
  ]
}
~~~

The paths above are placeholders, **not working checkpoints**. Only register
weights from training sessions you control or have explicitly verified.
Anomalib/Lightning checkpoints can deserialize arbitrary Python objects and
**must never be sourced from user uploads**.

Launch from a checkout with the dependencies installed:

~~~bash
export IHSANVISION_MODEL_REGISTRY="$HOME/ihsanvision-artifacts/model-registry.json"
python -m scripts.run_local_demo --no-browser
~~~

The one-command bottle launcher still expects both original bottle weights in
its default artifact directory. If you are operating **only** category-specific
checkpoints without bottle weights, launch `python -m uvicorn
ihsanvision.api:app --host 127.0.0.1 --port 8003` directly. Both forms keep
inference bound to localhost. The private JSON may augment or override matching
`bottle` checkpoint entries, without altering other registered pairs.

## API contract

`GET /models` preserves the legacy `available_models` and `default_model`
fields **for the default available category** and adds:

~~~json
{
  "available_models": ["padim", "patchcore"],
  "default_model": "padim",
  "default_category": "bottle",
  "available_categories": ["bottle", "cable"],
  "models_by_category": {
    "bottle": ["padim", "patchcore"],
    "cable": ["padim", "patchcore"]
  }
}
~~~

This is an illustrative contract, not a claim that the example cable weights
exist. Only registered and available categories/model names are exposed;
checkpoint paths are never returned. `GET /health` reports whether any trusted
category/model is configured.

For a registered category, send your usual multipart image upload to
`POST /predict?model=patchcore&category=cable`. The response contains the
additional `category` field alongside the existing model, score, predicted
label and base64 PNG relative visualization. For a legacy bottle-only server,
`POST /predict?model=padim` continues to work unchanged.

- Unknown category or model: HTTP 400.
- Supported category/model without a registered local checkpoint: HTTP 503.
- Invalid private registry file or missing trusted weights: configuration
  error that must be resolved by the operator, **not** silently ignored.
- The browser displays only available trained categories, updates the allowed
  models on selection and checks that returned model/category match the request.

## Real end-to-end acceptance gates

For each newly added category/model pair, record its dataset source and
license, per-category normal train/val/test split, checkpoint and exact SHA-256,
threshold provenance, image-level and pixel-level metrics and failure cases,
and then verify a real local image with `/predict` and the dashboard.
Do **not** reuse a threshold or raw score scale from a different category.
Only a verified per-category model should be presented to users.

The FastAPI routing/browser functionality is covered by offline regression
tests. Full real-checkpoint validation on **additional** categories still needs
the actual trained artifacts and user-machine execution. No MVTec images,
weights, secrets or source datasets are included in the repository.

## Next full-platform milestones

1. Train and validate two models for each supported MVTec AD category,
   applying the exact leakage-safe split and evaluation protocol.
2. Implement separate AD2 dataset adapters respecting their different
   structure and rights; only then test `sheet_metal`, `can`,
   and `wall_plugs`.
3. Add controlled multi-image inspection jobs, exportable inspection reports,
   calibration/error analysis, versioned model metadata and audit-friendly
   workflow storage.
4. Verify container lifecycle, access control, operator logs, rate limiting,
   memory, latency, installation, public deployment security and rollback
   with real infrastructure before production claims.
