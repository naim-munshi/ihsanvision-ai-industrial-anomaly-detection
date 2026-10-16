# M6 — Local MVP API error-state acceptance

## Scope and status

The user has already observed all four normal/defective × PaDiM/PatchCore API predictions on the Mac and browser overlays for both models. This gate focuses on **predictable failure behavior**, rather than repeating model training or performance benchmarks.

The offline FastAPI TestClient regression suite now covers the following error contracts:

| Case | Expected HTTP status | Acceptance requirement |
| --- | --- | --- |
| No configured trusted checkpoint | 503 | No inference occurs |
| Unknown model name | 400 | Clear validation response; predictor never called |
| Empty uploaded file | 400 | Clear validation response; predictor never called |
| Invalid image bytes | 415 | Reject malformed image |
| Image larger than 10 MiB | 413 | Reject oversized uploaded file |
| Internal predictor exception | 500 | Return a generic error, not internal exception text |
| Browser root | 200 | Dashboard HTML is served |

The new error-path tests also check that **X-Request-ID** is present, allowing a client to correlate failures with server logs without exposing local checkpoint paths. Tests use mocked inference and synthetic in-memory files only; they do **not** load checkpoints, require Kaggle data, or establish real-device behavior.

Run the complete offline suite from an installed development checkout:

~~~bash
python -m pytest -q
~~~

## Final real Mac verification still required

On the Mac that holds the two trusted saved checkpoints, update the checkout and run the one-command local dashboard launcher:

~~~bash
python -m scripts.run_local_demo --port 8004
~~~

The launcher should open the local dashboard and expose both models. Manually confirm one normal and one defective example per model, confirm an informative error when an invalid file is submitted, and stop the server with Ctrl+C. Use port 8004 only if the previously running server still occupies 8003.

**Not yet verified:** fresh real-Mac run of the one-command launcher; all error states in the real browser; source-image membership in the archived held-out split; final media/screenshots. Offline CI passing does not satisfy these device-specific gates.

## Release boundary

This is a **local research/portfolio MVP**. It is not a public cloud deployment, industrial safety certification, calibrated anomaly probability, production reliability claim or release authorization. Private model checkpoints, datasets and API secrets must remain outside Git. Source-code licensing and publishing require explicit approval.
