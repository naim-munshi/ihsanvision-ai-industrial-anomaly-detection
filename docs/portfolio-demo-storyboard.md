# IhsanVision.AI — real 90-second portfolio demo storyboard

**Purpose:** A recruiter should understand the end-to-end AI engineering
contribution in about 90 seconds. This is a **recording plan**, not a
fabricated product demo. Record the actual tested browser with real
operator-approved images; no fake heatmaps or scores.

## Before recording

- Run the latest code with **trusted Bottle PaDiM and PatchCore checkpoints**;
  verify both models via `GET /models` and actual expected-label smoke tests.
- Prepare one truly normal and one truly defective **rights-cleared,
  owner-created** Bottle image. Verify that the model outputs are honest for
  those samples. If a prediction fails, **show/explain the limitation** rather
  than change its label.
- Prepare a disposable private inspection history DB and strong API key;
  keep the browser key/password field and developer console **out of frame**.
- Use a made-up lot ID that contains no customer data. Do not expose local
  absolute checkpoint paths, `.env` values, private CSV contents or MVTec
  licensed source images.
- Plan to record at **1280×720 or better** with legible text; avoid browser
  pop-ups, private tabs or background desktop notifications.

## Target video sequence

| Time | Browser action | Short caption |
| --- | --- | --- |
| 0–8 s | Open the current `IhsanVision.AI` app | Industrial visual anomaly detection; PaDiM + PatchCore |
| 8–22 s | Select Bottle/PaDiM; inspect the actual normal image | Normal sample, uncalibrated model score, heatmap |
| 22–38 s | Inspect a real defect image; show its actual relative overlay | Anomalous region localized for operator review |
| 38–52 s | Switch to PatchCore and show the same sample | Two model families, same source category |
| 52–65 s | Open batch view; inspect a few rights-cleared images | Batch inspection and aggregate reporting |
| 65–79 s | Open private history; record a human review and lot summary | Audit-friendly operator workflow |
| 79–90 s | Show saved metric table / GitHub CI badge | Verified evaluation evidence and tested software |

If a real cold model load exceeds the allotted time, film a continuous
authentic recording and **label any editing/condensed segment**. Do not imply
the clipped segment is a latency benchmark.

## Three README screenshots

1. **Single image prediction:** Both original and overlaid normal/defect
   outcome with a visible model/category. Only publish an image that is
   yours or explicitly licensed for redistribution.
2. **Model switch:** Same permitted product sample inspected by the two
   genuinely loaded model families. The scores have different scales; never
   display them as probability percentages.
3. **Batch + human review:** Actual batch results, synthetic lot and
   confirmation/override workflow. Redact API key, real filenames, personal
   data and any customer data.

Suggested published files after rights review only:

```text
docs/media/real-bottle-inspection.png
docs/media/real-padim-patchcore.png
docs/media/real-operator-batch.png
```

No media files or images are created by this guide. Before committing any
assets, maintain a real, owner-approved
`docs/release-media-rights.md` stating filename, source/creator,
redistribution permission, anonymization and approval date. The source
hygiene CI checks the presence of this register, but **cannot certify
copyright or consent**.

## README/candidate presentation

Place the final **real** video link and rights-approved image links below
the project overview in README. Clearly mark whether the video shows the
latest tagged commit, the current model/category, or historical evidence.
Add the observed held-out PaDiM/PatchCore metric table, warm-serving p50/p95,
honest domain-shift and test-split cautions. Do not claim a public hosted
deployment, full MVTec AD leaderboard ranking or certified factory accuracy.

The original captured file(s) may be kept privately and hosted only with
owner consent. A YouTube/unlisted or public hosting account is **not**
required by the portfolio source itself.
