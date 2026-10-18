# 2026-10-19 (JST) — reachable Git history source-security audit

**Status:** Automated source-history scan completed with
`no_known_patterns_found`. This is a **partial P10 source-hygiene
evidence point**, not a full security, licensing, scientific or release
approval.

**Observed GitHub Actions evidence:**
[Review reachable Git history for release — run 39140144697](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/runs/39140144697).

| Audited property | Observed result |
| --- | --- |
| Workflow source SHA | `118ee17eb8c87150c92427e6c5df37cacb7b9ae1` |
| Repository history coverage | All branches/tags fetched by the workflow |
| Repository was shallow | No |
| Distinct historic paths inspected | **184** |
| Reachable Git blobs found | **509** |
| Git text blobs inspected | **509** |
| Historical text bytes scanned | **3,356,177** |
| Oversize blobs excluded from content scan | **0** |
| Recognized sensitive path/secret patterns found | **0** |
| Full rights/secret clearance claimed | **No** |
| Portfolio v1.0 release authorized | **No** |

The workflow performed a full-history fetch, scanned previous file paths
even when deleted from the current `main` tree, examined bounded Git
blob contents for known credential patterns, and stored an anonymous JSON
report in a 14-day GitHub Actions artifact. It does not output credential
values or private file names.

## Scientific and security interpretation

- **Zero findings** means no matches to this scanner's limited
  signatures among reachable fetched Git objects; it does **not** mean
  no sensitive data ever existed in forks, unreachable objects, encoded
  content, external storage or previously shared archives.
- The original source-only
  [HEAD hygiene check](../scripts/check_release_hygiene.py) remains an
  independent check. A passing history scan is not a substitute for it.
- Copyright/redistribution permissions for MVTec AD, pretrained model
  weights and original demo screenshots **still require human review**.
- The repository's
  [responsible-use license](../LICENSE) is custom and use-restricted,
  not OSI-approved; legal suitability and contributor rights still
  require review.
- No trained checkpoint, original MVTec image or user-owned private
  file was needed for this audit. This audit does not exercise model
  inference, training, Docker with weights or owner-facing browser flows.
- **Rerun this workflow immediately before tagging `v1.0.0`** because
  a new commit can introduce historical risk after this dated scan.

Related release evidence and remaining owner work:
[release readiness](../docs/portfolio-release-readiness.md) ·
[fixed v1.0 acceptance gates](../docs/portfolio-v1-plan.md).
