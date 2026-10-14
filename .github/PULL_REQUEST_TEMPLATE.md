## Scope

Describe **what** changed and **why**. Link the issue/experiment, if any.

## Verification

- [ ] `python -m pytest -q` passes on Python 3.11 and 3.12
- [ ] Canonical `ihsanvision.*` imports and legacy entry points still work
- [ ] Documentation and CLI commands updated where needed
- [ ] If model code changed: real checkpoint/dataset smoke test with provenance
- [ ] If evaluation code changed: train/val/test path-level split audit
- [ ] Dataset/model binaries, secrets and user images are not committed

## Evidence and limitations

Commands, environment, observed output/metrics, hardware and outstanding verification steps. **Do not claim measured results from mocked tests.**

## Release impact

Does this PR change API compatibility, checkpoint requirements, packaging, deployment risk or licensing assumptions?
