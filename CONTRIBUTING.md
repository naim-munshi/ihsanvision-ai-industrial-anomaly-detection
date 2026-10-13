# Contributing to IhsanVision.AI

Thank you for helping improve the research prototype. The default repository branch is currently `MdNaim`. Development and review take place on feature branches; the active portfolio-development branch is `feature/portfolio-production-pipeline`.

## Development workflow

1. Work in a feature branch rather than committing directly to `MdNaim`.
2. Keep experiments, model weights, training images, secrets and user data **out of Git**.
3. Use Python 3.11 or 3.12 and follow [development.md](docs/development.md) for setup.
4. Update unit tests and documentation for changes to API behavior, metrics or model evaluation.
5. Run `python -m compileall -q src scripts tests` and `python -m pytest -q` before opening a PR.
6. Include reproducible commands, package versions, environment, limitations and sample outputs when reporting results.
7. Keep the draft PR open until actual model/API smoke tests are observed. GitHub Actions covers **mocked** contracts, not a full live hosted service.

## Review checklist

- Does the change preserve checkpoint compatibility and experiment provenance?
- Are training, validation and test images disjoint?
- Are F1 and AUROC reported separately, with sufficient context?
- Does any endpoint reveal file paths, arbitrary exceptions, secrets or serialized checkpoint upload?
- Does the documentation distinguish verified behavior from pending work?
- Did both Python 3.11 and Python 3.12 CI tests pass?

## Code style

Favor explicit module boundaries, descriptive names, type hints for public interfaces and short testable functions. Do not silently change evaluation protocols or dependencies without recording the consequences.

## Security

Do not publicly paste credentials, private model artifacts or user images. Report vulnerabilities privately to the maintainer rather than publishing exploit details in a GitHub issue.

## Licensing

An explicit source code license has **not** yet been selected by the repository owner. Do not assume the training dataset license grants the right to redistribute the data or derived commercial assets.
