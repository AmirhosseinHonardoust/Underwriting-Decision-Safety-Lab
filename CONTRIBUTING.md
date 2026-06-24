# Contributing

Thanks for your interest in improving the Underwriting Decision Safety Lab.

## Development setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .          # optional: exposes the `underwriting-pipeline` command
pre-commit install        # optional: run the quality gate on every commit
```

## Quality gate

All changes must pass the same checks CI runs:

```bash
ruff check underwriting app tests
black --check underwriting app tests
mypy
coverage run -m unittest discover -s tests && coverage report
```

- `ruff` lints with `E, F, I, B, SIM, UP` at line length 100.
- `black` formats at line length 100.
- `mypy` runs in the project's strict configuration.
- Test coverage must stay at or above 85%.

Configuration for all of these lives in `pyproject.toml`.

## Conventions

- Keep changes small and focused; prefer the smallest change that achieves the goal.
- Preserve behavior in refactors. The pipeline is deterministic, so regenerated
  artifacts (`outputs/`, `reports/figures/`) should be identical unless a change is
  intended to alter them.
- Add or update tests for any behavior change.
- Internal imports inside the `underwriting` package are relative (e.g.
  `from .calibration import ...`).

## Pull requests

Keep each PR independently mergeable, describe what changed and why, and confirm
the quality gate passes locally before opening it.
