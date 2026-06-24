# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- Renamed the source package from `src` to `underwriting` so the project is
  distributable under a meaningful import path. The console entry point is
  `underwriting-pipeline`; run the module with `python -m underwriting.pipeline`.
- Array parameters and returns now use dtype-level aliases (`IntArray`,
  `FloatArray`) instead of bare `np.ndarray`.
- Raised the CI test-coverage floor to 85% and centralised coverage settings in
  `pyproject.toml`.
- The lint/format/type quality job now runs across Python 3.10, 3.11, and 3.12.

### Added
- Synthetic data generator (`underwriting.synthetic`, `underwriting-generate-data`)
  so the full pipeline runs end-to-end with no external data.
- Run provenance: every pipeline run writes `run_manifest.json` (input SHA-256,
  row count, config, and Python/library versions).
- `CONTRIBUTING.md` with development setup and the quality-gate commands.
- This changelog.

## [0.1.0]

### Added
- Calibrated underwriting pipeline with abstention/coverage policy, slice safety
  reporting, baselines, and policy variants.
- Streamlit dashboard, test suite, and GitHub Actions CI.
- Packaging metadata, pinned dev tooling, pre-commit hooks, and an enforced
  ruff/black/mypy quality gate.
