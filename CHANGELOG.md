# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. The README now has a plain-English problem statement, a pipeline
  diagram, key concepts, "What you will learn" and an updated layout. New `docs/walkthrough.md`.
- README quick start: the `--trades your_fills.csv` placeholder is replaced by commands that run as written.
- README: the synthetic signal is described as take-profit distance plus a slow regime variable, as in the code.

### Fixed
- `features.py` used `Timestamp.dayofweek`, which pandas 3 deprecates; it now uses `day_of_week`. This removes
  about 4,200 warnings from the test run.

### Added
- `scripts/train_bundle.py`, so the API serving steps in the README can be run as written, with a test.

## [1.1.0] - 2026-11-15

### Added
- `original/`: the original research code, sanitised (names, credentials, data and notebook outputs removed).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-04

### Added
- First public release: the maintained code, documentation, tests and CI.
