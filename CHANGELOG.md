# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-05

The method is unchanged: 5x5 pure-gamma lens grid per camera, relative-disparity
loss with hand-derived gradients, Laplacian smoothness, frozen centre node,
OLS triangulation. This release turns the two scripts into a tested, installable
package and measures how accurate the method is.

### Added
- `stereo_distance` package: `config`, `lens`, `triangulation`, `calibration`,
  `matching`, `io`, `synthetic`, `cli`, `ui`, with type hints and docstrings.
- `RigConfig` dataclass, so another sensor size, focal prior or grid size is a
  parameter rather than an edit to module constants.
- CLI: `python -m stereo_distance calibrate | measure | predict | info | ui`,
  plus a `stereo-distance` console script.
- Vectorised ray building, triangulation and gradient accumulation. A default
  9000-epoch calibration on the 27 shipped points takes 5.5 s instead of
  270 s, and the learned grids and baseline match the original
  implementation's to 4e-15.
- Opt-in Huber loss (`--huber`) and Adam optimiser (`--optimizer adam`) as
  derived in `docs/MATHEMATICAL_FOUNDATION.pdf`. Both are off by default, so
  default training behaves as before.
- `SyntheticRig`: projects known 3-D points through a known (optionally
  distorted) gamma model, for tests and benchmarks.
- 76 pytest tests: geometry round-trips, analytic gradients against finite
  differences, ZNCC/SSD/LK on planted shifts, calibration convergence,
  persistence and CLI.
- `benchmarks/synthetic_eval.py`: depth error against distance and against pixel
  noise on a synthetic rig, plus leave-one-out on the 27 real labelled
  points. Figures go to `docs/img/`, tables to `benchmarks/results.md`.
- `pyproject.toml` (version 1.0.0). The core depends only on numpy; `pygame`
  is the `[ui]` extra and `opencv-python` the `[collect]` extra. Ruff is
  configured.
- CHANGELOG.

### Changed
- `run_math.py` is now a thin wrapper that re-exports the original function
  names and keeps `python run_math.py left.jpg right.jpg` working.
- Legacy 4x4 calibration files are upsampled with a built-in bilinear resize,
  so loading them no longer needs scipy.
- CI runs ruff and pytest on Ubuntu and Windows with Python 3.12 and a pip
  cache, plus CLI and benchmark smoke steps.
- README rewritten: method, equations, measured results, layout and
  limitations.

### Fixed
- Calibration and point files are read and written as UTF-8 explicitly.
- `load_cal_pts` also accepts the `{left_pt, right_pt, distance}` records
  that `collect_points.py` writes.
