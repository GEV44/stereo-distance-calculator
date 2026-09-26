# Data

| File | Description |
|------|-------------|
| `../cal_pts.json` | The 27 labelled points of the real rig, `[uL, vL, uR, vR, Z]` (pixels at 2592×1944, depth in metres). Written by the measuring app; used for training and all real-data results. |
| `sample_training_pairs.json` | The first 25 of those points in the labelled-record format written by `collect_points.py` (`left_pt`, `right_pt`, `distance`). |
| `calibration_v1_legacy.json` | The calibration produced by v1 of this project (single-channel Γ, no rotation). Kept as a backward-compatibility test fixture: v2 loads it and reproduces v1's depths exactly. |

Both point formats are accepted everywhere (`stereo_gamma.data.load_points`).
