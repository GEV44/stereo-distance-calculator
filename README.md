# Stereo Distance Calculator

[![CI](https://github.com/GEV44/stereo-distance-calculator/actions/workflows/ci.yml/badge.svg)](https://github.com/GEV44/stereo-distance-calculator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Pure-NumPy stereo vision system** for measuring real-world distance (meters) from a calibrated pair of cameras. No deep learning, no PyTorch — only analytic geometry, hand-derived gradients, and a lightweight Pygame UI.

Built for a dual-camera rig (2592×1944 px). The full mathematical derivation is in [`docs/MATHEMATICAL_FOUNDATION.pdf`](docs/MATHEMATICAL_FOUNDATION.pdf).

---

## Highlights

| Aspect | Detail |
|--------|--------|
| **Model** | Per-camera 5×5 **Γ (gamma) grid** with bilinear interpolation — encodes lens distortion and effective focal length without polynomial distortion models |
| **Optimization** | Analytical gradients + **Adam** + Huber loss (δ = 5%) + Laplacian smoothness + anchor prior |
| **Matching** | Local fundamental matrix, **1D ZNCC** along epipolar line, parabolic sub-pixel refinement |
| **Triangulation** | OLS ray intersection: \(x = (A^\top A)^{-1} A^\top d\) |
| **Post-correction** | Quadratic polynomial \(Z_{\text{true}} = a + b Z_{\text{pred}} + c Z_{\text{pred}}^2\) |
| **Runtime** | Training ~10⁶ ops in **< 1 s** on CPU; inference O(1) per click |

---

## How it works (pipeline)

```mermaid
flowchart LR
    A[Click left image] --> B[Ray from Γ grid]
    B --> C[Epipolar line in right image]
    C --> D[1D ZNCC search]
    D --> E[OLS triangulation → Z]
    E --> F[Polynomial correction]
```

1. **Normalized coordinates** — \(u_n = (u - C_X)/C_X\), \(v_n = (v - C_Y)/C_Y\) for stable optimization.
2. **Ray** — \(\mathbf{v} = [u_n \gamma_x,\; v_n \gamma_y,\; 1]^\top\) with \(\gamma\) from bilinear interpolation on the grid.
3. **Stereo geometry** — \(Z \mathbf{v}_L - Z_R \mathbf{v}_R = \mathbf{d}\); solved by least squares when rays do not meet exactly.
4. **Loss** — Relative disparity \(e_x = (v_{Lx} - v_{Rx}) / (d_x / Z_{\text{true}}) - 1\) avoids \(Z^2\) gradient blow-up at large distances; epipolar \(e_y\) aligns vertical rays with baseline \(d_y\).

See the PDF for equations, gradient chain rule, Adam update, and function-to-math mapping (`bilinear`, `build_ray`, `triangulate`, `forward_grads`, `train`, etc.).

---

## Results

Leave-one-out evaluation on the 27 calibration points in `cal_pts.json` (0.41–4.74 m): for each point the model is trained on the other 26 with the default settings, then the held-out point is triangulated from its labelled left/right pixels.

| Metric | Value |
|---|---|
| Median distance error | **6.0%** |
| Points within 10% | 18 / 27 |
| Points within 5% | 11 / 27 |
| Mean distance error | 10.1% (two outliers: 31% and 77%) |
| In-sample error (all 27 points) | 3.0% |

The gap between in-sample (3%) and held-out (6% median) error is the honest generalisation cost of a 52-parameter model fitted on 27 points. The two large outliers (true 3.35 m and 4.05 m) are the first calibration samples to re-check. This test covers the geometric model only; automatic ZNCC matching adds its own error on top.

Reproduce with:

```bash
python eval_loocv.py            # ~25 min on one CPU core (27 trainings × 9000 epochs)
```

<details><summary>Per-point results</summary>

| True distance (m) | Predicted (m) | Error |
|---:|---:|---:|
| 0.41 | 0.47 | 14.0% |
| 0.72 | 0.81 | 12.7% |
| 1.35 | 1.35 | 0.4% |
| 1.64 | 1.63 | 0.8% |
| 1.64 | 1.69 | 3.3% |
| 1.90 | 1.88 | 1.3% |
| 2.02 | 1.97 | 2.3% |
| 2.17 | 1.99 | 8.1% |
| 2.25 | 2.01 | 10.8% |
| 2.35 | 2.23 | 4.9% |
| 2.50 | 2.43 | 2.8% |
| 2.52 | 2.74 | 8.6% |
| 2.55 | 2.53 | 0.8% |
| 2.60 | 2.45 | 5.6% |
| 2.95 | 3.12 | 5.8% |
| 3.11 | 3.38 | 8.7% |
| 3.33 | 2.99 | 10.2% |
| 3.35 | 5.94 | 77.2% |
| 3.40 | 3.46 | 1.9% |
| 3.45 | 3.24 | 6.1% |
| 3.76 | 3.68 | 2.2% |
| 3.96 | 3.56 | 10.1% |
| 4.05 | 2.79 | 31.1% |
| 4.30 | 4.04 | 6.0% |
| 4.45 | 4.61 | 3.5% |
| 4.54 | 3.78 | 16.8% |
| 4.74 | 3.98 | 16.1% |

</details>

---

## Repository layout

```
├── run_math.py              # Main app: calibration, matching, measurement UI
├── collect_points.py        # Batch tool to label left/right pairs + distances
├── eval_loocv.py            # Leave-one-out accuracy evaluation
├── stereo_calibration.json  # Pre-trained Γ grids + baseline (example)
├── cal_pts.json             # Calibration point cache
├── data/
│   └── sample_training_pairs.json
├── docs/
│   └── MATHEMATICAL_FOUNDATION.pdf
├── left/  right/            # Place stereo image pairs here (for collect_points.py)
├── requirements.txt
└── README.md
```

**Removed from the working tree (not needed in repo):** IDE settings (`.claude/`), duplicate dataset `training_data32.json`.

---

## Requirements

- Python 3.10+
- NumPy, Pygame, OpenCV (for `collect_points.py` only)

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Measure distance (interactive)

Place or use any left/right image pair (full resolution 2592×1944 recommended):

```bash
python run_math.py path/to/left.jpg path/to/right.jpg
```

If `stereo_calibration.json` exists, the app loads it and is ready to measure. Otherwise, add **≥ 15 calibration points** (known distance in meters); training runs automatically and saves calibration.

### 2. Collect calibration data (optional)

```bash
# Put matched images in left/ and right/
python collect_points.py
```

Exports `training_data.json` with `(left_pt, right_pt, distance)` entries.

---

## Controls (`run_math.py`)

| Key | Action |
|-----|--------|
| **Left click** | Pick point on left, then right |
| **K** | Add calibration point (enter distance, then click L → R) |
| **U** | Undo last calibration point |
| **C** | Clear calibration and reset model |
| **R** | Clear stored measurements |
| **T** | Swap left/right images |
| **Wheel** | Zoom panel under cursor |
| **Right-drag** | Pan |
| **Esc** | Quit |

The right panel shows the **epipolar curve**, **ZNCC hint** (green cross), and depth rings at 1 m / 2 m / 5 m when calibrated.

---

## Technical parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| \(C_X, C_Y\) | 1296, 972 | Optical center (image center) |
| Grid | 5×5 | Γ nodes per camera (center frozen to \(G_0 = C_X / f_{\text{init}}\)) |
| \(f_{\text{init}}\) | 2880 px | Nominal focal length |
| \(d_x\) min | 0.02 m | Baseline floor |
| Γ clamp | [0.05, 5.0] | Physical focal-scale bounds |
| Huber δ | 0.05 | 5% relative error threshold |
| Min cal points | 15 | Before auto-train |

---

## Author

**Gevorg** — [GitHub @GEV44](https://github.com/GEV44)

---

## License

MIT — see [LICENSE](LICENSE).
