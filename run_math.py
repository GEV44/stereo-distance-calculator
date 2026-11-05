"""
Pure-Gamma Stereo Model - 5x5 Grid per Camera
==============================================
Thin compatibility wrapper.  The implementation now lives in the
``stereo_distance`` package; this script keeps the original entry point and
re-exports the original function names so older notebooks keep working:

    python run_math.py <left_image> <right_image>      (same as before)
    python -m stereo_distance ui <left> <right>        (new entry point)

MODEL   v = [un*g, vn*g, 1],  g = bilinear(u, v, Gamma)  (5x5 grid per camera)
LOSS    relative disparity  L_1D = 1/2 (disp_pred/disp_true - 1)^2  +  L_Y
TRAIN   analytic gradients, batch GD, Laplacian smoothness, frozen centre node
INFER   A = [vL, -vR];  x = (A^T A)^-1 A^T d;  Z = x[0]
"""

import sys

from stereo_distance import DEFAULT_RIG as _RIG
from stereo_distance import (
    bilinear,
    build_ray,
    forward_grads,
    laplacian_grad,
    lk_refine,
    load_cal_pts,
    load_calibration,
    ncc_match,
    predict_right,
    save_cal_pts,
    save_calibration,
    ssd_subpixel,
    triangulate,
)
from stereo_distance.calibration import TrainConfig
from stereo_distance.calibration import train as _train
from stereo_distance.io import CAL_PTS_PATH, CALIB_PATH
from stereo_distance.lens import default_params as defaults
from stereo_distance.matching import clahe as _clahe

# Original module-level constants, derived from the default rig
IMG_W, IMG_H = _RIG.img_w, _RIG.img_h
CX, CY = _RIG.cx, _RIG.cy
GRID_ROWS, GRID_COLS = _RIG.grid_rows, _RIG.grid_cols
CELL_W, CELL_H = _RIG.cell_w, _RIG.cell_h
MIN_CAL_PTS = _RIG.min_cal_pts
F_INIT = _RIG.f_init
DX_MIN = _RIG.dx_min
G_MIN, G_MAX = _RIG.g_min, _RIG.g_max
SMOOTH_W = _RIG.smooth_w
G0_INIT = _RIG.g0


def train(cal_pts, epochs=9000, verbose=300):
    """Original signature: returns ``(GammaL, GammaR, d)``."""
    res = _train(cal_pts, TrainConfig(epochs=epochs, verbose_every=verbose))
    return res.gamma_l, res.gamma_r, res.d


def run(left_path, right_path):
    from stereo_distance.ui import run as _run

    _run(left_path, right_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_math.py <left_image> <right_image>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
