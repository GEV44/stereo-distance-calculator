"""Tests for the pure-NumPy stereo model in run_math.py (no display needed)."""

import math
from pathlib import Path

import numpy as np
import pytest

import run_math as rm

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def calibration():
    return rm.load_calibration(str(ROOT / rm.CALIB_PATH))


@pytest.fixture(scope="module")
def cal_pts():
    return rm.load_cal_pts(str(ROOT / rm.CAL_PTS_PATH))


# ─── bilinear lookup ──────────────────────────────────────────────────────────
def test_bilinear_returns_node_value_on_grid_nodes():
    rng = np.random.default_rng(0)
    gamma = rng.uniform(0.3, 0.6, size=(rm.GRID_ROWS, rm.GRID_COLS))
    for iy in range(rm.GRID_ROWS - 1):
        for ix in range(rm.GRID_COLS - 1):
            g, _, _ = rm.bilinear(ix * rm.CELL_W, iy * rm.CELL_H, gamma)
            assert g == pytest.approx(gamma[iy, ix])


def test_bilinear_weights_form_a_partition_of_unity():
    gamma = np.ones((rm.GRID_ROWS, rm.GRID_COLS))
    for u, v in [(0, 0), (123.4, 987.6), (rm.IMG_W - 1, rm.IMG_H - 1), (1296, 972)]:
        g, w, _ = rm.bilinear(u, v, gamma)
        assert sum(w) == pytest.approx(1.0)
        assert g == pytest.approx(1.0)


# ─── triangulation ───────────────────────────────────────────────────────────
def test_shipped_calibration_reproduces_measured_distances(calibration, cal_pts):
    gamma_l, gamma_r, d = calibration
    errors = []
    for uL, vL, uR, vR, z_true in cal_pts:
        z, _, _, err = rm.triangulate(uL, vL, uR, vR, gamma_l, gamma_r, d)
        assert err is None
        errors.append(abs(z - z_true))
    assert len(errors) >= rm.MIN_CAL_PTS
    assert np.mean(errors) < 0.15  # metres; currently ≈ 0.095 m on 27 points


def test_triangulate_rejects_non_positive_disparity(calibration):
    gamma_l, gamma_r, d = calibration
    z, _, _, err = rm.triangulate(1000, 900, 1000, 900, gamma_l, gamma_r, d)
    assert math.isnan(z)
    assert "bad disparity" in err


def test_depth_decreases_as_disparity_grows(calibration):
    gamma_l, gamma_r, d = calibration
    depths = [rm.triangulate(1500, 972, 1500 - disp, 972, gamma_l, gamma_r, d)[0]
              for disp in (100, 200, 400)]
    assert depths[0] > depths[1] > depths[2] > 0


def test_predict_right_round_trips_through_triangulate(calibration):
    gamma_l, gamma_r, d = calibration
    for uL, vL, z_true in [(1800, 900, 2.0), (2200, 1200, 3.5), (1400, 700, 1.2)]:
        uR, vR = rm.predict_right(uL, vL, z_true, gamma_l, gamma_r, d)
        z, _, _, err = rm.triangulate(uL, vL, uR, vR, gamma_l, gamma_r, d)
        assert err is None
        assert z == pytest.approx(z_true, rel=0.02)


# ─── analytic gradients ──────────────────────────────────────────────────────
def _loss(sample, gamma_l, gamma_r, d):
    return rm.forward_grads(*sample, gamma_l, gamma_r, d)[0]


def test_analytic_gradients_match_finite_differences(calibration, cal_pts):
    gamma_l, gamma_r, d = (a.copy() for a in calibration)
    # Perturb so the loss and its gradients are clearly non-zero.
    rng = np.random.default_rng(1)
    gamma_l += rng.normal(0, 0.01, gamma_l.shape)
    gamma_r += rng.normal(0, 0.01, gamma_r.shape)
    d = d + np.array([0.01, 0.002, 0.0])
    eps = 1e-6
    for sample in cal_pts[:5]:
        _, g_l, g_r, g_d = rm.forward_grads(*sample, gamma_l, gamma_r, d)
        for grid, grad in ((gamma_l, g_l), (gamma_r, g_r)):
            for idx in np.ndindex(grid.shape):
                old = grid[idx]
                grid[idx] = old + eps
                up = _loss(sample, gamma_l, gamma_r, d)
                grid[idx] = old - eps
                down = _loss(sample, gamma_l, gamma_r, d)
                grid[idx] = old
                assert grad[idx] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-6)
        for k in (0, 1):
            dp, dm = d.copy(), d.copy()
            dp[k] += eps
            dm[k] -= eps
            numeric = (_loss(sample, gamma_l, gamma_r, dp) - _loss(sample, gamma_l, gamma_r, dm)) / (2 * eps)
            assert g_d[k] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_laplacian_gradient_is_zero_for_a_flat_grid():
    flat = np.full((rm.GRID_ROWS, rm.GRID_COLS), rm.G0_INIT)
    assert np.allclose(rm.laplacian_grad(flat), 0.0)


def test_laplacian_gradient_pushes_a_spike_down():
    grid = np.full((rm.GRID_ROWS, rm.GRID_COLS), 0.45)
    grid[2, 2] += 0.1
    grad = rm.laplacian_grad(grid)
    assert grad[2, 2] > 0          # descent lowers the spike
    assert grad[1, 2] < 0          # and raises its neighbours


# ─── persistence ─────────────────────────────────────────────────────────────
def test_calibration_save_load_round_trip(tmp_path, calibration):
    gamma_l, gamma_r, d = calibration
    path = tmp_path / "calib.json"
    rm.save_calibration(gamma_l, gamma_r, d, path=str(path))
    gl2, gr2, d2 = rm.load_calibration(str(path))
    assert np.allclose(gl2, gamma_l) and np.allclose(gr2, gamma_r) and np.allclose(d2, d)


def test_load_cal_pts_handles_missing_and_empty_files(tmp_path):
    assert rm.load_cal_pts(str(tmp_path / "missing.json")) == []
    empty = tmp_path / "empty.json"
    empty.write_text("  ")
    assert rm.load_cal_pts(str(empty)) == []
