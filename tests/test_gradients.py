"""Analytic gradients of the calibration loss against central finite differences."""

import numpy as np
import pytest

from stereo_distance import forward_grads, forward_grads_batch, loss_only
from stereo_distance.calibration import _huber_grad


def _loss(p, gl, gr, d, huber=None):
    return forward_grads(*p, gl, gr, d, huber_delta=huber)[0]


def _random_state(seed):
    rng = np.random.default_rng(seed)
    gl = 0.45 + rng.uniform(-0.05, 0.05, (5, 5))
    gr = 0.45 + rng.uniform(-0.05, 0.05, (5, 5))
    d = np.array([rng.uniform(0.15, 0.35), rng.uniform(-0.01, 0.01), 0.0])
    # a plausible pair: uL to the right of uR, same-ish row
    u_r = rng.uniform(100, 2300)
    v = rng.uniform(50, 1900)
    p = (u_r + rng.uniform(80, 250), v, u_r, v + rng.uniform(-10, 10), rng.uniform(0.8, 6.0))
    return p, gl, gr, d


@pytest.mark.parametrize("seed", range(6))
def test_gamma_gradients_match_finite_differences(seed):
    p, gl, gr, d = _random_state(seed)
    _, g_gl, g_gr, _ = forward_grads(*p, gl, gr, d)
    eps = 1e-6
    for grid, analytic in ((gl, g_gl), (gr, g_gr)):
        num = np.zeros_like(grid)
        for i in range(5):
            for j in range(5):
                plus = grid.copy()
                plus[i, j] += eps
                minus = grid.copy()
                minus[i, j] -= eps
                if grid is gl:
                    num[i, j] = (_loss(p, plus, gr, d) - _loss(p, minus, gr, d)) / (2 * eps)
                else:
                    num[i, j] = (_loss(p, gl, plus, d) - _loss(p, gl, minus, d)) / (2 * eps)
        assert np.allclose(analytic, num, rtol=1e-5, atol=1e-7)
        # only the four nodes around the pixel receive gradient
        assert np.count_nonzero(analytic) <= 4


@pytest.mark.parametrize("seed", range(6))
def test_baseline_gradient_matches_finite_differences(seed):
    p, gl, gr, d = _random_state(seed)
    _, _, _, g_d = forward_grads(*p, gl, gr, d)
    eps = 1e-7
    for k in (0, 1):
        plus, minus = d.copy(), d.copy()
        plus[k] += eps
        minus[k] -= eps
        num = (_loss(p, gl, gr, plus) - _loss(p, gl, gr, minus)) / (2 * eps)
        assert g_d[k] == pytest.approx(num, rel=1e-5, abs=1e-6)
    assert g_d[2] == 0.0


def test_huber_gradients_match_finite_differences():
    # make a point with a large relative error so the Huber branch is active
    p, gl, gr, d = _random_state(11)
    p = (p[0], p[1], p[2], p[3], p[4] * 1.6)
    delta = 0.05
    loss, g_gl, g_gr, g_d = forward_grads(*p, gl, gr, d, huber_delta=delta)
    eps = 1e-6
    i, j = np.argwhere(g_gl != 0)[0]
    plus, minus = gl.copy(), gl.copy()
    plus[i, j] += eps
    minus[i, j] -= eps
    num = (_loss(p, plus, gr, d, delta) - _loss(p, minus, gr, d, delta)) / (2 * eps)
    assert g_gl[i, j] == pytest.approx(num, rel=1e-5)
    plus, minus = d.copy(), d.copy()
    plus[0] += eps
    minus[0] -= eps
    num_d = (_loss(p, gl, gr, plus, delta) - _loss(p, gl, gr, minus, delta)) / (2 * eps)
    assert g_d[0] == pytest.approx(num_d, rel=1e-5)
    # Huber caps the penalty relative to the squared loss
    assert loss < forward_grads(*p, gl, gr, d)[0]


def test_huber_helper_branches():
    e = np.array([0.01, -0.2])
    loss, grad = _huber_grad(e, 0.05)
    assert loss[0] == pytest.approx(0.5 * 0.01**2)
    assert grad[0] == pytest.approx(0.01)
    assert loss[1] == pytest.approx(0.05 * (0.2 - 0.025))
    assert grad[1] == pytest.approx(-0.05)
    loss_sq, grad_sq = _huber_grad(e, None)
    assert np.allclose(loss_sq, 0.5 * e * e) and np.allclose(grad_sq, e)


def test_loss_is_zero_at_ground_truth_and_positive_elsewhere(distorted, scene_points):
    left, right = distorted.project(scene_points)
    ok = distorted.in_frame(left, right)
    pts = [(*left[i], *right[i], scene_points[i, 2]) for i in np.flatnonzero(ok)]
    assert loss_only(pts, distorted.gamma_l, distorted.gamma_r, distorted.d) < 1e-6
    wrong_d = distorted.d * 1.1
    assert loss_only(pts, distorted.gamma_l, distorted.gamma_r, wrong_d) > 1e-3
    # gradient at the optimum vanishes
    _, g_gl, g_gr, g_d = forward_grads_batch(np.asarray(pts), distorted.gamma_l,
                                             distorted.gamma_r, distorted.d)
    assert np.abs(g_gl).max() < 1e-5 and np.abs(g_gr).max() < 1e-5 and np.abs(g_d).max() < 1e-5


def test_batch_gradient_equals_sum_of_per_point_gradients():
    rng = np.random.default_rng(5)
    states = [_random_state(s) for s in range(12)]
    _, gl, gr, d = states[0]
    pts = np.asarray([s[0] for s in states])
    tot, b_gl, b_gr, b_d = forward_grads_batch(pts, gl, gr, d)
    s_tot, s_gl, s_gr, s_d = 0.0, np.zeros((5, 5)), np.zeros((5, 5)), np.zeros(3)
    for p in pts:
        loss, a, b, c = forward_grads(*p, gl, gr, d)
        s_tot += loss
        s_gl += a
        s_gr += b
        s_d += c
    assert tot == pytest.approx(s_tot, rel=1e-12)
    assert np.allclose(b_gl, s_gl) and np.allclose(b_gr, s_gr) and np.allclose(b_d, s_d)
    del rng


def test_degenerate_pair_gives_zero_gradient():
    gl = gr = np.full((5, 5), 0.45)
    d = np.array([0.25, 0.0, 0.0])
    loss, g_gl, g_gr, g_d = forward_grads(1296.0, 972.0, 1296.0, 972.0, 2.0, gl, gr, d)
    assert loss == 0.0 and not g_gl.any() and not g_gr.any() and not g_d.any()
