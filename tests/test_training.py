import numpy as np
import pytest

from stereo_gamma.config import DEFAULT_SENSOR as S
from stereo_gamma.config import TrainConfig
from stereo_gamma.evaluation import fit_disparity_offset, metrics
from stereo_gamma.model import StereoModel
from stereo_gamma.synthetic import SyntheticRig
from stereo_gamma.training import Adam, fit_postcorrection, huber, linear_init, objective, train


def _theta(rng):
    return {"gL": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
            "gR": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
            "d": np.array([0.2, 0.01]), "w": np.array([0.01, -0.02, 0.03])}


@pytest.mark.parametrize("delta", [0.02, 10.0])  # Huber in its linear and quadratic regime
def test_analytic_gradient_matches_finite_differences(real_points, rng, delta):
    theta = _theta(rng)
    cfg = TrainConfig(huber_delta=delta, anchor=0.1, smooth=0.05, weight_y=0.7)
    _, g = objective(theta, real_points, cfg)
    h = 1e-6
    for k in theta:
        for idx in np.ndindex(theta[k].shape):
            tp = {a: b.copy() for a, b in theta.items()}
            tm = {a: b.copy() for a, b in theta.items()}
            tp[k][idx] += h
            tm[k][idx] -= h
            num = (objective(tp, real_points, cfg, need_grad=False)[0]
                   - objective(tm, real_points, cfg, need_grad=False)[0]) / (2 * h)
            assert num == pytest.approx(g[k][idx], rel=1e-4, abs=1e-9), (k, idx)


def test_huber():
    e = np.array([-2.0, -0.01, 0.0, 0.03, 1.0])
    rho, psi = huber(e, 0.05)
    assert np.allclose(psi, np.clip(e, -0.05, 0.05))
    assert np.isclose(rho[1], 0.5 * 0.01 ** 2) and np.isclose(rho[4], 0.05 * (1 - 0.025))


def test_adam_minimises_quadratic():
    theta = {"x": np.array([3.0, -2.0])}
    opt = Adam({"x": 0.1})
    for _ in range(2000):
        opt.step(theta, {"x": 2 * theta["x"]})
    assert np.allclose(theta["x"], 0, atol=1e-3)


def test_linear_init_is_close_for_small_rotations(rng):
    """First-order (small-angle) warm start: within 0.03° — Adam refines the rest."""
    m = StereoModel.nominal(S, dx=0.25)
    m.rotation = np.array([2e-4, 4e-3, -3e-3])
    m.baseline[1] = 0.002
    pts = []
    for _ in range(40):
        uL, vL, Z = rng.uniform(200, 2400), rng.uniform(200, 1700), rng.uniform(1, 6)
        uR, vR, _ = m.project_left_point_to_right(uL, vL, Z)
        pts.append([uL, vL, uR[0], vR[0], Z])
    d, w = linear_init(np.array(pts), S)
    assert np.allclose(d, m.baseline[:2], rtol=0.02, atol=2e-4)
    assert np.allclose(w, m.rotation, atol=5e-4)


def test_training_recovers_a_distorted_rig():
    """Γ grid + rotation fits a Brown–Conrady rig it was not derived from (held-out MAPE < 1.5 %)."""
    rig, rng = SyntheticRig(), np.random.default_rng(7)
    train_pts, _ = rig.sample(60, rng, click_sigma=0.5)
    test_pts, _ = rig.sample(300, rng, click_sigma=0.5)
    model, hist = train(train_pts, TrainConfig(log_every=500), log=None)
    assert hist[-1][1] < hist[0][1]
    held_out = metrics(model.triangulate(*test_pts[:, :4].T)[0], test_pts[:, 4])
    assert held_out["mape_pct"] < 1.5
    # and clearly beats the pinhole-with-offset baseline on the same data
    a, b = fit_disparity_offset(train_pts)
    base = metrics(a / (test_pts[:, 0] - test_pts[:, 2] - b), test_pts[:, 4])
    assert held_out["mape_pct"] < 0.5 * base["mape_pct"]


def test_training_is_deterministic(real_points):
    m1, _ = train(real_points, TrainConfig(epochs=200), log=None)
    m2, _ = train(real_points, TrainConfig(epochs=200), log=None)
    assert np.array_equal(m1.gamma_R, m2.gamma_R)


def test_centre_node_of_left_grid_is_frozen(real_points):
    m, _ = train(real_points, TrainConfig(epochs=200), log=None)
    assert np.allclose(m.gamma_L[S.center_node], S.gamma0)


def test_postcorrection_fit_is_exact_on_quadratic():
    z = np.linspace(0.5, 5, 20)
    assert np.allclose(fit_postcorrection(z, 0.1 + 0.9 * z + 0.02 * z * z), [0.1, 0.9, 0.02])


def test_too_few_points():
    with pytest.raises(ValueError):
        train(np.ones((2, 5)))
