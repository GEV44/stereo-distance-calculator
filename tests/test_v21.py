"""Tests for the v2.1 additions: radial prior, noise model, λ selection, ensemble, evaluation statistics."""

import numpy as np
import pytest

from stereo_gamma.baselines import BrownConradyStereo, brown_distort
from stereo_gamma.config import DEFAULT_SENSOR as S
from stereo_gamma.config import TrainConfig
from stereo_gamma.evaluation import bootstrap_ci, coverage, metrics, signflip_test
from stereo_gamma.model import StereoModel
from stereo_gamma.solver import radial_basis, regulariser_hessian
from stereo_gamma.synthetic import Camera, SyntheticRig
from stereo_gamma.training import estimate_noise, noise_scales, objective, select_smoothing, train

FAST = dict(auto_smooth=False, ensemble=False)


def test_objective_gradient_with_radial_prior_and_noise_scales(real_points, rng):
    th = {"gL": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
          "gR": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
          "d": np.array([0.2, 0.01]), "w": np.array([0.01, -0.02, 0.03]), "a": np.array([0.1, -0.3, 0.2, 0.5])}
    pts = np.c_[real_points, noise_scales(real_points[:, 4], 1.0, 3.0)]
    cfg = TrainConfig(anchor=0.1, smooth=0.05, huber_delta=0.03)
    _, g = objective(th, pts, cfg)
    h = 1e-6
    for k in th:
        for idx in np.ndindex(th[k].shape):
            tp = {a: b.copy() for a, b in th.items()}
            tm = {a: b.copy() for a, b in th.items()}
            tp[k][idx] += h
            tm[k][idx] -= h
            num = (objective(tp, pts, cfg, need_grad=False)[0] - objective(tm, pts, cfg, need_grad=False)[0]) / (2 * h)
            assert num == pytest.approx(g[k][idx], rel=1e-4, abs=1e-9), (k, idx)


def test_radial_prior_is_exactly_quadratic():
    """The regulariser vanishes on a Γ that equals its radial prior field, for any coefficients."""
    cfg = TrainConfig()
    H = regulariser_hessian(S, cfg)
    B = radial_basis(S)
    a = np.array([0.3, -0.1])
    g = np.tile(S.gamma0, 25) + B @ a
    x = np.zeros(H.shape[0])
    x[:50], x[50:100] = g, g
    x[105:107], x[107:109] = a, a
    x0 = np.zeros_like(x)
    x0[:50] = x0[50:100] = np.tile(S.gamma0, 25)
    dv = x - x0
    assert 0.5 * dv @ H @ dv == pytest.approx(0.5 * 1e-8 * 2 * a @ a, abs=1e-12)
    assert np.allclose(H, H.T) and np.linalg.eigvalsh(H).min() > -1e-10


def test_noise_model_recovers_true_noise():
    rig, rng = SyntheticRig(), np.random.default_rng(3)
    pts, _ = rig.sample(400, rng, click_sigma=2.0, label_sigma=0.02)
    m, _ = train(pts, TrainConfig(noise_model="relative", **FAST), log=None)
    th = {"gL": m.gamma_L, "gR": m.gamma_R, "d": m.baseline[:2], "w": m.rotation}
    nm = estimate_noise(th, pts, S)
    assert nm["sigma_label"] == pytest.approx(0.02, rel=0.35)
    # σ_px is in the nominal-focal gauge; true click σ = 2 px at f ≈ 2750–2790
    assert nm["sigma_px"] == pytest.approx(2.0, rel=0.35)


def test_smoothing_selection_adapts_to_data():
    """Few noisy points → strong smoothing; many clean points → weak smoothing."""
    rig = SyntheticRig()
    small, _ = rig.sample(20, np.random.default_rng(1), click_sigma=3.0, label_sigma=0.03)
    big, _ = rig.sample(400, np.random.default_rng(2), click_sigma=0.2)
    lam_small = select_smoothing(small, TrainConfig(cv_repeats=1), S)[0]
    lam_big = select_smoothing(big, TrainConfig(cv_repeats=1), S)[0]
    assert lam_big < lam_small


def test_brown_conrady_baseline_recovers_its_own_model():
    rig = SyntheticRig(Camera(2750.0, 1310.0, 960.0, -0.1, 0.04, 3e-4, -2e-4),
                       Camera(2790.0, 1285.0, 985.0, -0.08, 0.03, -1e-4, 4e-4))
    tr, _ = rig.sample(150, np.random.default_rng(0), click_sigma=0.0)
    te, _ = rig.sample(200, np.random.default_rng(1), click_sigma=0.0)
    bc = BrownConradyStereo().fit(tr, noise_model="relative")
    assert metrics(bc.triangulate(*te[:, :4].T), te[:, 4])["mape_pct"] < 0.05
    assert bc.p[6] == pytest.approx(2790 * S.f_init / 2750, rel=0.01)  # f_R in the f_L = f₀ gauge


def test_brown_distort_matches_opencv_formula():
    x, y, k1, k2, p1, p2 = 0.3, -0.2, -0.1, 0.02, 1e-3, -2e-3
    r2 = x * x + y * y
    ex = x * (1 + k1 * r2 + k2 * r2 ** 2) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    ey = y * (1 + k1 * r2 + k2 * r2 ** 2) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    assert np.allclose(brown_distort(x, y, k1, k2, p1, p2), (ex, ey))


def test_ensemble_is_the_average_and_survives_save_load(tmp_path, real_points):
    m, _ = train(real_points, TrainConfig(auto_smooth=False), log=None)
    q = real_points[:, :4].T
    zg, zb = m.triangulate_gamma(*q)[0], m.companion_model().triangulate(*q)
    assert np.allclose(m.triangulate(*q)[0], 0.5 * (zg + zb))
    assert np.allclose(m.triangulate(*q, ensemble=False)[0], zg)
    m.save(tmp_path / "c.json")
    m2 = StereoModel.load(tmp_path / "c.json")
    assert np.allclose(m2.triangulate(*q)[0], m.triangulate(*q)[0])


def test_oracle_is_exact_on_clean_clicks():
    rig = SyntheticRig.freeform()
    _, clean = rig.sample(100, np.random.default_rng(0), click_sigma=0.0)
    assert np.allclose(rig.triangulate(*clean[:, :4].T), clean[:, 4], rtol=1e-9)


def test_statistics_helpers():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    c = coverage(z, np.ones_like(z), np.zeros_like(z))
    assert c["within_1sigma_pct"] == pytest.approx(68.3, abs=1.5) and c["rms_z"] == pytest.approx(1, abs=0.03)
    a = rng.normal(0, 1, 40)
    assert signflip_test(a, a)["p_value"] == 1.0
    assert signflip_test(a * 0.1, a * 3)["p_value"] < 0.01
    lo, hi = bootstrap_ci(rng.normal(5, 1, 400))
    assert lo < 5 < hi and hi - lo < 0.3
