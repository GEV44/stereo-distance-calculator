import numpy as np
import pytest

from stereo_gamma.config import DEFAULT_SENSOR as S
from stereo_gamma.config import TrainConfig
from stereo_gamma.geometry import membrane_energy_grad
from stereo_gamma.solver import (
    diagnostics,
    free_mask,
    graph_laplacian,
    pack,
    residual_jacobian,
    unpack,
)
from stereo_gamma.training import _forward, objective, train


def _theta(rng):
    return {"gL": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
            "gR": S.gamma0 + 0.02 * rng.standard_normal((5, 5, 2)),
            "d": np.array([0.2, 0.01]), "w": np.array([0.01, -0.02, 0.03])}


def test_residual_jacobian_matches_finite_differences(real_points, rng):
    th = _theta(rng)
    _, _, Jx, Jy = residual_jacobian(th, real_points, S)
    x, h = pack(th), 1e-7
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        a = _forward(unpack(xp, S), real_points, S)[:2]
        b = _forward(unpack(xm, S), real_points, S)[:2]
        assert np.allclose((a[0] - b[0]) / (2 * h), Jx[:, j], rtol=1e-4, atol=1e-6)
        assert np.allclose((a[1] - b[1]) / (2 * h), Jy[:, j], rtol=1e-4, atol=1e-6)


def test_pack_roundtrip(rng):
    th = _theta(rng)
    back = unpack(pack(th), S)
    assert all(np.array_equal(th[k], back[k]) for k in th)


def test_graph_laplacian_equals_membrane_energy(rng):
    G = rng.normal(size=(5, 5))
    L = graph_laplacian(5, 5)
    assert np.isclose(0.5 * G.ravel() @ L @ G.ravel(), membrane_energy_grad(G[..., None])[0])


FIXED = dict(auto_smooth=False, ensemble=False, noise_model="relative")


def test_lm_and_adam_reach_the_same_optimum(real_points):
    """Independent optimisers agree ⇒ both converge to the minimum of the same objective."""
    m_lm, hist = train(real_points, TrainConfig(solver="lm", **FIXED), log=None)
    m_ad, _ = train(real_points, TrainConfig(solver="adam", **FIXED), log=None)
    assert m_lm.meta["final_loss"] == pytest.approx(m_ad.meta["final_loss"], rel=1e-8)
    Z_lm = m_lm.triangulate(*real_points[:, :4].T)[0]
    Z_ad = m_ad.triangulate(*real_points[:, :4].T)[0]
    assert np.allclose(Z_lm, Z_ad, rtol=1e-5)
    assert all(b[1] <= a[1] + 1e-15 for a, b in zip(hist, hist[1:]))  # LM is monotone


def test_lm_final_gradient_vanishes(real_points):
    from stereo_gamma.solver import levenberg_marquardt
    from stereo_gamma.training import linear_init

    cfg = TrainConfig(**FIXED)
    d0, w0 = linear_init(real_points, S)
    g0 = np.broadcast_to(S.gamma0, (5, 5, 2)).copy()
    th, _ = levenberg_marquardt({"gL": g0, "gR": g0.copy(), "d": d0, "w": w0}, real_points, cfg, S)
    _, g = objective(th, real_points, cfg)
    gv = pack(g if isinstance(g, dict) else g)[free_mask(S, cfg)]
    assert np.max(np.abs(gv)) < 1e-7


def test_diagnostics(real_points):
    cfg = TrainConfig(**FIXED)
    m, _ = train(real_points, cfg, log=None)
    a = np.r_[m.meta["radial_prior"]["left"], m.meta["radial_prior"]["right"]]
    th = {"gL": m.gamma_L, "gR": m.gamma_R, "d": m.baseline[:2], "w": m.rotation, "a": a}
    d = diagnostics(th, real_points, cfg, S)
    # 2×50 Γ values + dx, dy + 3 angles + 4 radial coefficients = 109, − 2 gauge-fixed = 107 free
    assert 5 < d["df_eff"] < d["n_params_free"] and d["n_params_free"] == 107
    assert set(d["stderr"]) == {"dx", "dy", "pitch", "yaw", "roll"}
    assert all(0 < v < 0.1 for v in d["stderr"].values())
    # stronger regularisation ⇒ fewer effective degrees of freedom
    d2 = diagnostics(th, real_points, TrainConfig(smooth=30.0, **FIXED), S)
    assert d2["df_eff"] < d["df_eff"]
