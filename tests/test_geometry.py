import numpy as np

from stereo_gamma.config import DEFAULT_SENSOR as S
from stereo_gamma.geometry import (
    bilinear_weights,
    interpolate_gamma,
    membrane_energy_grad,
    resample_grid,
    rotation,
    rotation_with_jacobians,
)


def test_bilinear_partition_of_unity(rng):
    u, v = rng.uniform(0, S.width - 1, 500), rng.uniform(0, S.height - 1, 500)
    w, idx = bilinear_weights(S, u, v)
    assert np.allclose(w.sum(1), 1.0)
    assert (w >= -1e-12).all() and idx.min() >= 0 and idx.max() < S.grid_rows * S.grid_cols


def test_bilinear_exact_at_nodes_and_linear_fields(rng):
    G = rng.uniform(0.3, 0.6, (S.grid_rows, S.grid_cols, 2))
    for i in range(S.grid_rows - 1):
        for j in range(S.grid_cols - 1):
            w, idx = bilinear_weights(S, [j * S.cell_w], [i * S.cell_h])
            assert np.allclose(interpolate_gamma(G, w, idx)[0], G[i, j])
    # a field linear in (u, v) is reproduced exactly everywhere
    jj, ii = np.meshgrid(np.arange(S.grid_cols), np.arange(S.grid_rows))
    lin = np.stack([0.4 + 0.01 * jj - 0.02 * ii, 0.3 + 0.03 * ii], -1)
    u, v = rng.uniform(0, S.width, 100), rng.uniform(0, S.height, 100)
    w, idx = bilinear_weights(S, u, v)
    expect = np.stack([0.4 + 0.01 * u / S.cell_w - 0.02 * v / S.cell_h, 0.3 + 0.03 * v / S.cell_h], 1)
    assert np.allclose(interpolate_gamma(lin, w, idx), expect)


def test_rotation_orthonormal_and_jacobians():
    omega = np.array([0.03, -0.02, 0.05])
    R, dR = rotation_with_jacobians(omega)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    h = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        assert np.allclose((rotation(omega + e) - rotation(omega - e)) / (2 * h), dR[k], atol=1e-8)
    # small-angle limit: R ≈ I + [ω]×
    w = np.array([1e-4, 2e-4, -3e-4])
    skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    assert np.allclose(rotation(w), np.eye(3) + skew, atol=1e-7)


def test_membrane_gradient_matches_energy(rng):
    G = rng.normal(size=(5, 5, 2))
    E, g = membrane_energy_grad(G)
    h = 1e-6
    for idx in [(0, 0, 0), (2, 2, 1), (4, 3, 0), (1, 4, 1)]:
        Gp, Gm = G.copy(), G.copy()
        Gp[idx] += h
        Gm[idx] -= h
        num = (membrane_energy_grad(Gp)[0] - membrane_energy_grad(Gm)[0]) / (2 * h)
        assert np.isclose(num, g[idx], atol=1e-6)
    assert membrane_energy_grad(np.full((5, 5, 2), 0.45))[0] == 0.0


def test_resample_grid():
    g = np.arange(16.0).reshape(4, 4)
    assert np.allclose(resample_grid(g, 4, 4), g)
    up = resample_grid(g, 7, 7)
    assert np.allclose(up[::2, ::2], g) and np.isclose(up[1, 1], g[:2, :2].mean())
