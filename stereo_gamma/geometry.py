"""
Low-level geometry of the Γ-grid camera model (vectorised NumPy).

    ray(u, v) = [ ũ · γx(u, v),  ṽ · γy(u, v),  1 ]ᵀ

where γx, γy are bilinearly interpolated from a ``rows × cols × 2`` grid Γ.
In a perfect pinhole camera γx = CX/fx and γy = CY/fy everywhere; letting Γ
vary over the image absorbs lens distortion without any polynomial model.

The right camera's ray is rotated into the left frame with R(ω) and
perspective-normalised back to z = 1 ("rectification in ray space"):

    w  = R(ω) · ray_R ,     r̂_R = w / w_z
"""

from __future__ import annotations

import numpy as np

from .config import Sensor


# ─── BILINEAR Γ LOOKUP ────────────────────────────────────────────────────────
def bilinear_weights(sensor: Sensor, u, v):
    """
    Bilinear weights of the four grid nodes surrounding each pixel.

    Returns ``(w, idx)`` with shape ``(N, 4)``: weights sum to one, ``idx`` are
    flat node indices (``row * cols + col``).  The cell index is clamped to the
    grid so pixels on the far border use the last cell (linear extrapolation).
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    fx, fy = u / sensor.cell_w, v / sensor.cell_h
    ix = np.clip(np.floor(fx).astype(np.int64), 0, sensor.grid_cols - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, sensor.grid_rows - 2)
    tx, ty = fx - ix, fy - iy
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    c = sensor.grid_cols
    base = iy * c + ix
    idx = np.stack([base, base + 1, base + c, base + c + 1], axis=1)
    return w, idx


def interpolate_gamma(gamma: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """γ(u, v) for every pixel → ``(N, 2)`` array ``[γx, γy]``."""
    flat = gamma.reshape(-1, gamma.shape[-1])
    return np.einsum("nk,nkc->nc", w, flat[idx])


def normalize(sensor: Sensor, u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (u - sensor.cx) / sensor.cx, (v - sensor.cy) / sensor.cy


def rays(sensor: Sensor, gamma: np.ndarray, u, v):
    """
    Build rays for arrays of pixels.

    Returns ``(r, cache)`` where ``r`` is ``(N, 3)`` and ``cache`` holds the
    intermediate quantities needed by the analytic gradients.
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    un, vn = normalize(sensor, u, v)
    w, idx = bilinear_weights(sensor, u, v)
    g = interpolate_gamma(gamma, w, idx)
    r = np.stack([un * g[:, 0], vn * g[:, 1], np.ones_like(un)], axis=1)
    return r, {"un": un, "vn": vn, "w": w, "idx": idx, "g": g}


# ─── ROTATION  R = Rz(roll) · Ry(yaw) · Rx(pitch) ────────────────────────────
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]])


def _ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), np.array([[-s, 0, c], [0, 0, 0], [-c, 0, -s]])


def _rz(g):
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.array([[-s, -c, 0], [c, -s, 0], [0, 0, 0]])


def rotation(omega) -> np.ndarray:
    """Rotation taking right-camera coordinates into the left-camera frame."""
    return rotation_with_jacobians(omega)[0]


def rotation_with_jacobians(omega):
    """
    ``R`` and its exact partial derivatives ``[∂R/∂ω_pitch, ∂R/∂ω_yaw, ∂R/∂ω_roll]``.

    For small angles R ≈ I + [ω]×, so yaw shifts every ray horizontally
    (a constant disparity offset) and roll couples x and y — two effects a
    purely multiplicative Γ grid cannot represent.
    """
    Rx, dRx = _rx(float(omega[0]))
    Ry, dRy = _ry(float(omega[1]))
    Rz, dRz = _rz(float(omega[2]))
    R = Rz @ Ry @ Rx
    return R, (Rz @ Ry @ dRx, Rz @ dRy @ Rx, dRz @ Ry @ Rx)


def right_rays_in_left(sensor: Sensor, gamma_R, omega, uR, vR):
    """
    Right-camera rays expressed in the left frame and normalised to z = 1.

    Returns ``(r_hat, w, r_R, cache)``; ``r_hat`` is ``(N, 3)`` with last column 1.
    """
    r_R, cache = rays(sensor, gamma_R, uR, vR)
    R = rotation(omega)
    w = r_R @ R.T
    r_hat = w / w[:, 2:3]
    return r_hat, w, r_R, cache


# ─── GRID UTILITIES ───────────────────────────────────────────────────────────
def resample_grid(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Bilinear resampling of a ``(r, c[, ch])`` node grid (corner-aligned)."""
    g = np.asarray(grid, dtype=np.float64)
    squeeze = g.ndim == 2
    if squeeze:
        g = g[..., None]
    r0, c0 = g.shape[:2]
    ys = np.linspace(0, r0 - 1, rows)
    xs = np.linspace(0, c0 - 1, cols)
    y0 = np.clip(np.floor(ys).astype(int), 0, max(r0 - 2, 0))
    x0 = np.clip(np.floor(xs).astype(int), 0, max(c0 - 2, 0))
    y1, x1 = np.minimum(y0 + 1, r0 - 1), np.minimum(x0 + 1, c0 - 1)
    ty, tx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    out = ((1 - ty) * (1 - tx) * g[y0][:, x0] + (1 - ty) * tx * g[y0][:, x1]
           + ty * (1 - tx) * g[y1][:, x0] + ty * tx * g[y1][:, x1])
    return out[..., 0] if squeeze else out


def membrane_energy_grad(G: np.ndarray):
    """
    Membrane (Dirichlet) smoothness energy ``E = ½ Σ_edges (G_i − G_j)²`` over
    the 4-connected grid, applied per channel, and its exact gradient — the
    graph Laplacian ``∇E = L·G`` (interior: 4G − Σ neighbours; borders use
    their true degree, so the gradient is consistent with the energy).
    """
    dh = G[:, 1:] - G[:, :-1]
    dv = G[1:, :] - G[:-1, :]
    E = 0.5 * (float((dh * dh).sum()) + float((dv * dv).sum()))
    grad = np.zeros_like(G)
    grad[:, 1:] += dh
    grad[:, :-1] -= dh
    grad[1:, :] += dv
    grad[:-1, :] -= dv
    return E, grad
