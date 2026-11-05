"""Pure-gamma lens model.

A pixel ``(u, v)`` is mapped to a 3-D ray direction

    r = [u_n * g,  v_n * g,  1]        u_n = (u - cx) / cx,  v_n = (v - cy) / cy

where ``g = bilinear(u, v, Gamma)`` is read from a small trainable grid
(5x5 by default).  A uniform grid ``g = cx / f`` is exactly a pinhole camera
with focal length ``f``; letting the nodes vary models lens distortion and
focal-length anisotropy without a polynomial distortion model.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_RIG, RigConfig

Weights = tuple[float, float, float, float]
NodeIndices = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def default_params(
    rig: RigConfig = DEFAULT_RIG, dx: float = 0.10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(GammaL, GammaR, d)`` initialised to the pinhole prior.

    Both grids are filled with ``rig.g0`` and the baseline is ``[dx, 0, 0]``.
    """
    gamma_l = np.full(rig.grid_shape, rig.g0, dtype=np.float64)
    gamma_r = np.full(rig.grid_shape, rig.g0, dtype=np.float64)
    d = np.array([dx, 0.0, 0.0], dtype=np.float64)
    return gamma_l, gamma_r, d


def bilinear(
    u: float, v: float, gamma: np.ndarray, rig: RigConfig = DEFAULT_RIG
) -> tuple[float, Weights, NodeIndices]:
    """Bilinear interpolation of the gamma grid at pixel ``(u, v)``.

    The pixel is mapped to continuous grid coordinates ``fx = u / cell_w``,
    ``fy = v / cell_h``; the enclosing cell is clamped to the grid so that
    pixels outside the image extrapolate from the border cell.

    Returns:
        ``(g, weights, node_indices)`` where ``g = sum(w_k * Gamma[idx_k])``.
        The weights and indices are what the analytic gradient needs
        (``dg / dGamma[idx_k] = w_k``).
    """
    fx, fy = u / rig.cell_w, v / rig.cell_h
    ix = min(max(int(fx), 0), rig.grid_cols - 2)
    iy = min(max(int(fy), 0), rig.grid_rows - 2)
    tx, ty = fx - ix, fy - iy
    w: Weights = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    idx: NodeIndices = ((iy, ix), (iy, ix + 1), (iy + 1, ix), (iy + 1, ix + 1))
    g = sum(w[k] * gamma[idx[k]] for k in range(4))
    return float(g), w, idx


def build_ray(
    u: float, v: float, gamma: np.ndarray, rig: RigConfig = DEFAULT_RIG
) -> tuple[np.ndarray, dict]:
    """Ray direction ``[u_n*g, v_n*g, 1]`` for pixel ``(u, v)``.

    Returns:
        ``(ray, info)``; ``info`` carries ``un, vn, g, w, idx`` for the
        chain rule in :func:`stereo_distance.calibration.forward_grads`.
    """
    un = (u - rig.cx) / rig.cx
    vn = (v - rig.cy) / rig.cy
    g, w, idx = bilinear(u, v, gamma, rig)
    info = {"u": u, "v": v, "un": un, "vn": vn, "g": g, "w": w, "idx": idx}
    return np.array([un * g, vn * g, 1.0]), info


def bilinear_batch(
    u: np.ndarray, v: np.ndarray, gamma: np.ndarray, rig: RigConfig = DEFAULT_RIG
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`bilinear` for arrays of pixels.

    Returns:
        ``(g, w, iy, ix)`` with ``g`` shape ``(N,)``, ``w`` shape ``(N, 4)`` and
        the four node rows/cols as ``(N, 4)`` integer arrays, in the same
        node order as the scalar version.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    fx, fy = u / rig.cell_w, v / rig.cell_h
    ix = np.clip(np.floor(fx).astype(int), 0, rig.grid_cols - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, rig.grid_rows - 2)
    tx, ty = fx - ix, fy - iy
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    rows = np.stack([iy, iy, iy + 1, iy + 1], axis=1)
    cols = np.stack([ix, ix + 1, ix, ix + 1], axis=1)
    g = (w * gamma[rows, cols]).sum(axis=1)
    return g, w, rows, cols


def build_rays_batch(
    u: np.ndarray, v: np.ndarray, gamma: np.ndarray, rig: RigConfig = DEFAULT_RIG
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`build_ray`.

    Returns:
        ``(rays, un, vn, w, rows, cols)`` where ``rays`` has shape ``(N, 3)``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    un = (u - rig.cx) / rig.cx
    vn = (v - rig.cy) / rig.cy
    g, w, rows, cols = bilinear_batch(u, v, gamma, rig)
    rays = np.stack([un * g, vn * g, np.ones_like(g)], axis=1)
    return rays, un, vn, w, rows, cols


def laplacian_grad(gamma: np.ndarray) -> np.ndarray:
    """Gradient of the discrete Laplacian smoothness penalty.

    Interior nodes receive ``4*G[i,j] - G[i-1,j] - G[i+1,j] - G[i,j-1] - G[i,j+1]``;
    border nodes are left untouched (Dirichlet boundary).  Adding this to the
    data gradient pulls each node towards the mean of its four neighbours, so
    the grid behaves like an elastic sheet - real lenses distort smoothly.
    """
    grad = np.zeros_like(gamma)
    grad[1:-1, 1:-1] = (
        4.0 * gamma[1:-1, 1:-1]
        - gamma[:-2, 1:-1]
        - gamma[2:, 1:-1]
        - gamma[1:-1, :-2]
        - gamma[1:-1, 2:]
    )
    return grad


def ray_to_pixel(
    target: np.ndarray,
    gamma: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
    iters: int = 5,
) -> tuple[float, float] | None:
    """Invert the lens model: find ``(u, v)`` whose ray equals ``target``.

    ``target`` must have a positive z component; it is first normalised to
    ``z = 1``.  Starting from the pinhole guess ``g = g0`` the gamma value is
    re-read at the current estimate and the pixel re-solved - a fixed-point
    iteration that reaches < 0.1 px in a handful of steps on a smooth grid.

    Returns:
        ``(u, v)`` or ``None`` if the point is behind the camera.
    """
    tgt = np.asarray(target, dtype=np.float64)
    if tgt[2] < 1e-6:
        return None
    tgt = tgt / tgt[2]
    u = rig.cx * float(tgt[0]) / rig.g0 + rig.cx
    v = rig.cy * float(tgt[1]) / rig.g0 + rig.cy
    for _ in range(iters):
        u = max(0.0, min(rig.img_w - 1.0, u))
        v = max(0.0, min(rig.img_h - 1.0, v))
        g, _, _ = bilinear(u, v, gamma, rig)
        if abs(g) < 1e-6:
            break
        u = rig.cx * float(tgt[0]) / g + rig.cx
        v = rig.cy * float(tgt[1]) / g + rig.cy
    return u, v


def predict_right(
    u_l: float,
    v_l: float,
    z: float,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
) -> tuple[float, float] | None:
    """Right-image pixel of the point seen at left pixel ``(u_l, v_l)`` at depth ``z``.

    The target right ray is ``r_L - d / z``; it is inverted through the right
    gamma grid with :func:`ray_to_pixel`.  Sampling ``z`` traces the exact,
    distorted epipolar curve the UI draws.
    """
    ray_l, _ = build_ray(u_l, v_l, gamma_l, rig)
    return ray_to_pixel(ray_l - np.asarray(d, dtype=np.float64) / z, gamma_r, rig)


def resize_grid(gamma: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a gamma grid to ``shape`` (for legacy 4x4 files)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    rows_in, cols_in = gamma.shape
    rows_out, cols_out = shape
    ys = np.linspace(0.0, rows_in - 1.0, rows_out)
    xs = np.linspace(0.0, cols_in - 1.0, cols_out)
    y0 = np.clip(np.floor(ys).astype(int), 0, rows_in - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, cols_in - 2)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    g00 = gamma[y0][:, x0]
    g10 = gamma[y0][:, x0 + 1]
    g01 = gamma[y0 + 1][:, x0]
    g11 = gamma[y0 + 1][:, x0 + 1]
    return (1 - tx) * (1 - ty) * g00 + tx * (1 - ty) * g10 + (1 - tx) * ty * g01 + tx * ty * g11
