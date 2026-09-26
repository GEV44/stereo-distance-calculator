"""
Faithful, vectorised re-implementation of the **v1** calibration algorithm
(the original ``run_math.py``), kept only as a reference for the ablation
study.  It reproduces v1 exactly, including its quirks:

* one γ per node shared by x and y  (ray_y = ṽ·γ ⇒ wrong aspect CX/CY),
* no relative rotation between the cameras,
* plain gradient descent on squared (non-robust) errors,
* interior-only Laplacian "gradient", both grid centres frozen,
* ``best_state`` stores the parameters *after* the step of the best epoch.

``tests/test_legacy.py`` checks it against the original per-point code.
"""

from __future__ import annotations

import math

import numpy as np

from .config import DEFAULT_SENSOR, DX_MIN, G_MAX, G_MIN
from .geometry import rays
from .model import StereoModel

LEGACY_SMOOTH_W = 1e-2


def _laplacian_interior(G):
    g = np.zeros_like(G)
    g[1:-1, 1:-1] = 4 * G[1:-1, 1:-1] - G[:-2, 1:-1] - G[2:, 1:-1] - G[1:-1, :-2] - G[1:-1, 2:]
    return g


def train_legacy(points, epochs: int = 9000) -> StereoModel:
    s = DEFAULT_SENSOR
    pts = np.asarray(points, float)
    uL, vL, uR, vR, Z = pts.T
    N = len(pts)
    G0 = s.cx / s.f_init
    GL = np.full((s.grid_rows, s.grid_cols), G0)
    GR = GL.copy()
    d = np.array([0.10, 0.0, 0.0])
    d[0] = max(float(np.mean(Z)) * float(np.mean(np.abs(uL - uR))) * (G0 / s.cx), DX_MIN)
    lr_g, lr_d = 1e-1, 1e-3
    c = s.center_node
    best_loss, best = math.inf, None
    nodes = s.grid_rows * s.grid_cols
    for _ in range(epochs):
        rL, cL = rays(s, np.stack([GL, GL], -1), uL, vL)
        rR, cR = rays(s, np.stack([GR, GR], -1), uR, vR)
        denom = rL[:, 0] - rR[:, 0]
        live = np.abs(denom) >= 1e-12
        dx = max(float(d[0]), DX_MIN)
        rel = denom * Z / dx - 1.0
        ey = (rL[:, 1] - rR[:, 1]) - d[1] / Z
        loss = np.where(live, 0.5 * rel ** 2 + 0.5 * ey ** 2, 0.0)
        gx = np.where(live, rel * Z / dx, 0.0)
        gy = np.where(live, ey, 0.0)
        gd = np.array([np.sum(np.where(live, rel * (-denom * Z / dx ** 2), 0.0)),
                       np.sum(np.where(live, -ey / Z, 0.0)), 0.0]) / N
        sL, sR = gx * cL["un"] + gy * cL["vn"], -gx * cR["un"] - gy * cR["vn"]
        aL, aR = np.zeros(nodes), np.zeros(nodes)
        np.add.at(aL, cL["idx"].ravel(), (sL[:, None] * cL["w"]).ravel())
        np.add.at(aR, cR["idx"].ravel(), (sR[:, None] * cR["w"]).ravel())
        aL = aL.reshape(GL.shape) / N + LEGACY_SMOOTH_W * _laplacian_interior(GL)
        aR = aR.reshape(GR.shape) / N + LEGACY_SMOOTH_W * _laplacian_interior(GR)
        aL[c] = aR[c] = 0.0
        for g in (aL, aR, gd):
            n = float(np.linalg.norm(g))
            if n > 1e3:
                g *= 1e3 / n
        GL, GR, d = GL - lr_g * aL, GR - lr_g * aR, d - lr_d * gd
        d[2] = 0.0
        d[0] = max(d[0], DX_MIN)
        np.clip(GL, G_MIN, G_MAX, out=GL)
        np.clip(GR, G_MIN, G_MAX, out=GR)
        GL[c] = GR[c] = G0
        total = float(loss.sum())
        if total < best_loss:
            best_loss, best = total, (GL.copy(), GR.copy(), d.copy())
    GL, GR, d = best
    return StereoModel(np.stack([GL, GL], -1), np.stack([GR, GR], -1), d, np.zeros(3), s,
                       meta={"trained": True, "model": "legacy-v1"})


def loocv_legacy(points, epochs: int = 9000):
    pts = np.asarray(points, float)
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        m = train_legacy(np.delete(pts, i, axis=0), epochs)
        uL, vL, uR, vR = pts[i, :4]
        if uL > uR:  # v1 rejected non-positive pixel disparity
            out[i] = m.depth(uL, vL, uR, vR)
    return out
