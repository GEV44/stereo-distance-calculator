"""
Reference models for the evaluation.

* :class:`BrownConradyStereo` — the classical *parametric* lens model used by
  OpenCV / Zhang calibration: per camera a focal length, principal point,
  radial (k₁, k₂) and tangential (p₁, p₂) distortion, plus the same rig
  rotation and baseline.  It is
  fitted with **exactly the same residuals, robust loss and noise model** as
  the Γ-grid model, so a comparison isolates the lens representation.
* :func:`fit_disparity_offset` — the 2-parameter pinhole-with-offset model.

The Brown–Conrady fit uses Levenberg–Marquardt with a forward-difference
Jacobian (18 parameters — analytic derivatives are not needed for a baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_SENSOR, Sensor
from .geometry import rotation

# parameter layout (18):
#   0-5   left : cx, cy, k1, k2, p1, p2          (f_L fixed = nominal focal: scale gauge)
#   6-12  right: f, cx, cy, k1, k2, p1, p2
#   13-15 rotation (pitch, yaw, roll)   16-17 baseline dx, dy
_K = [2, 3, 4, 5, 9, 10, 11, 12]  # distortion coefficients (tiny ridge)


def brown_distort(x, y, k1, k2, p1, p2):
    """OpenCV / Brown–Conrady forward model on normalised coordinates."""
    r2 = x * x + y * y
    k = 1 + k1 * r2 + k2 * r2 * r2
    return (x * k + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * k + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def _undistort(u, v, f, cx, cy, k1, k2, p1, p2, iters: int = 15):
    xd, yd = (np.asarray(u, float) - cx) / f, (np.asarray(v, float) - cy) / f
    x, y = xd.copy(), yd.copy()
    with np.errstate(all="ignore"):  # trial LM steps may diverge → NaN, rejected by the solver
        for _ in range(iters):  # fixed point  x ← x + (x_d − distort(x))
            dx_, dy_ = brown_distort(x, y, k1, k2, p1, p2)
            x, y = np.clip(x + (xd - dx_), -5, 5), np.clip(y + (yd - dy_), -5, 5)
    return np.stack([x, y, np.ones_like(x)], axis=1)


def _huber_w(e, delta):
    a = np.abs(e)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))


@dataclass
class BrownConradyStereo:
    sensor: Sensor = DEFAULT_SENSOR
    p: np.ndarray = field(default_factory=lambda: np.zeros(18))
    huber_delta: float = 0.05
    weight_y: float = 0.5
    ridge_k: float = 1e-4  # tiny ridge on the distortion coefficients (numerical safety, few points)

    def _init(self, pts):
        from .training import linear_init

        s = self.sensor
        d, w = linear_init(pts, s)
        return np.array([s.cx, s.cy, 0, 0, 0, 0, s.f_init, s.cx, s.cy, 0, 0, 0, 0, *w, *d], float)

    def rays(self, p, uL, vL, uR, vR):
        s = self.sensor
        rL = _undistort(uL, vL, s.f_init, *p[0:6])
        rR = _undistort(uR, vR, *p[6:13])
        W = rR @ rotation(p[13:16]).T
        return rL, W / W[:, 2:3]

    def _residuals(self, p, pts, scales):
        uL, vL, uR, vR, Z = pts[:, :5].T
        rL, rR = self.rays(p, uL, vL, uR, vR)
        dx, dy = max(p[16], 1e-3), p[17]
        ex = (rL[:, 0] - rR[:, 0]) * Z / dx - 1
        ey = (rL[:, 1] - rR[:, 1]) * Z / dx - dy / dx
        return np.concatenate([scales[:, 0] * ex, math.sqrt(self.weight_y) * scales[:, 1] * ey])

    def _objective(self, p, pts, scales):
        r = self._residuals(p, pts, scales)
        if not np.all(np.isfinite(r)):
            return float("inf")
        a = np.abs(r)
        rho = np.where(a <= self.huber_delta, 0.5 * r * r, self.huber_delta * (a - 0.5 * self.huber_delta))
        return float(rho.sum()) + 0.5 * self.ridge_k * len(pts) * float(p[_K] @ p[_K])

    def _lm(self, p, pts, scales, iters=100):
        step = np.full(18, 1e-7)
        step[[0, 1, 6, 7, 8]] = 1e-3
        step[_K] = 1e-6
        F, mu = self._objective(p, pts, scales), 1e-3
        ridge = np.zeros(18)
        ridge[_K] = self.ridge_k * len(pts)
        for _ in range(iters):
            r = self._residuals(p, pts, scales)
            J = np.empty((len(r), 18))
            for j in range(18):  # central differences
                q1, q2 = p.copy(), p.copy()
                q1[j] += step[j]
                q2[j] -= step[j]
                J[:, j] = (self._residuals(q1, pts, scales) - self._residuals(q2, pts, scales)) / (2 * step[j])
            w = _huber_w(r, self.huber_delta)
            A = J.T @ (w[:, None] * J) + np.diag(ridge)
            g = J.T @ (w * r) + ridge * p
            done = False
            while mu < 1e12:
                pn = p - np.linalg.solve(A + mu * np.diag(np.maximum(np.diag(A), 1e-12)), g)
                Fn = self._objective(pn, pts, scales)
                if np.isfinite(Fn) and Fn <= F:
                    done = (F - Fn) < 1e-13 * max(F, 1e-300)
                    p, F, mu = pn, Fn, max(mu * 0.3, 1e-12)
                    break
                mu *= 5
            else:
                break
            if done:
                break
        return p

    def fit(self, pts, noise_model: str = "fgls"):
        """Relative fit, then (optionally) the same FGLS noise re-weighting as the Γ model."""
        from .training import noise_scales

        pts = np.asarray(pts, float)[:, :5]
        ones = np.ones((len(pts), 2))
        self.p = self._lm(self._init(pts), pts, ones)
        if noise_model == "fgls":
            Z = pts[:, 4]
            r = self._residuals(self.p, pts, ones)
            ex, ey = r[:len(pts)], r[len(pts):] / math.sqrt(self.weight_y)
            c = max(1.4826 * float(np.median(np.abs(ey / Z))), 1e-6)
            sl2 = max((1.4826 * float(np.median(np.abs(ex)))) ** 2 - c * c * float(np.median(Z * Z)), 0.002 ** 2)
            scales = noise_scales(Z, math.sqrt(sl2) / c, float(np.sqrt(np.mean(Z * Z))))
            self.p = self._lm(self.p, pts, scales)
        return self

    def triangulate(self, uL, vL, uR, vR):
        uL, vL, uR, vR = (np.atleast_1d(np.asarray(a, float)) for a in (uL, vL, uR, vR))
        rL, rR = self.rays(self.p, uL, vL, uR, vR)
        d = np.array([self.p[16], self.p[17], 0.0])
        a1, a2 = rL, -rR
        s11, s12, s22 = (a1 * a1).sum(1), (a1 * a2).sum(1), (a2 * a2).sum(1)
        b1, b2 = a1 @ d, a2 @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = (s22 * b1 - s12 * b2) / (s11 * s22 - s12 * s12)
        return np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)


def loocv_brown(points, noise_model: str = "fgls"):
    pts = np.asarray(points, float)
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        m = BrownConradyStereo().fit(np.delete(pts, i, axis=0), noise_model)
        out[i] = m.triangulate(*pts[i:i + 1, :4].T)[0]
    return out
