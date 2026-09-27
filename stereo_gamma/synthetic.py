"""
Synthetic stereo rig with known ground truth.

The real dataset is small (27 hand-clicked points) and its labels are noisy,
so it cannot separate models whose true accuracy differs by less than the
label noise.  This module simulates a rig from a *different* model family
than the one we fit — Brown–Conrady radial distortion, off-centre principal
points, unequal focal lengths and a small relative rotation — so we can
measure how well the Γ grid approximates a real lens as the number of
calibration points and the click noise vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_SENSOR, Sensor
from .geometry import rotation


@dataclass
class Camera:
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0  # tangential (decentering) distortion
    p2: float = 0.0
    wave: float = 0.0  # non-parametric "moulded lens" waviness (normalised units)

    def distort(self, x, y):
        r2 = x * x + y * y
        k = 1 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * k + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * k + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        if self.wave:
            xd = xd + self.wave * np.sin(7.0 * x + 2.0 * y + 0.4)
            yd = yd + self.wave * np.cos(6.0 * y - 3.0 * x + 1.1)
        return xd, yd

    def project(self, P: np.ndarray):
        xd, yd = self.distort(P[:, 0] / P[:, 2], P[:, 1] / P[:, 2])
        return self.f * xd + self.cx, self.f * yd + self.cy

    def unproject(self, u, v, iters: int = 12):
        """Pixel → undistorted normalised ray (fixed-point undistortion)."""
        xd, yd = (np.asarray(u) - self.cx) / self.f, (np.asarray(v) - self.cy) / self.f
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            dx, dy = self.distort(x, y)
            x, y = x + (xd - dx), y + (yd - dy)
        return np.stack([x, y, np.ones_like(x)], axis=1)


@dataclass
class SyntheticRig:
    left: Camera = field(default_factory=lambda: Camera(2750.0, 1310.0, 960.0, -0.10, 0.04))
    right: Camera = field(default_factory=lambda: Camera(2790.0, 1285.0, 985.0, -0.08, 0.03))
    baseline: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.004, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.radians([0.15, 0.9, -1.1]))
    sensor: Sensor = DEFAULT_SENSOR

    @classmethod
    def freeform(cls) -> SyntheticRig:
        """
        Same rig, but each lens also has decentering (p₁, p₂) and a smooth
        non-radial waviness of ±0.0015 normalised units (≈ ±4 px) — the kind of
        field a moulded lens or tilted sensor produces and no low-order
        parametric model describes.
        """
        return cls(Camera(2750.0, 1310.0, 960.0, -0.10, 0.04, 4e-4, -3e-4, 1.5e-3),
                   Camera(2790.0, 1285.0, 985.0, -0.08, 0.03, -2e-4, 5e-4, 1.5e-3))

    def scaled(self, k: float) -> SyntheticRig:
        """Same rig at ``k``× the resolution (fast rendering for tests)."""
        def cam(c):
            return Camera(c.f * k, c.cx * k, c.cy * k, c.k1, c.k2, c.p1, c.p2, c.wave)
        s = self.sensor
        return SyntheticRig(cam(self.left), cam(self.right), self.baseline.copy(), self.omega.copy(),
                            Sensor(round(s.width * k), round(s.height * k), s.grid_rows, s.grid_cols,
                                   s.f_init * k))

    def triangulate(self, uL, vL, uR, vR):
        """
        *Oracle* depth: OLS triangulation with the true cameras and rig.  On
        noisy clicks this is the error floor no calibration can beat.
        """
        rL = self.left.unproject(np.asarray(uL, float), np.asarray(vL, float))
        W = self.right.unproject(np.asarray(uR, float), np.asarray(vR, float)) @ rotation(self.omega).T
        rR = W / W[:, 2:3]
        a1, a2, d = rL, -rR, self.baseline
        s11, s12, s22 = (a1 * a1).sum(1), (a1 * a2).sum(1), (a2 * a2).sum(1)
        return (s22 * (a1 @ d) - s12 * (a2 @ d)) / (s11 * s22 - s12 * s12)

    def sample(self, n: int, rng: np.random.Generator, z_range=(0.5, 6.0),
               click_sigma: float = 1.0, label_sigma: float = 0.0, margin: int = 20):
        """
        Draw ``n`` calibration points ``[uL, vL, uR, vR, Z]`` visible in both
        images.  Depth is uniform in inverse depth (like disparity).  Clicks
        get Gaussian pixel noise; labels get relative Gaussian noise.
        """
        s, R = self.sensor, rotation(self.omega)
        out = []
        while sum(len(o) for o in out) < n:
            m = 4 * n
            uL = rng.uniform(margin, s.width - margin, m)
            vL = rng.uniform(margin, s.height - margin, m)
            Z = 1.0 / rng.uniform(1 / z_range[1], 1 / z_range[0], m)
            P = self.left.unproject(uL, vL) * Z[:, None]
            PR = (P - self.baseline) @ R  # Rᵀ(P − d)
            uR, vR = self.right.project(PR)
            ok = ((PR[:, 2] > 0) & (uR > margin) & (uR < s.width - margin)
                  & (vR > margin) & (vR < s.height - margin))
            out.append(np.stack([uL, vL, uR, vR, Z], axis=1)[ok])
        pts = np.concatenate(out)[:n]
        clean = pts.copy()
        pts[:, :4] += rng.normal(0, click_sigma, (n, 4))
        pts[:, 4] *= 1 + rng.normal(0, label_sigma, n)
        return pts, clean


# ─── RAY-TRACED DEMO SCENE ────────────────────────────────────────────────────
@dataclass
class Plane:
    """Fronto-parallel rectangle at depth ``z`` (``axis=2``) or a floor ``y = h`` (``axis=1``)."""

    axis: int
    value: float
    x_range: tuple = (-np.inf, np.inf)
    y_range: tuple = (-np.inf, np.inf)
    z_range: tuple = (-np.inf, np.inf)
    albedo: float = 1.0
    seed: int = 0


DEMO_SCENE = (
    Plane(2, 6.0, albedo=0.85, seed=1),                                          # back wall
    Plane(1, 0.95, z_range=(0.5, 6.0), albedo=0.6, seed=2),                       # floor
    Plane(2, 3.5, x_range=(0.35, 1.45), y_range=(-0.9, 0.25), albedo=1.0, seed=3),  # panel
    Plane(2, 2.0, x_range=(-0.75, 0.15), y_range=(-0.35, 0.55), albedo=1.1, seed=4),  # box face
)


def _value_noise(a, b, seed):
    """Multi-octave value noise evaluated at plane coordinates (metres)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(a)
    for scale, amp in ((0.25, 0.45), (0.06, 0.35), (0.015, 0.2)):
        n = 512
        grid = rng.uniform(-1, 1, (n + 1, n + 1))
        grid[n, :], grid[:, n] = grid[0, :], grid[:, 0]  # periodic → no seam where coordinates wrap
        x, y = (a / scale) % n, (b / scale) % n
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        out += amp * ((1 - fx) * (1 - fy) * grid[y0, x0] + fx * (1 - fy) * grid[y0, x0 + 1]
                      + (1 - fx) * fy * grid[y0 + 1, x0] + fx * fy * grid[y0 + 1, x0 + 1])
    return out


def render(rig: SyntheticRig, which: str, scene=DEMO_SCENE, step: int = 1, noise: float = 1.5,
           seed: int = 0):
    """
    Ray-trace one camera of the rig.  Returns ``(gray uint8 (H, W), depth (H, W))``
    where ``depth`` is the left-frame Z of the surface seen by each pixel.
    """
    s = rig.sensor
    vv, uu = np.mgrid[0:s.height:step, 0:s.width:step].astype(np.float64)
    cam = rig.left if which == "left" else rig.right
    d = cam.unproject(uu.ravel(), vv.ravel())
    if which == "left":
        o = np.zeros(3)
    else:
        o, d = rig.baseline, d @ rotation(rig.omega).T
    best_t = np.full(len(d), np.inf)
    shade = np.zeros(len(d))
    zmap = np.full(len(d), np.nan)
    for pl in scene:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (pl.value - o[pl.axis]) / d[:, pl.axis]
            P = o + t[:, None] * d
        hit = ((t > 0) & (t < best_t)
               & (P[:, 0] >= pl.x_range[0]) & (P[:, 0] <= pl.x_range[1])
               & (P[:, 1] >= pl.y_range[0]) & (P[:, 1] <= pl.y_range[1])
               & (P[:, 2] >= pl.z_range[0]) & (P[:, 2] <= pl.z_range[1]))
        a, b = (P[hit, 0], P[hit, 1]) if pl.axis == 2 else (P[hit, 0], P[hit, 2])
        best_t[hit] = t[hit]
        shade[hit] = pl.albedo * (0.5 + 0.5 * _value_noise(a, b, pl.seed))
        zmap[hit] = P[hit, 2]
    img = 30 + 190 * np.clip(shade, 0, 1.2) / 1.2
    img += np.random.default_rng(seed + (which == "right")).normal(0, noise, img.shape)
    shape = vv.shape
    return np.clip(img, 0, 255).astype(np.uint8).reshape(shape), zmap.reshape(shape)


def scene_points(rig: SyntheticRig, n: int, rng: np.random.Generator, click_sigma: float = 0.7,
                 scene=DEMO_SCENE):
    """Calibration points on the demo scene's surfaces (what a user would click)."""
    pts = []
    s, R = rig.sensor, rotation(rig.omega)
    while len(pts) < n:
        m = max(4, s.width // 64)
        uL, vL = rng.uniform(m, s.width - m), rng.uniform(m, s.height - m)
        d = rig.left.unproject(np.array([uL]), np.array([vL]))[0]
        best, Z = np.inf, None
        for pl in scene:
            if abs(d[pl.axis]) < 1e-12:
                continue
            t = pl.value / d[pl.axis]
            P = t * d
            if (t > 0 and t < best and pl.x_range[0] <= P[0] <= pl.x_range[1]
                    and pl.y_range[0] <= P[1] <= pl.y_range[1] and pl.z_range[0] <= P[2] <= pl.z_range[1]):
                best, Z = t, P[2]
        if Z is None or Z > 5.5:
            continue
        P = d * Z
        uR, vR = rig.right.project(((P - rig.baseline) @ R)[None, :])
        if m < uR[0] < s.width - m and m < vR[0] < s.height - m:
            pts.append([uL, vL, uR[0], vR[0], Z])
    pts = np.array(pts)
    pts[:, :4] = np.round(pts[:, :4] + rng.normal(0, click_sigma, (n, 4)))
    return pts
