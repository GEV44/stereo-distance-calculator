"""Synthetic stereo rig: project known 3-D points through the gamma model.

Used by the tests and by ``benchmarks/synthetic_eval.py``.  The forward
projection inverts the lens model numerically (:func:`lens.ray_to_pixel`), so
for a uniform grid it reduces to the pinhole ``u = cx + f X / Z`` exactly and
for a distorted grid it is consistent with :func:`lens.build_ray` to well under
0.01 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RIG, RigConfig
from .lens import default_params, ray_to_pixel


@dataclass
class SyntheticRig:
    """Ground-truth parameters of a simulated stereo pair."""

    gamma_l: np.ndarray
    gamma_r: np.ndarray
    d: np.ndarray
    rig: RigConfig = DEFAULT_RIG

    @classmethod
    def pinhole(cls, focal_px: float | None = None, dx: float = 0.25, dy: float = 0.0,
                rig: RigConfig = DEFAULT_RIG) -> SyntheticRig:
        """Ideal pair of identical pinhole cameras (uniform gamma grids)."""
        gamma_l, gamma_r, d = default_params(rig, dx)
        if focal_px is not None:
            gamma_l[:] = rig.cx / focal_px
            gamma_r[:] = rig.cx / focal_px
            gamma_l[rig.centre] = rig.g0
            gamma_r[rig.centre] = rig.g0
        d[1] = dy
        return cls(gamma_l, gamma_r, d, rig)

    @classmethod
    def distorted(cls, amplitude: float = 0.08, dx: float = 0.25, dy: float = 0.0,
                  seed: int = 0, rig: RigConfig = DEFAULT_RIG) -> SyntheticRig:
        """Pinhole prior plus a smooth random perturbation of the gamma nodes.

        The centre node stays at ``g0`` (as the trainer assumes) and the
        perturbation is relative (``g0 * amplitude``), so ``amplitude=0.08``
        means up to about 8 % focal-scale variation across the frame.
        """
        rng = np.random.default_rng(seed)
        gamma_l, gamma_r, d = default_params(rig, dx)
        for g in (gamma_l, gamma_r):
            g += rig.g0 * amplitude * rng.uniform(-1.0, 1.0, size=g.shape)
            g[rig.centre] = rig.g0
        d[1] = dy
        return cls(gamma_l, gamma_r, d, rig)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ``(N, 3)`` points in the left camera frame to both images.

        Returns:
            ``(left_px, right_px)`` arrays of shape ``(N, 2)``; rows are ``nan``
            when a point projects behind a camera.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        left = np.full((len(pts), 2), np.nan)
        right = np.full((len(pts), 2), np.nan)
        for i, p in enumerate(pts):
            pl = ray_to_pixel(p, self.gamma_l, self.rig, iters=12)
            pr = ray_to_pixel(p - self.d, self.gamma_r, self.rig, iters=12)
            if pl is not None:
                left[i] = pl
            if pr is not None:
                right[i] = pr
        return left, right

    def in_frame(self, left: np.ndarray, right: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of pairs visible in both images."""
        w, h = self.rig.img_w, self.rig.img_h

        def inside(px: np.ndarray) -> np.ndarray:
            return (
                (px[:, 0] >= margin) & (px[:, 0] <= w - 1 - margin)
                & (px[:, 1] >= margin) & (px[:, 1] <= h - 1 - margin)
            )

        return inside(left) & inside(right) & ~np.isnan(left).any(1) & ~np.isnan(right).any(1)

    def make_cal_pts(
        self,
        n: int,
        z_range: tuple[float, float] = (0.8, 6.0),
        noise_px: float = 0.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Random visible scene points as ``(N, 5)`` calibration rows.

        Points are drawn uniformly over the left image and in ``1/Z`` (so near
        and far are equally represented), projected, and optionally perturbed
        with Gaussian pixel noise in both images.
        """
        rng = np.random.default_rng(seed)
        rows = []
        while len(rows) < n:
            m = max(2 * (n - len(rows)), 8)
            u = rng.uniform(0, self.rig.img_w - 1, m)
            v = rng.uniform(0, self.rig.img_h - 1, m)
            inv_z = rng.uniform(1.0 / z_range[1], 1.0 / z_range[0], m)
            z = 1.0 / inv_z
            # pinhole-ish back-projection just to seed the point cloud
            x = (u - self.rig.cx) / self.rig.cx * self.rig.g0 * z
            y = (v - self.rig.cy) / self.rig.cy * self.rig.g0 * z
            pts = np.stack([x, y, z], axis=1)
            left, right = self.project(pts)
            ok = self.in_frame(left, right, margin=2.0)
            for i in np.flatnonzero(ok):
                rows.append([left[i, 0], left[i, 1], right[i, 0], right[i, 1], z[i]])
                if len(rows) == n:
                    break
        arr = np.asarray(rows, dtype=np.float64)
        if noise_px > 0:
            arr[:, :4] += rng.normal(0.0, noise_px, size=(n, 4))
        return arr


def make_texture(h: int, w: int, seed: int = 0, scale: int = 4) -> np.ndarray:
    """Band-limited random texture (0-255, ``float32``) for matching tests.

    White noise up-sampled by ``scale`` with bilinear interpolation gives a
    texture with correlation length of a few pixels - enough structure for
    ZNCC / SSD / LK without being periodic.
    """
    rng = np.random.default_rng(seed)
    small = rng.uniform(0.0, 1.0, size=(h // scale + 2, w // scale + 2))
    ys = np.arange(h) / scale
    xs = np.arange(w) / scale
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    img = (
        (1 - tx) * (1 - ty) * small[y0][:, x0]
        + tx * (1 - ty) * small[y0][:, x0 + 1]
        + (1 - tx) * ty * small[y0 + 1][:, x0]
        + tx * ty * small[y0 + 1][:, x0 + 1]
    )
    return (255.0 * img).astype(np.float32)


def shift_image(img: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    """Translate an image by a (possibly fractional) ``(dx, dy)`` with bilinear sampling.

    ``out[y, x] = img[y - dy, x - dx]``; i.e. content moves right by ``dx``.
    Border samples clamp.
    """
    h, w = img.shape
    ys = np.arange(h, dtype=np.float64) - dy
    xs = np.arange(w, dtype=np.float64) - dx
    ys = np.clip(ys, 0, h - 1.001)
    xs = np.clip(xs, 0, w - 1.001)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    return (
        (1 - tx) * (1 - ty) * img[y0][:, x0]
        + tx * (1 - ty) * img[y0][:, x0 + 1]
        + (1 - tx) * ty * img[y0 + 1][:, x0]
        + tx * ty * img[y0 + 1][:, x0 + 1]
    ).astype(np.float32)
