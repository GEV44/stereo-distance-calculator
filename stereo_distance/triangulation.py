"""OLS ray intersection.

Given the two rays ``r_L`` and ``r_R`` of a matched pixel pair and the
baseline ``d``, the point satisfies ``Z r_L - Z_R r_R = d`` - three equations in
two unknowns.  The depth is the least-squares solution

    A = [r_L | -r_R]        x = (A^T A)^-1 A^T d        Z = x[0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RIG, RigConfig
from .lens import build_ray, build_rays_batch


@dataclass(frozen=True)
class Triangulation:
    """Result of :func:`triangulate_pair`.

    ``z`` is ``nan`` when the pair is geometrically invalid and ``error`` then
    explains why.  ``z_right`` is the depth along the right camera axis and
    ``residual`` the norm of ``A x - d`` (0 when the rays meet exactly).
    """

    z: float
    z_right: float
    residual: float
    ray_l: np.ndarray | None
    ray_r: np.ndarray | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and not math.isnan(self.z)

    @property
    def point(self) -> np.ndarray | None:
        """3-D point in the left camera frame (``Z * r_L``) or ``None``."""
        if not self.ok or self.ray_l is None:
            return None
        return self.z * self.ray_l


def triangulate_pair(
    u_l: float,
    v_l: float,
    u_r: float,
    v_r: float,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
    gd_steps: int = 50,
) -> Triangulation:
    """Depth of one matched pixel pair via the OLS normal equations.

    ``gd_steps`` iterations of batch gradient descent on ``||Ax - d||^2`` are
    run first for pedagogy (the original course requirement); the closed-form
    normal-equation solve is authoritative.  Set ``gd_steps=0`` to skip them.
    """
    nan = float("nan")
    if u_l <= u_r:
        return Triangulation(nan, nan, nan, None, None, f"bad disparity uL={u_l} <= uR={u_r}")
    ray_l, _ = build_ray(u_l, v_l, gamma_l, rig)
    ray_r, _ = build_ray(u_r, v_r, gamma_r, rig)
    denom = float(ray_l[0]) - float(ray_r[0])
    if abs(denom) < 1e-9:
        return Triangulation(nan, nan, nan, ray_l, ray_r, "zero x-disparity (rays parallel)")

    a = np.column_stack([ray_l, -ray_r])
    ata = a.T @ a
    atd = a.T @ np.asarray(d, dtype=np.float64)

    x_gd = np.zeros(2)
    for _ in range(gd_steps):
        x_gd -= 0.01 * (2.0 * ata @ x_gd - 2.0 * atd)

    try:
        x = np.linalg.solve(ata, atd)
    except np.linalg.LinAlgError:
        x = np.array([float(d[0]) / denom, float(d[0]) / denom])
    z, z_r = float(x[0]), float(x[1])
    residual = float(np.linalg.norm(a @ x - d))
    if z <= 0:
        return Triangulation(nan, z_r, residual, ray_l, ray_r, f"negative depth Z = {z:.2f} m")
    return Triangulation(z, z_r, residual, ray_l, ray_r, None)


def triangulate(
    u_l: float,
    v_l: float,
    u_r: float,
    v_r: float,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
) -> tuple[float, np.ndarray | None, np.ndarray | None, str | None]:
    """Original tuple-returning interface: ``(Z, ray_l, ray_r, error)``."""
    t = triangulate_pair(u_l, v_l, u_r, v_r, gamma_l, gamma_r, d, rig)
    return t.z, t.ray_l, t.ray_r, t.error


def triangulate_batch(
    u_l: np.ndarray,
    v_l: np.ndarray,
    u_r: np.ndarray,
    v_r: np.ndarray,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
) -> np.ndarray:
    """Vectorised OLS depth for ``N`` pairs; invalid pairs give ``nan``.

    Solves the 2x2 normal equations in closed form for every pair at once.
    """
    rays_l = build_rays_batch(u_l, v_l, gamma_l, rig)[0]
    rays_r = build_rays_batch(u_r, v_r, gamma_r, rig)[0]
    d = np.asarray(d, dtype=np.float64)
    # A = [rL | -rR];  AtA = [[rL.rL, -rL.rR], [-rL.rR, rR.rR]];  Atd = [rL.d, -rR.d]
    ll = (rays_l * rays_l).sum(1)
    rr = (rays_r * rays_r).sum(1)
    lr = (rays_l * rays_r).sum(1)
    ld = rays_l @ d
    rd = rays_r @ d
    det = ll * rr - lr * lr
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (rr * ld - lr * rd) / det
    denom = rays_l[:, 0] - rays_r[:, 0]
    bad = (np.asarray(u_l) <= np.asarray(u_r)) | (np.abs(denom) < 1e-9) | ~(z > 0)
    z = np.where(bad, np.nan, z)
    return z
