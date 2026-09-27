"""
The calibrated stereo model: parameters, triangulation, epipolar geometry,
uncertainty and (de)serialisation.

Parameters θ
    Γ_L, Γ_R  (rows × cols × 2)   per-camera γx / γy grids  (lens model)
    d = [dx, dy, 0]               baseline, right camera centre in left frame
    ω = [pitch, yaw, roll]        rotation of the right camera (radians)

Geometry
    P_L = Z · ray_L = R(ω) · P_R + d                                   (1)
    ⇒  Z · ray_L − Z' · r̂_R = d          (r̂_R = R·ray_R / (R·ray_R)_z)   (2)
    solved in the least-squares sense:  x = (AᵀA)⁻¹Aᵀd,  A = [ray_L, −r̂_R]
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_SENSOR, DX_MIN, Sensor
from .geometry import (
    bilinear_weights,
    interpolate_gamma,
    rays,
    resample_grid,
    right_rays_in_left,
    rotation,
)

FORMAT_VERSION = "stereo-gamma/2"


@dataclass
class StereoModel:
    gamma_L: np.ndarray
    gamma_R: np.ndarray
    baseline: np.ndarray  # (3,)  [dx, dy, dz=0]
    rotation: np.ndarray  # (3,)  [pitch, yaw, roll] rad
    sensor: Sensor = DEFAULT_SENSOR
    postcorr: np.ndarray | None = None  # [a, b, c] :  Z ← a + bZ + cZ²
    meta: dict = field(default_factory=dict)
    companion: np.ndarray | None = None  # Brown–Conrady parameters of the ensemble partner (or None)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def nominal(cls, sensor: Sensor = DEFAULT_SENSOR, dx: float = 0.10) -> StereoModel:
        """Ideal pinhole pair: Γ = Γ₀ everywhere, no rotation."""
        g = np.broadcast_to(sensor.gamma0, (sensor.grid_rows, sensor.grid_cols, 2)).copy()
        return cls(g, g.copy(), np.array([dx, 0.0, 0.0]), np.zeros(3), sensor)

    def copy(self) -> StereoModel:
        return StereoModel(
            self.gamma_L.copy(), self.gamma_R.copy(), self.baseline.copy(),
            self.rotation.copy(), self.sensor,
            None if self.postcorr is None else self.postcorr.copy(), dict(self.meta),
            None if self.companion is None else self.companion.copy(),
        )

    @property
    def is_trained(self) -> bool:
        return bool(self.meta.get("trained", False))

    # ── core geometry ────────────────────────────────────────────────────────
    def left_rays(self, uL, vL):
        return rays(self.sensor, self.gamma_L, uL, vL)[0]

    def right_rays(self, uR, vR):
        """Right rays in the left frame, normalised to z = 1."""
        return right_rays_in_left(self.sensor, self.gamma_R, self.rotation, uR, vR)[0]

    def triangulate(self, uL, vL, uR, vR, correct: bool = True, ensemble: bool = True):
        """
        Depth along the left optical axis (m), vectorised.  Returns ``(Z, ok)``.

        The Γ-grid depth is the OLS triangulation (normal equations,
        closed-form 2×2 inverse).  If the model carries a Brown–Conrady
        ``companion`` and ``ensemble`` is true, the result is the equal-weight
        average of the two models' depths (see README, "Model averaging").
        Invalid entries (parallel rays, point behind the rig) are NaN.
        """
        Z, ok = self.triangulate_gamma(uL, vL, uR, vR)
        if ensemble and self.companion is not None:
            Zb = self.companion_model().triangulate(uL, vL, uR, vR)
            Z = np.where(ok & np.isfinite(Zb), 0.5 * (Z + Zb), Z)
        if correct and self.postcorr is not None:
            Z = self.apply_postcorr(Z)
        return Z, ok

    def companion_model(self):
        from .baselines import BrownConradyStereo

        return BrownConradyStereo(self.sensor, np.asarray(self.companion, float))

    def triangulate_gamma(self, uL, vL, uR, vR):
        """Γ-grid OLS triangulation only.  Returns ``(Z, ok)``."""
        rL = self.left_rays(uL, vL)
        rR = self.right_rays(uR, vR)
        a1, a2 = rL, -rR
        d = self.baseline
        s11 = np.einsum("ij,ij->i", a1, a1)
        s12 = np.einsum("ij,ij->i", a1, a2)
        s22 = np.einsum("ij,ij->i", a2, a2)
        b1, b2 = a1 @ d, a2 @ d
        det = s11 * s22 - s12 * s12
        disp = rL[:, 0] - rR[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(np.abs(det) > 1e-14,
                         (s22 * b1 - s12 * b2) / det,
                         d[0] / disp)                      # near-singular 1-D fallback
        ok = np.isfinite(Z) & (Z > 0) & (disp > 1e-9)
        return np.where(ok, Z, np.nan), ok

    def depth(self, uL, vL, uR, vR) -> float:
        """Convenience scalar version of :meth:`triangulate`."""
        return float(self.triangulate([uL], [vL], [uR], [vR])[0][0])

    @property
    def lateral_scale(self) -> float:
        """
        Scale k of the lateral coordinates X, Y (depth is unaffected).  Depth
        labels determine only the product focal length × baseline, so the
        absolute focal length — and with it X, Y and lengths — is fixed by the
        assumed f₀.  k = f₀ / f_true corrects this; it is fitted from known
        lengths (:meth:`fit_lateral_scale`) and is 1 until then.
        """
        return float(self.meta.get("lateral_scale", 1.0))

    def point_3d(self, uL, vL, uR, vR) -> np.ndarray:
        """
        3-D point(s) ``(N, 3)`` = [X, Y, Z] in metres in the left-camera frame
        (x right, y down, z along the optical axis): the measured depth Z placed
        on the left ray, ``P = Z · ray_L``, lateral part scaled by ``lateral_scale``.
        """
        Z = self.triangulate(uL, vL, uR, vR)[0]
        P = Z[:, None] * self.left_rays(uL, vL)
        P[:, :2] *= self.lateral_scale
        return P

    def fit_lateral_scale(self, refs=None) -> float:
        """
        Fit k from reference lengths ``[[uL1, vL1, uR1, vR1, uL2, vL2, uR2, vR2, L], …]``
        (defaults to ``meta["length_refs"]``) by least squares on
        ``‖(k·ΔX, k·ΔY, ΔZ)‖ = L`` (Gauss–Newton, a 1-D convex-in-practice problem).
        Stores the refs and k in ``meta`` and returns k.
        """
        refs = np.asarray(self.meta.get("length_refs", []) if refs is None else refs, float).reshape(-1, 9)
        self.meta["length_refs"] = refs.tolist()
        if len(refs) == 0:
            self.meta.pop("lateral_scale", None)
            return 1.0
        self.meta["lateral_scale"] = 1.0
        P1, P2 = self.point_3d(*refs[:, 0:4].T), self.point_3d(*refs[:, 4:8].T)
        D = P1 - P2
        lat2, dz2, L = D[:, 0] ** 2 + D[:, 1] ** 2, D[:, 2] ** 2, refs[:, 8]
        use = np.isfinite(lat2) & (lat2 > 1e-6)
        if not use.any():
            raise ValueError("the reference lengths have no lateral component — measure across the image")
        k = 1.0
        for _ in range(50):
            n = np.sqrt(k * k * lat2[use] + dz2[use])
            r, J = n - L[use], k * lat2[use] / n
            step = float(J @ r / (J @ J))
            k -= step
            if abs(step) < 1e-12:
                break
        self.meta["lateral_scale"] = float(k)
        return float(k)

    def range(self, uL, vL, uR, vR) -> float:
        """
        Straight-line (Euclidean) distance ‖P‖ from the left camera's optical
        centre.  Differs from the depth Z off-axis: ‖P‖ = Z·‖ray_L‖ (≈ +21 %
        in the image corners of this rig).  Calibration labels are depths.
        """
        return float(np.linalg.norm(self.point_3d([uL], [vL], [uR], [vR])[0]))

    def triangulate_gd(self, uL, vL, uR, vR, max_iter: int = 20000, tol: float = 1e-12,
                       accelerated: bool = True):
        """
        The same least-squares problem solved by *batch gradient descent* on
        f(x) = ‖Ax − d‖²,  ∇f = 2(AᵀAx − Aᵀd),  with step η = 1/L where
        L = 2·λ_max(AᵀA) is the Lipschitz constant of ∇f.

        Plain GD contracts the error by (1 − 1/κ) per step, κ = λ_max/λ_min.
        For distant points the rays are nearly parallel and κ reaches 10³–10⁴,
        so ``accelerated=True`` adds Nesterov momentum β = (√κ−1)/(√κ+1),
        which needs only O(√κ) steps.  Either way the result converges to the
        closed-form OLS solution, which is why OLS is the authoritative path.

        It solves the Γ-grid problem only (compare with :meth:`triangulate_gamma`).
        Returns ``(Z, iterations)``; ``iterations == max_iter`` means the
        relative gradient norm did not reach ``tol``.
        """
        rL = self.left_rays(uL, vL)
        rR = self.right_rays(uR, vR)
        A = np.stack([rL, -rR], axis=2)  # (N, 3, 2)
        AtA = np.einsum("nki,nkj->nij", A, A)
        Atd = np.einsum("nki,k->ni", A, self.baseline)
        lam = np.linalg.eigvalsh(AtA)
        L = 2.0 * lam[:, -1]
        sk = np.sqrt(lam[:, -1] / np.maximum(lam[:, 0], 1e-300))
        beta = ((sk - 1) / (sk + 1))[:, None] if accelerated else np.zeros((len(L), 1))
        x = np.zeros_like(Atd)
        y = x.copy()
        scale = np.linalg.norm(Atd, axis=1) + 1e-300
        it = 0
        for it in range(1, max_iter + 1):  # noqa: B007 — iteration count is returned
            g = 2.0 * (np.einsum("nij,nj->ni", AtA, y) - Atd)
            x_new = y - g / L[:, None]
            y = x_new + beta * (x_new - x)
            x = x_new
            if float(np.max(np.linalg.norm(g, axis=1) / scale)) < tol:
                break
        return x[:, 0], it

    # ── post-correction (PDF §12) ────────────────────────────────────────────
    def apply_postcorr(self, Z):
        a, b, c = self.postcorr
        return a + b * Z + c * Z * Z

    # ── forward projection / epipolar curve ──────────────────────────────────
    def _pixel_from_ray(self, gamma, tx, ty, iters: int = 12):
        """
        Invert a Γ grid: find (u, v) with ũ·γx(u, v) = tx and ṽ·γy(u, v) = ty by
        fixed-point iteration  u ← CX + CX·tx / γx(u, v)  (converges in a few
        steps because Γ varies slowly over the image).
        """
        s = self.sensor
        g0 = s.gamma0
        u = s.cx + s.cx * tx / g0[0]
        v = s.cy + s.cy * ty / g0[1]
        for _ in range(iters):
            uc = np.clip(u, -0.25 * s.width, 1.25 * s.width)
            vc = np.clip(v, -0.25 * s.height, 1.25 * s.height)
            w, idx = bilinear_weights(s, uc, vc)
            g = np.maximum(interpolate_gamma(gamma, w, idx), 1e-6)
            u = s.cx + s.cx * tx / g[:, 0]
            v = s.cy + s.cy * ty / g[:, 1]
        return u, v

    def project_left_point_to_right(self, uL, vL, Z):
        """
        Pixel in the right image where the 3-D point at raw depth ``Z`` along
        the left ray through (uL, vL) appears.  Vectorised over ``Z``.
        Returns ``(uR, vR, valid)``.
        """
        Z = np.atleast_1d(np.asarray(Z, dtype=np.float64))
        rL = self.left_rays([uL], [vL])[0]
        PR = (Z[:, None] * rL[None, :] - self.baseline[None, :]) @ rotation(self.rotation)  # Rᵀ(P − d)
        valid = PR[:, 2] > 1e-6
        zs = np.where(valid, PR[:, 2], 1.0)
        uR, vR = self._pixel_from_ray(self.gamma_R, PR[:, 0] / zs, PR[:, 1] / zs)
        return uR, vR, valid

    def project_right_point_to_left(self, uR, vR, t):
        """
        Left-image pixel of the point at right-frame depth ``t`` along the right
        ray through (uR, vR) — used for the left-right consistency check.
        Returns ``(uL, vL, valid)``.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        r = rays(self.sensor, self.gamma_R, [uR], [vR])[0][0]
        P = (t[:, None] * r[None, :]) @ rotation(self.rotation).T + self.baseline[None, :]
        valid = P[:, 2] > 1e-6
        zs = np.where(valid, P[:, 2], 1.0)
        uL, vL = self._pixel_from_ray(self.gamma_L, P[:, 0] / zs, P[:, 1] / zs)
        return uL, vL, valid

    def raw_depth_for(self, Z_corrected):
        """Invert the post-correction polynomial (identity if none)."""
        Zc = np.atleast_1d(np.asarray(Z_corrected, dtype=np.float64))
        if self.postcorr is None:
            return Zc
        grid = np.geomspace(0.05, 500.0, 4000)
        mapped = self.apply_postcorr(grid)
        if np.all(np.diff(mapped) > 0):
            return np.interp(Zc, mapped, grid)
        return Zc

    def epipolar_curve(self, uL, vL, z_min: float = 0.3, z_max: float = 60.0, n: int = 400):
        """
        Sampled epipolar curve in the right image for depths in ``[z_min, z_max]``,
        uniformly spaced in inverse depth (≈ uniform in pixels).
        Returns ``(uR, vR, Z)`` of the in-image samples.
        """
        inv = np.linspace(1.0 / z_min, 1.0 / z_max, n)
        Z = self.raw_depth_for(1.0 / inv)
        uR, vR, ok = self.project_left_point_to_right(uL, vL, Z)
        s = self.sensor
        ok &= (uR >= 0) & (uR <= s.width - 1) & (vR >= 0) & (vR <= s.height - 1)
        Zout = self.apply_postcorr(Z) if self.postcorr is not None else Z
        return uR[ok], vR[ok], Zout[ok]

    def epipolar_curve_left(self, uR, vR, z_min: float = 0.3, z_max: float = 60.0, n: int = 400):
        """Epipolar curve in the *left* image of a right-image pixel.  Returns ``(uL, vL)``."""
        t = 1.0 / np.linspace(1.0 / z_min, 1.0 / z_max, n)
        uL, vL, ok = self.project_right_point_to_left(uR, vR, t)
        s = self.sensor
        ok &= (uL >= 0) & (uL <= s.width - 1) & (vL >= 0) & (vL <= s.height - 1)
        return uL[ok], vL[ok]

    # ── uncertainty ──────────────────────────────────────────────────────────
    def depth_uncertainty(self, uL, vL, uR, vR, sigma_px: float | None = None):
        """
        First-order error propagation of click noise through the full model,

            σ_Z,click² = Σ_p (∂Z/∂p)² σ_px² ,   p ∈ {uL, vL, uR, vR}

        combined with the calibration's relative model error ε (estimated by
        leave-one-out cross-validation at training time):

            σ_Z = sqrt(σ_Z,click² + (ε·Z)²)

        Returns ``(Z, sigma_total, sigma_click)``.
        """
        if sigma_px is None:
            sigma_px = float(self.meta.get("sigma_px", 1.0))
        rel = float(self.meta.get("rel_model_error", 0.0))
        p = np.array([uL, vL, uR, vR], dtype=np.float64)
        Z = self.depth(*p)
        if not math.isfinite(Z):
            return Z, float("nan"), float("nan")
        h = 0.5
        var = 0.0
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = h
            zp, zm = self.depth(*(p + dp)), self.depth(*(p - dp))
            if math.isfinite(zp) and math.isfinite(zm):
                var += ((zp - zm) / (2 * h) * sigma_px) ** 2
        s_click = math.sqrt(var)
        return Z, math.sqrt(var + (rel * Z) ** 2), s_click

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "format": FORMAT_VERSION,
            "model": f"gamma_grid_{self.sensor.grid_rows}x{self.sensor.grid_cols}x2+rotation",
            "sensor": self.sensor.to_dict(),
            "gamma_L": self.gamma_L.tolist(),
            "gamma_R": self.gamma_R.tolist(),
            "baseline": {"dx": float(self.baseline[0]), "dy": float(self.baseline[1]), "dz": 0.0},
            "rotation_rad": {"pitch": float(self.rotation[0]), "yaw": float(self.rotation[1]),
                             "roll": float(self.rotation[2])},
            "postcorrection": None if self.postcorr is None else [float(x) for x in self.postcorr],
            "companion": None if self.companion is None else {
                "type": "brown_conrady",
                "layout": "cxL cyL k1L k2L p1L p2L fR cxR cyR k1R k2R p1R p2R pitch yaw roll dx dy",
                "params": [float(x) for x in self.companion]},
            "meta": self.meta,
        }

    def save(self, path: str | os.PathLike) -> None:
        self.meta.setdefault("saved", _dt.datetime.now().isoformat(timespec="seconds"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=float)

    @classmethod
    def from_dict(cls, p: dict) -> StereoModel:
        if "gamma_L" in p:  # v2
            sensor = Sensor.from_dict(p.get("sensor", {}))
            gL, gR = np.asarray(p["gamma_L"], float), np.asarray(p["gamma_R"], float)
            rot = p.get("rotation_rad", {})
            omega = np.array([rot.get("pitch", 0.0), rot.get("yaw", 0.0), rot.get("roll", 0.0)])
            pc = p.get("postcorrection")
            comp = p.get("companion")
            model = cls(gL, gR, _baseline(p), omega, sensor,
                        None if pc is None else np.asarray(pc, float), dict(p.get("meta", {})),
                        None if comp is None else np.asarray(comp["params"], float))
        elif "gamma_L_4x4" in p:  # v1 — single γ per node, no rotation
            sensor = DEFAULT_SENSOR
            grids = []
            for key in ("gamma_L_4x4", "gamma_R_4x4"):
                g = np.asarray(p[key], float)
                if g.shape != (sensor.grid_rows, sensor.grid_cols):
                    g = resample_grid(g, sensor.grid_rows, sensor.grid_cols)
                # v1 used ray = [ũ·g, ṽ·g, 1]  ⇒  γx = γy = g reproduces it exactly
                grids.append(np.stack([g, g], axis=-1))
            model = cls(grids[0], grids[1], _baseline(p), np.zeros(3), sensor,
                        meta={"trained": True, "converted_from": p.get("model", "v1")})
        else:
            raise ValueError("unrecognised calibration file (no gamma grids found)")
        model.meta.setdefault("trained", True)
        return model

    @classmethod
    def load(cls, path: str | os.PathLike) -> StereoModel:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"{path} is empty")
        return cls.from_dict(json.loads(content))


def _baseline(p: dict) -> np.ndarray:
    b = p["baseline"]
    return np.array([max(float(b["dx"]), DX_MIN), float(b.get("dy", 0.0)), 0.0])
