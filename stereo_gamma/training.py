"""
Calibration: analytic gradients + Adam (no autodiff, no SciPy).

Per calibration point i with known depth Z_i:

    D_x = ray_L,x − r̂_R,x        D_y = ray_L,y − r̂_R,y
    e_x = D_x · Z/dx − 1           relative disparity error  (≈ relative depth error)
    e_y = D_y · Z/dx − dy/dx       relative epipolar error   (same units as e_x)

    L = 1/N Σ [ ρδ(e_x) + λ_y ρδ(e_y) ]  +  λ_a ½‖Γ − Γ₀‖²  +  λ_s E_membrane(Γ)

ρδ is the Huber function.  Working in *relative disparity* rather than Z
avoids the Z² gradient blow-up of a naive (Z_pred − Z)² loss: a 5 % error at
1 m and at 5 m cost the same.

Gauge: the problem is (almost) invariant to scaling every γ and dx by the
same factor, so the centre node of Γ_L is frozen at the nominal pinhole value
Γ₀ = [CX/f, CY/f].  Γ_R's centre stays free so the two cameras may differ.
"""

from __future__ import annotations

import math

import numpy as np

from .config import DEFAULT_SENSOR, DX_MIN, G_MAX, G_MIN, Sensor, TrainConfig
from .geometry import membrane_energy_grad, rays, rotation_with_jacobians
from .model import StereoModel


# ─── ROBUST LOSS ──────────────────────────────────────────────────────────────
def huber(e: np.ndarray, delta: float):
    """Huber ρδ(e) and its derivative ψ(e) = clip(e, −δ, δ)."""
    a = np.abs(e)
    rho = np.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    return rho, np.clip(e, -delta, delta)


# ─── OBJECTIVE + ANALYTIC GRADIENT ────────────────────────────────────────────
def _scatter(grad_flat: np.ndarray, idx: np.ndarray, w: np.ndarray, coef: np.ndarray, ch: int):
    """grad[node, ch] += Σ_i coef_i · w_ik   for the 4 nodes k of every point i."""
    np.add.at(grad_flat[:, ch], idx.ravel(), (coef[:, None] * w).ravel())


def residuals(theta: dict, pts: np.ndarray, sensor: Sensor):
    """Relative disparity / epipolar residuals ``(e_x, e_y)`` for every point."""
    return _forward(theta, pts, sensor)[0:2]


def _forward(theta, pts, sensor):
    uL, vL, uR, vR, Z = pts.T
    rL, cL = rays(sensor, theta["gL"], uL, vL)
    rR, cR = rays(sensor, theta["gR"], uR, vR)
    R, dR = rotation_with_jacobians(theta["w"])
    W = rR @ R.T
    wz = W[:, 2]
    hx, hy = W[:, 0] / wz, W[:, 1] / wz
    dx, dy = float(theta["d"][0]), float(theta["d"][1])
    s = Z / dx
    ex = (rL[:, 0] - hx) * s - 1.0
    ey = (rL[:, 1] - hy) * s - dy / dx
    return ex, ey, (rL, cL, rR, cR, R, dR, W, s, dx)


def objective(theta: dict, pts: np.ndarray, cfg: TrainConfig, sensor: Sensor = DEFAULT_SENSOR,
              need_grad: bool = True):
    """
    Total loss and its exact gradient w.r.t. ``theta = {gL, gR, d=[dx,dy], w=ω}``.
    Every derivative below is the hand-derived chain rule documented in
    ``docs/MATHEMATICAL_FOUNDATION.pdf`` §7.
    """
    ex, ey, (rL, cL, rR, cR, R, dR, W, s, dx) = _forward(theta, pts, sensor)
    N = len(pts)
    rho_x, psi_x = huber(ex, cfg.huber_delta)
    rho_y, psi_y = huber(ey, cfg.huber_delta)
    data = float(np.mean(rho_x + cfg.weight_y * rho_y))

    g0 = sensor.gamma0
    loss = data
    reg = {}
    for key in ("gL", "gR"):
        dev = theta[key] - g0
        E, _ = membrane_energy_grad(theta[key])
        reg[key] = (dev, E)
        loss += cfg.anchor * 0.5 * float((dev * dev).sum()) + cfg.smooth * E
    if not need_grad:
        return loss, None

    gx = psi_x / N                      # ∂L/∂e_x
    gy = cfg.weight_y * psi_y / N       # ∂L/∂e_y

    # ── baseline ──  ∂e_x/∂dx = −(e_x+1)/dx,  ∂e_y/∂dx = −e_y/dx,  ∂e_y/∂dy = −1/dx
    g_d = np.array([float(np.sum(-gx * (ex + 1.0) / dx - gy * ey / dx)),
                    float(np.sum(-gy / dx))])

    # ── left grid ──  ray_L = [ũ γx, ṽ γy, 1],  γ = Σ_k w_k Γ_k
    rows, cols = sensor.grid_rows, sensor.grid_cols
    gL = np.zeros((rows * cols, 2))
    _scatter(gL, cL["idx"], cL["w"], gx * s * cL["un"], 0)
    _scatter(gL, cL["idx"], cL["w"], gy * s * cL["vn"], 1)

    # ── right ray through rotation + perspective division ──
    #    r̂ = W_xy / W_z,  W = R·ray_R
    wz = W[:, 2]
    a0, a1 = -gx * s, -gy * s           # ∂L/∂r̂_x, ∂L/∂r̂_y
    gW = np.stack([a0 / wz, a1 / wz, -(a0 * W[:, 0] + a1 * W[:, 1]) / (wz * wz)], axis=1)
    g_rR = gW @ R                       # ∂L/∂ray_R = Rᵀ·∂L/∂W
    gR = np.zeros((rows * cols, 2))
    _scatter(gR, cR["idx"], cR["w"], g_rR[:, 0] * cR["un"], 0)
    _scatter(gR, cR["idx"], cR["w"], g_rR[:, 1] * cR["vn"], 1)
    g_w = np.array([float(np.sum(gW * (rR @ dRj.T))) for dRj in dR])

    grads = {"gL": gL.reshape(rows, cols, 2), "gR": gR.reshape(rows, cols, 2), "d": g_d, "w": g_w}
    for key in ("gL", "gR"):
        dev, _ = reg[key]
        _, gs = membrane_energy_grad(theta[key])
        grads[key] = grads[key] + cfg.anchor * dev + cfg.smooth * gs
    return loss, grads


# ─── ADAM ─────────────────────────────────────────────────────────────────────
class Adam:
    """Adam (Kingma & Ba, 2015) with a separate learning rate per parameter block."""

    def __init__(self, lrs: dict, betas=(0.9, 0.999), eps=1e-8):
        self.lrs, (self.b1, self.b2), self.eps = lrs, betas, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, theta: dict, grads: dict, lr_scale: float = 1.0) -> None:
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k], self.v[k] = np.zeros_like(g), np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            theta[k] = theta[k] - lr_scale * self.lrs[k] * m_hat / (np.sqrt(v_hat) + self.eps)


def _cosine(ep: int, total: int, final_frac: float) -> float:
    """Cosine learning-rate schedule from 1 down to ``final_frac``."""
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * ep / max(total, 1)))


# ─── LINEAR WARM START ────────────────────────────────────────────────────────
def linear_init(pts: np.ndarray, sensor: Sensor, learn_rotation: bool = True):
    """
    Closed-form initialisation of (dx, dy, ω) with nominal Γ.  Linearising
    R ≈ I + [ω]× gives two equations per point that are *linear* in the
    unknowns (DLT-style, cf. Hartley & Zisserman §4.1):

        ray_L,x − ray_R,x = dx/Z + ω_yaw   − ω_roll · ray_R,y
        ray_L,y − ray_R,y = dy/Z − ω_pitch + ω_roll · ray_R,x

    Rows are weighted by Z so residuals are relative, like the loss.
    """
    g = np.broadcast_to(sensor.gamma0, (sensor.grid_rows, sensor.grid_cols, 2))
    uL, vL, uR, vR, Z = pts.T
    rL = rays(sensor, g, uL, vL)[0]
    rR = rays(sensor, g, uR, vR)[0]
    one, zero, iz = np.ones_like(Z), np.zeros_like(Z), 1.0 / Z
    #           dx    dy    pitch  yaw   roll
    Ax = np.stack([iz, zero, zero, one, -rR[:, 1]], 1)
    Ay = np.stack([zero, iz, -one, zero, rR[:, 0]], 1)
    A = np.vstack([Ax, Ay]) * np.concatenate([Z, Z])[:, None]
    b = np.concatenate([rL[:, 0] - rR[:, 0], rL[:, 1] - rR[:, 1]]) * np.concatenate([Z, Z])
    cols = [0, 1, 2, 3, 4] if learn_rotation else [0, 1]
    x = np.zeros(5)
    x[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
    return np.array([max(x[0], DX_MIN), x[1]]), x[2:5]


# ─── POST-CORRECTION (PDF §12) ────────────────────────────────────────────────
def fit_postcorrection(Z_pred: np.ndarray, Z_true: np.ndarray):
    """Least-squares fit of Z_true ≈ a + b·Z_pred + c·Z_pred² (Vandermonde system)."""
    ok = np.isfinite(Z_pred)
    V = np.stack([np.ones(ok.sum()), Z_pred[ok], Z_pred[ok] ** 2], axis=1)
    return np.linalg.lstsq(V, Z_true[ok], rcond=None)[0]


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def _to_model(theta, sensor) -> StereoModel:
    return StereoModel(theta["gL"].copy(), theta["gR"].copy(),
                       np.array([theta["d"][0], theta["d"][1], 0.0]), theta["w"].copy(), sensor)


def train(points, cfg: TrainConfig | None = None, sensor: Sensor = DEFAULT_SENSOR,
          log=print) -> tuple[StereoModel, list]:
    """
    Fit the stereo model to calibration points ``[uL, vL, uR, vR, Z]``.

    Returns ``(model, history)``; ``history`` rows are
    ``(epoch, loss, mean_abs_rel_error)``.
    """
    cfg = cfg or TrainConfig()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 5)
    if len(pts) < 3:
        raise ValueError("need at least 3 calibration points")

    base = StereoModel.nominal(sensor)
    theta = {"gL": base.gamma_L, "gR": base.gamma_R, "d": np.array([0.10, 0.0]), "w": np.zeros(3)}
    if cfg.warm_start:
        theta["d"], w0 = linear_init(pts, sensor, cfg.learn_rotation)
        if cfg.learn_rotation:
            theta["w"] = w0

    frozen = np.zeros((sensor.grid_rows, sensor.grid_cols, 2), dtype=bool)
    frozen[sensor.center_node] = True  # gauge fix (see module docstring)
    opt = Adam({"gL": cfg.lr_gamma, "gR": cfg.lr_gamma, "d": cfg.lr_baseline, "w": cfg.lr_rotation},
               cfg.adam_betas, cfg.adam_eps)

    history, best = [], (math.inf, None)
    for ep in range(cfg.epochs + 1):
        loss, grads = objective(theta, pts, cfg, sensor)
        if loss < best[0]:
            best = (loss, {k: v.copy() for k, v in theta.items()})
        if cfg.log_every and (ep % cfg.log_every == 0 or ep == cfg.epochs):
            mare = float(np.nanmean(np.abs(_to_model(theta, sensor).triangulate(*pts[:, :4].T)[0]
                                           / pts[:, 4] - 1)))
            history.append((ep, loss, mare))
            if log:
                log(f"  ep {ep:5d}  L={loss:.6f}  MAPE={100 * mare:5.2f}%  dx={theta['d'][0]:.4f}  "
                    f"ω=[{', '.join(f'{math.degrees(a):+.3f}°' for a in theta['w'])}]")
        if ep == cfg.epochs:
            break
        if not cfg.learn_gamma:
            grads["gL"][:] = 0.0
            grads["gR"][:] = 0.0
        if not cfg.learn_rotation:
            grads["w"][:] = 0.0
        grads["gL"][frozen] = 0.0
        opt.step(theta, grads, _cosine(ep, cfg.epochs, cfg.lr_final_frac))
        np.clip(theta["gL"], G_MIN, G_MAX, out=theta["gL"])
        np.clip(theta["gR"], G_MIN, G_MAX, out=theta["gR"])
        theta["gL"][frozen] = base.gamma_L[frozen]
        theta["d"][0] = max(theta["d"][0], DX_MIN)

    model = _to_model(best[1], sensor)
    Z_pred = model.triangulate(*pts[:, :4].T)[0]
    if cfg.postcorrection and len(pts) >= 6:
        model.postcorr = fit_postcorrection(Z_pred, pts[:, 4])
        Z_pred = model.apply_postcorr(Z_pred)

    rel = Z_pred / pts[:, 4] - 1
    ey_px = residuals(best[1], pts, sensor)[1] * (best[1]["d"][0] / pts[:, 4]) * sensor.f_init
    mad = float(np.median(np.abs(ey_px - np.median(ey_px))))
    model.meta.update({
        "trained": True,
        "n_points": int(len(pts)),
        "final_loss": float(best[0]),
        "in_sample_mape": float(np.nanmean(np.abs(rel))),
        "in_sample_rmse_m": float(np.sqrt(np.nanmean((Z_pred - pts[:, 4]) ** 2))),
        # relative model error; replaced by the LOOCV estimate when available
        "rel_model_error": float(np.sqrt(np.nanmean(rel ** 2))),
        "rel_model_error_source": "in-sample",
        # per-click noise from the epipolar residual (robust σ, split over 2 clicks)
        "sigma_px": float(max(0.5, 1.4826 * mad / math.sqrt(2))),
        "train_config": cfg.to_dict(),
    })
    return model, history
