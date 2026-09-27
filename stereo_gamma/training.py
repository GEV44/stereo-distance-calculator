"""
Calibration: analytic gradients, Levenberg–Marquardt or Adam (no autodiff, no SciPy).

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
from dataclasses import replace

import numpy as np

from .config import DEFAULT_SENSOR, DX_MIN, G_MAX, G_MIN, Sensor, TrainConfig
from .geometry import rays, rotation_with_jacobians
from .model import StereoModel
from .solver import (
    N_RADIAL,
    diagnostics,
    levenberg_marquardt,
    pack,
    prior_mean,
    regulariser_hessian,
    unpack,
)

_H_CACHE: dict = {}


def _reg_hessian(sensor: Sensor, cfg: TrainConfig) -> np.ndarray:
    key = (sensor, cfg.anchor, cfg.smooth, cfg.radial_prior)
    if key not in _H_CACHE:
        if len(_H_CACHE) > 64:
            _H_CACHE.clear()
        _H_CACHE[key] = regulariser_hessian(sensor, cfg)
    return _H_CACHE[key]


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


def residual_scales(pts: np.ndarray):
    """
    Per-point residual scales ``(c_x, c_y)``: columns 5–6 of ``pts`` when
    present (noise-aware weighting, see :func:`noise_scales`), else ones.
    The robust loss sees ``c·e``; all derivative formulas stay unchanged.
    """
    if pts.shape[1] >= 7:
        return pts[:, 5], pts[:, 6]
    one = np.ones(len(pts))
    return one, one


def _forward(theta, pts, sensor):
    uL, vL, uR, vR, Z = pts[:, :5].T
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
    Total loss and its exact gradient w.r.t. ``theta = {gL, gR, d=[dx,dy], w=ω, a}``
    (``a`` = radial-prior coefficients, which enter only the regulariser).
    Every derivative below is the hand-derived chain rule documented in
    ``docs/MATHEMATICAL_FOUNDATION.pdf`` §7.
    """
    ex, ey, (rL, cL, rR, cR, R, dR, W, s, dx) = _forward(theta, pts, sensor)
    N = len(pts)
    cx, cy = residual_scales(pts)
    rho_x, psi_x = huber(cx * ex, cfg.huber_delta)
    rho_y, psi_y = huber(cy * ey, cfg.huber_delta)
    data = float(np.mean(rho_x + cfg.weight_y * rho_y))

    # ── regulariser: exact quadratic form ½(θ−θ₀)ᵀH(θ−θ₀) (anchor + membrane, radial prior) ──
    x = pack(theta)
    H = _reg_hessian(sensor, cfg)
    dv = x - prior_mean(sensor, x)
    Hdv = H @ dv
    loss = data + 0.5 * float(dv @ Hdv)
    if not need_grad:
        return loss, None

    gx = cx * psi_x / N                      # ∂L/∂e_x
    gy = cfg.weight_y * cy * psi_y / N       # ∂L/∂e_y

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

    grads = {"gL": gL.reshape(rows, cols, 2), "gR": gR.reshape(rows, cols, 2), "d": g_d, "w": g_w,
             "a": np.zeros(N_RADIAL)}
    reg = unpack(Hdv, sensor)
    for key in ("gL", "gR", "d", "w", "a"):
        grads[key] = grads[key] + reg[key]
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
    uL, vL, uR, vR, Z = pts[:, :5].T
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


def _mape(theta, pts, sensor):
    Z = _to_model(theta, sensor).triangulate(*pts[:, :4].T)[0]
    return float(np.nanmean(np.abs(Z / pts[:, 4] - 1)))


def _fit_lm(theta, pts, cfg, sensor, log):
    """Levenberg–Marquardt (see :mod:`stereo_gamma.solver`); history per iteration."""
    theta, info = levenberg_marquardt(theta, pts, cfg, sensor, max_iter=cfg.lm_max_iter)
    N = len(pts)
    history = [(k, F / N, float("nan")) for k, F in enumerate(info["history"])]
    if history:
        history[-1] = (history[-1][0], history[-1][1], _mape(theta, pts, sensor))
    if cfg.log_every and log:
        log(f"  LM: {info['iterations']} iterations  L={info['F'] / N:.6f}  "
            f"MAPE={100 * history[-1][2]:5.2f}%  dx={theta['d'][0]:.4f}  "
            f"ω=[{', '.join(f'{math.degrees(a):+.3f}°' for a in theta['w'])}]")
    return (info["F"] / N, theta), history


def _fit_adam(theta, pts, cfg, sensor, log):
    """Adam with cosine decay; keeps the best iterate.  History every ``log_every`` epochs."""
    base = StereoModel.nominal(sensor)
    frozen = np.zeros((sensor.grid_rows, sensor.grid_cols, 2), dtype=bool)
    frozen[sensor.center_node] = True  # gauge fix (see module docstring)
    opt = Adam({"gL": cfg.lr_gamma, "gR": cfg.lr_gamma, "d": cfg.lr_baseline, "w": cfg.lr_rotation,
                "a": cfg.lr_gamma * 10},
               cfg.adam_betas, cfg.adam_eps)
    history, best = [], (math.inf, None)
    for ep in range(cfg.epochs + 1):
        loss, grads = objective(theta, pts, cfg, sensor)
        if loss < best[0]:
            best = (loss, {k: v.copy() for k, v in theta.items()})
        if cfg.log_every and (ep % cfg.log_every == 0 or ep == cfg.epochs):
            mare = _mape(theta, pts, sensor)
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
        if not (cfg.radial_prior and cfg.learn_gamma):
            grads["a"][:] = 0.0
        grads["gL"][frozen] = 0.0
        opt.step(theta, grads, _cosine(ep, cfg.epochs, cfg.lr_final_frac))
        np.clip(theta["gL"], G_MIN, G_MAX, out=theta["gL"])
        np.clip(theta["gR"], G_MIN, G_MAX, out=theta["gR"])
        theta["gL"][frozen] = base.gamma_L[frozen]
        theta["d"][0] = max(theta["d"][0], DX_MIN)
    return best, history


def _initial_theta(pts, cfg, sensor):
    base = StereoModel.nominal(sensor)
    theta = {"gL": base.gamma_L, "gR": base.gamma_R, "d": np.array([0.10, 0.0]), "w": np.zeros(3),
             "a": np.zeros(N_RADIAL)}
    if cfg.warm_start:
        theta["d"], w0 = linear_init(pts, sensor, cfg.learn_rotation)
        if cfg.learn_rotation:
            theta["w"] = w0
    return theta


def _fit(pts, cfg, sensor, log=None):
    """One optimisation run (warm start → LM or Adam).  Returns ``((loss, theta), history)``."""
    theta = _initial_theta(pts, cfg, sensor)
    if cfg.solver == "lm":
        return _fit_lm(theta, pts, cfg, sensor, log)
    if cfg.solver == "adam":
        return _fit_adam(theta, pts, cfg, sensor, log)
    raise ValueError(f"unknown solver {cfg.solver!r}")


# ─── NOISE MODEL (feasible generalised least squares) ─────────────────────────
def estimate_noise(theta: dict, pts: np.ndarray, sensor: Sensor) -> dict:
    """
    Two noise sources act on a calibration point:

    * click noise (σ_px per coordinate) perturbs the disparity by √2·σ_px/f,
      i.e. the *relative* residuals by  c·Z  with  c = √2·σ_px / (f·d_x);
    * label noise (tape measurement) perturbs e_x by a constant relative σ_ℓ.

    Hence  Var(e_x) = σ_ℓ² + c²Z²  and  Var(e_y) = c²Z²  (labels barely enter e_y).
    c is estimated robustly from e_y, σ_ℓ from the remainder of e_x
    (median-absolute-deviation estimators, robust to mis-clicks).  Returns
    ``{c, sigma_label, kappa = σ_ℓ/c (m), sigma_px}``.
    """
    ex, ey = residuals(theta, pts, sensor)
    Z = pts[:, 4]
    c = max(1.4826 * float(np.median(np.abs(ey / Z))), 1e-6)
    sl2 = max((1.4826 * float(np.median(np.abs(ex)))) ** 2 - c * c * float(np.median(Z * Z)), 0.002 ** 2)
    return {"c": c, "sigma_label": math.sqrt(sl2), "kappa": math.sqrt(sl2) / c,
            "sigma_px": c * sensor.f_init * float(theta["d"][0]) / math.sqrt(2)}


def noise_scales(Z: np.ndarray, kappa: float, Z_ref: float) -> np.ndarray:
    """
    Residual scales ∝ 1/σ_i (maximum-likelihood weighting), normalised to 1 at
    Z_ref so the Huber threshold keeps its meaning.  Returns ``(N, 2)``.
    """
    cx = np.sqrt((kappa ** 2 + Z_ref ** 2) / (kappa ** 2 + Z ** 2))
    cy = Z_ref / Z
    return np.stack([cx, cy], axis=1)


# ─── REGULARISATION STRENGTH BY INNER CROSS-VALIDATION ────────────────────────
def select_smoothing(pts: np.ndarray, cfg: TrainConfig, sensor: Sensor):
    """
    Choose λ_s from ``cfg.smooth_grid`` by *repeated* k-fold cross-validation
    on the training points only (``cfg.cv_repeats`` shuffles, averaged, to
    reduce the variance of the selection on small data).  Criterion: held-out
    depth MAPE; ties → the stronger, simpler model.

    Returns ``(λ_s, {λ: cv_mape}, cv_rel_errors_of_chosen)``; the relative
    errors are averaged over repeats per point.
    """
    N = len(pts)
    k = max(2, min(cfg.cv_folds, N))
    splits = []
    for rep in range(max(1, cfg.cv_repeats)):
        order = np.random.default_rng(rep).permutation(N)
        splits.append(np.array_split(order, k))
    scores, errs = {}, {}
    for lam in cfg.smooth_grid:
        c = replace(cfg, smooth=lam)
        rel = np.zeros((len(splits), N))
        for r, folds in enumerate(splits):
            for f in folds:
                tr = np.setdiff1d(np.arange(N), f)
                (_, th), _ = _fit(pts[tr], c, sensor)
                Z = _to_model(th, sensor).triangulate(*pts[f, :4].T)[0]
                rel[r, f] = Z / pts[f, 4] - 1
        scores[lam], errs[lam] = float(np.nanmean(np.abs(rel))), np.nanmean(rel, axis=0)
    best = min(scores.values())
    lam = max(l for l, v in scores.items() if v <= best * (1 + 1e-3))
    return lam, scores, errs[lam]


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────
def train(points, cfg: TrainConfig | None = None, sensor: Sensor = DEFAULT_SENSOR,
          log=print) -> tuple[StereoModel, list]:
    """
    Fit the stereo model to calibration points ``[uL, vL, uR, vR, Z]``.

    Pipeline (each step uses the training points only):

    1. closed-form warm start → robust fit with equal relative weights;
    2. ``noise_model="fgls"``: estimate click / label noise from the residuals
       and switch to maximum-likelihood weights;
    3. ``auto_smooth``: pick λ_s by k-fold cross-validation;
    4. final fit (the noise model is re-estimated once more).

    Returns ``(model, history)``; ``history`` rows are
    ``(iteration, loss, mean_abs_rel_error)`` (NaN where not evaluated).
    """
    cfg = cfg or TrainConfig()
    pts = np.asarray(points, dtype=np.float64)
    pts = pts.reshape(-1, pts.shape[-1] if pts.ndim == 2 and pts.shape[-1] in (5, 7) else 5)
    if len(pts) < 3:
        raise ValueError("need at least 3 calibration points")
    pts5 = pts[:, :5]
    Z_ref = float(np.sqrt(np.mean(pts5[:, 4] ** 2)))

    noise, work = None, pts
    if cfg.noise_model == "fgls":
        (_, th), _ = _fit(pts5, cfg, sensor)
        noise = estimate_noise(th, pts5, sensor)
        work = np.c_[pts5, noise_scales(pts5[:, 4], noise["kappa"], Z_ref)]
    elif cfg.noise_model != "relative":
        raise ValueError(f"unknown noise_model {cfg.noise_model!r}")

    cv_scores, cv_rel = None, None
    if cfg.auto_smooth and len(pts) >= 10:
        lam, cv_scores, cv_rel = select_smoothing(work, cfg, sensor)
        cfg = replace(cfg, smooth=lam)

    best, history = _fit(work, cfg, sensor, log)
    if cfg.noise_model == "fgls":  # one more FGLS iteration at the final λ
        noise = estimate_noise(best[1], pts5, sensor)
        work = np.c_[pts5, noise_scales(pts5[:, 4], noise["kappa"], Z_ref)]
        best, history = _fit(work, cfg, sensor, log)

    model = _to_model(best[1], sensor)
    if cfg.ensemble:
        from .baselines import BrownConradyStereo

        bc = BrownConradyStereo(sensor, huber_delta=cfg.huber_delta, weight_y=cfg.weight_y)
        model.companion = bc.fit(pts5, cfg.noise_model).p
    diag = diagnostics(best[1], work, cfg, sensor)
    Z_pred = model.triangulate(*pts5[:, :4].T)[0]
    if cfg.postcorrection and len(pts) >= 6:
        model.postcorr = fit_postcorrection(Z_pred, pts5[:, 4])
        Z_pred = model.apply_postcorr(Z_pred)

    rel = Z_pred / pts5[:, 4] - 1
    ey_px = residuals(best[1], pts5, sensor)[1] * (best[1]["d"][0] / pts5[:, 4]) * sensor.f_init
    mad = float(np.median(np.abs(ey_px - np.median(ey_px))))
    sigma_px = noise["sigma_px"] if noise else 1.4826 * mad / math.sqrt(2)
    if cv_rel is not None:
        rel_err, src = float(np.sqrt(np.nanmean(cv_rel ** 2))), "inner-cv"
    else:
        rel_err, src = float(np.sqrt(np.nanmean(rel ** 2))), "in-sample"
    model.meta.update({
        "trained": True,
        "n_points": int(len(pts)),
        "final_loss": float(best[0]),
        "in_sample_mape": float(np.nanmean(np.abs(rel))),
        "in_sample_rmse_m": float(np.sqrt(np.nanmean((Z_pred - pts5[:, 4]) ** 2))),
        # RMS relative prediction error, used for the ± shown by the app
        "rel_model_error": rel_err,
        "rel_model_error_source": src,
        # per-click noise σ (px): from the noise model, else from the epipolar residual
        "sigma_px": float(max(0.5, sigma_px)),
        "noise_model": noise,
        "smooth_selected": cfg.smooth,
        "smooth_cv_mape": None if cv_scores is None else {str(k): v for k, v in cv_scores.items()},
        # effective degrees of freedom of the penalised fit and rig-parameter standard errors
        "df_eff": diag.get("df_eff"),
        "radial_prior": {"left": [float(v) for v in best[1]["a"][:2]],
                         "right": [float(v) for v in best[1]["a"][2:]]} if cfg.radial_prior else None,
        "stderr": diag.get("stderr"),
        "train_config": cfg.to_dict(),
    })
    return model, history
