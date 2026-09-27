"""
Second-order solver and post-fit diagnostics.

The calibration objective of :mod:`stereo_gamma.training` is a robust,
Tikhonov-regularised non-linear least-squares problem.  Multiplying it by N:

    F(θ) = Σ_i [ ρδ(e_x,i) + λ_y ρδ(e_y,i) ]  +  N · ½ (θ − θ₀)ᵀ H_reg (θ − θ₀)

where H_reg = blockdiag(λ_a I + λ_s L_graph) on the Γ entries (L_graph is the
4-neighbour graph Laplacian — anchor prior and membrane energy are both
quadratic) and 0 on the rig parameters.

This module provides

* the analytic residual Jacobian ∂(e_x, e_y)/∂θ (finite-difference verified),
* Levenberg–Marquardt with iteratively re-weighted least squares for the
  Huber loss (Holland & Welsch 1977; Moré 1978) — the standard solver for
  calibration / bundle adjustment,
* diagnostics at the optimum: effective degrees of freedom of the penalised
  fit, and the sandwich covariance of the parameters.

Parameter vector θ = [Γ_L (R·C·2), Γ_R (R·C·2), dx, dy, ω_pitch, ω_yaw, ω_roll,
a_L1, a_L2, a_R1, a_R2]; the last four are the radial-prior coefficients
(they enter only the regulariser, never the rays).
"""

from __future__ import annotations

import numpy as np

from .config import DX_MIN, G_MAX, G_MIN, Sensor, TrainConfig


# ─── PACKING ──────────────────────────────────────────────────────────────────
def n_grid(sensor: Sensor) -> int:
    return sensor.grid_rows * sensor.grid_cols * 2


N_RIG, N_RADIAL = 5, 4


def n_params(sensor: Sensor) -> int:
    return 2 * n_grid(sensor) + N_RIG + N_RADIAL


def pack(theta: dict) -> np.ndarray:
    a = theta.get("a", np.zeros(N_RADIAL))
    return np.concatenate([theta["gL"].ravel(), theta["gR"].ravel(), theta["d"], theta["w"], a])


def unpack(x: np.ndarray, sensor: Sensor) -> dict:
    G, shape = n_grid(sensor), (sensor.grid_rows, sensor.grid_cols, 2)
    return {"gL": x[:G].reshape(shape).copy(), "gR": x[G:2 * G].reshape(shape).copy(),
            "d": x[2 * G:2 * G + 2].copy(), "w": x[2 * G + 2:2 * G + 5].copy(),
            "a": x[2 * G + 5:2 * G + 9].copy()}


def free_mask(sensor: Sensor, cfg: TrainConfig) -> np.ndarray:
    """Which entries of θ are optimised (gauge node and ablated blocks are fixed)."""
    G = n_grid(sensor)
    m = np.ones(n_params(sensor), dtype=bool)
    ci, cj = sensor.center_node
    node = ci * sensor.grid_cols + cj
    m[[2 * node, 2 * node + 1]] = False  # Γ_L centre node = scale gauge
    if not cfg.learn_gamma:
        m[:2 * G] = False
    if not cfg.learn_rotation:
        m[2 * G + 2:2 * G + 5] = False
    if not (cfg.radial_prior and cfg.learn_gamma):
        m[2 * G + 5:] = False
    return m


# ─── REGULARISER ──────────────────────────────────────────────────────────────
def graph_laplacian(rows: int, cols: int) -> np.ndarray:
    """4-neighbour graph Laplacian L so that ½ Σ_edges (g_i − g_j)² = ½ gᵀ L g."""
    n = rows * cols
    L = np.zeros((n, n))
    for i in range(rows):
        for j in range(cols):
            a = i * cols + j
            for di, dj in ((0, 1), (1, 0)):
                if i + di < rows and j + dj < cols:
                    b = (i + di) * cols + j + dj
                    L[a, a] += 1
                    L[b, b] += 1
                    L[a, b] -= 1
                    L[b, a] -= 1
    return L


def radial_basis(sensor: Sensor) -> np.ndarray:
    """
    ``B`` (R·C·2, 2): radial prior field of each node and channel per unit
    coefficient, ``Γ₀,ch · [r², r⁴]`` with ``r² = ((u−C_X)² + (v−C_Y)²)/f₀²``
    evaluated at the node — so ``Γ_prior = Γ₀ + B·a``, i.e.
    ``γ(r) = γ₀ (1 + a₁r² + a₂r⁴)``, the inverse of radial lens distortion.
    """
    s = sensor
    jj, ii = np.meshgrid(np.arange(s.grid_cols), np.arange(s.grid_rows))
    r2 = (((jj * s.cell_w - s.cx) ** 2 + (ii * s.cell_h - s.cy) ** 2) / s.f_init ** 2).ravel()
    B = np.zeros((2 * r2.size, 2))
    for ch in (0, 1):
        B[ch::2, 0] = s.gamma0[ch] * r2
        B[ch::2, 1] = s.gamma0[ch] * r2 * r2
    return B


def regulariser_hessian(sensor: Sensor, cfg: TrainConfig) -> np.ndarray:
    """
    Constant Hessian of the (exactly quadratic) regulariser.  Per camera, with
    deviation ``δ = Γ − Γ₀ − B·a`` (``a = 0`` without the radial prior):

        R = ½ δᵀ (λ_a I + λ_s L) δ = ½ zᵀ Mᵀ(λ_a I + λ_s L)M z,  z = [Γ − Γ₀; a],  M = [I, −B].
    """
    A = cfg.anchor * np.eye(sensor.grid_rows * sensor.grid_cols) + cfg.smooth * graph_laplacian(
        sensor.grid_rows, sensor.grid_cols)
    A2 = np.kron(A, np.eye(2))  # channels interleaved: index = node·2 + channel
    G = n_grid(sensor)
    P = n_params(sensor)
    H = np.zeros((P, P))
    B = radial_basis(sensor)
    for cam, a0 in ((0, 2 * G + 5), (1, 2 * G + 7)):
        idx = np.r_[cam * G:(cam + 1) * G, a0:a0 + 2]
        M = np.hstack([np.eye(G), -B]) if cfg.radial_prior else np.hstack([np.eye(G), np.zeros((G, 2))])
        H[np.ix_(idx, idx)] += M.T @ A2 @ M
    if cfg.radial_prior:
        H[2 * G + 5:, 2 * G + 5:] += 1e-8 * np.eye(N_RADIAL)  # keeps a identifiable when λ's are tiny
    return H


def prior_mean(sensor: Sensor, x: np.ndarray) -> np.ndarray:
    """θ₀: Γ₀ on the grids, a = 0; the rig block of θ₀ is irrelevant (zero Hessian) — copy x."""
    x0 = x.copy()
    G = n_grid(sensor)
    g0 = np.tile(sensor.gamma0, G // 2)
    x0[:G], x0[G:2 * G] = g0, g0
    x0[2 * G + 5:] = 0.0
    return x0


# ─── RESIDUAL JACOBIAN ────────────────────────────────────────────────────────
def residual_jacobian(theta: dict, pts: np.ndarray, sensor: Sensor):
    """
    Residuals ``e_x, e_y`` (N,) and their exact Jacobians ``J_x, J_y`` (N, P).

    Chain rule (see PDF §7):  ∂e_x/∂Γ_L,x = s ũ_L w_k;  ∂e/∂Γ_R through
    r̂ = W_xy/W_z, W = R·ray_R;  ∂e/∂ω_j = −s (∂r̂/∂W)(∂R/∂ω_j) ray_R;
    ∂e_x/∂dx = −(e_x+1)/dx,  ∂e_y/∂dx = −e_y/dx,  ∂e_y/∂dy = −1/dx.
    """
    from .training import _forward  # shared forward pass (avoids an import cycle)

    ex, ey, (rL, cL, rR, cR, R, dR, W, s, dx) = _forward(theta, pts, sensor)
    N, G = len(pts), n_grid(sensor)
    pts = pts[:, :5]
    P = n_params(sensor)
    Jx, Jy = np.zeros((N, P)), np.zeros((N, P))
    rows = np.repeat(np.arange(N)[:, None], 4, axis=1)

    np.add.at(Jx, (rows, 2 * cL["idx"]), (s * cL["un"])[:, None] * cL["w"])
    np.add.at(Jy, (rows, 2 * cL["idx"] + 1), (s * cL["vn"])[:, None] * cL["w"])

    wz = W[:, 2]
    zero = np.zeros_like(wz)
    dhx_dW = np.stack([1 / wz, zero, -W[:, 0] / wz ** 2], axis=1)
    dhy_dW = np.stack([zero, 1 / wz, -W[:, 1] / wz ** 2], axis=1)
    for J, dh in ((Jx, dhx_dW), (Jy, dhy_dW)):
        dh_drR = dh @ R
        np.add.at(J, (rows, G + 2 * cR["idx"]), (-s * dh_drR[:, 0] * cR["un"])[:, None] * cR["w"])
        np.add.at(J, (rows, G + 2 * cR["idx"] + 1), (-s * dh_drR[:, 1] * cR["vn"])[:, None] * cR["w"])
        for j, dRj in enumerate(dR):
            J[:, 2 * G + 2 + j] = -s * np.einsum("ij,ij->i", dh, rR @ dRj.T)
    Jx[:, 2 * G] = -(ex + 1.0) / dx
    Jy[:, 2 * G] = -ey / dx
    Jy[:, 2 * G + 1] = -1.0 / dx
    return ex, ey, Jx, Jy


def scaled_residuals(theta, pts, sensor, jacobian=False):
    """Residuals (and Jacobians) multiplied by the per-point noise scales."""
    from .training import _forward, residual_scales

    cx, cy = residual_scales(pts)
    if not jacobian:
        ex, ey = _forward(theta, pts, sensor)[:2]
        return cx * ex, cy * ey
    ex, ey, Jx, Jy = residual_jacobian(theta, pts, sensor)
    return cx * ex, cy * ey, cx[:, None] * Jx, cy[:, None] * Jy


def _irls_weights(e, delta):
    a = np.abs(e)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))


def _total(ex, ey, x, x0, H, N, cfg):
    from .training import huber

    dv = x - x0
    return (float(np.sum(huber(ex, cfg.huber_delta)[0] + cfg.weight_y * huber(ey, cfg.huber_delta)[0]))
            + N * 0.5 * float(dv @ H @ dv))


def _clamp(x, sensor):
    G = n_grid(sensor)
    x[:2 * G] = np.clip(x[:2 * G], G_MIN, G_MAX)
    x[2 * G] = max(x[2 * G], DX_MIN)
    return x


# ─── LEVENBERG–MARQUARDT ──────────────────────────────────────────────────────
def levenberg_marquardt(theta: dict, pts: np.ndarray, cfg: TrainConfig, sensor: Sensor,
                        max_iter: int = 200, tol: float = 1e-13):
    """
    Minimise F(θ) by LM on the IRLS-weighted normal equations

        (JᵀWJ + N·H_reg + μ·diag) δ = −(JᵀW e + N·H_reg (θ − θ₀)),

    accepting a step only if F decreases (so F is monotone).  Returns
    ``(theta, info)`` with ``info = {"iterations", "F", "history"}``.
    """
    N = len(pts)
    x = pack(theta)
    free = free_mask(sensor, cfg)
    H = regulariser_hessian(sensor, cfg)
    x0 = prior_mean(sensor, x)
    ex, ey = scaled_residuals(unpack(x, sensor), pts, sensor)
    F = _total(ex, ey, x, x0, H, N, cfg)
    mu, hist, it = 1e-3, [F], 0
    for it in range(1, max_iter + 1):  # noqa: B007 — iteration count is returned
        ex, ey, Jx, Jy = scaled_residuals(unpack(x, sensor), pts, sensor, jacobian=True)
        wx = _irls_weights(ex, cfg.huber_delta)
        wy = cfg.weight_y * _irls_weights(ey, cfg.huber_delta)
        g = Jx.T @ (wx * ex) + Jy.T @ (wy * ey) + N * H @ (x - x0)
        A = Jx.T @ (wx[:, None] * Jx) + Jy.T @ (wy[:, None] * Jy) + N * H
        gf, Af = g[free], A[np.ix_(free, free)]
        improved = False
        while mu < 1e12:
            step = np.linalg.solve(Af + mu * np.diag(np.maximum(np.diag(Af), 1e-12)), -gf)
            xn = x.copy()
            xn[free] += step
            _clamp(xn, sensor)
            exn, eyn = scaled_residuals(unpack(xn, sensor), pts, sensor)
            Fn = _total(exn, eyn, xn, x0, H, N, cfg)
            if np.isfinite(Fn) and Fn <= F:
                improved = True
                break
            mu *= 5.0
        if not improved:
            break
        rel = (F - Fn) / max(F, 1e-300)
        x, F = xn, Fn
        mu = max(mu * 0.3, 1e-12)
        hist.append(F)
        if rel < tol:
            break
    return unpack(x, sensor), {"iterations": it, "F": F, "history": hist}


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────
def diagnostics(theta: dict, pts: np.ndarray, cfg: TrainConfig, sensor: Sensor) -> dict:
    """
    Effective degrees of freedom and parameter standard errors at the optimum.

    * ``df_eff = tr[(JᵀWJ + N H_reg)⁻¹ JᵀWJ]`` — the trace of the hat matrix
      of the penalised fit (Hastie & Tibshirani 1990): how many parameters the
      data actually determine, as opposed to the nominal count.
    * Sandwich covariance of the penalised M-estimator,
      ``Cov ≈ A⁻¹ (σ_x² J_xᵀW_xJ_x + σ_y² J_yᵀW_yJ_y) A⁻¹`` with residual
      scales estimated from the fit, reported for the rig parameters.
    """
    N = len(pts)
    free = free_mask(sensor, cfg)
    H = regulariser_hessian(sensor, cfg)
    ex, ey, Jx, Jy = scaled_residuals(theta, pts, sensor, jacobian=True)
    wx = _irls_weights(ex, cfg.huber_delta)
    wy = cfg.weight_y * _irls_weights(ey, cfg.huber_delta)
    Bx = (Jx.T @ (wx[:, None] * Jx))[np.ix_(free, free)]
    By = (Jy.T @ (wy[:, None] * Jy))[np.ix_(free, free)]
    A = Bx + By + N * H[np.ix_(free, free)]
    Ainv = np.linalg.inv(A)
    df = float(np.trace(Ainv @ (Bx + By)))
    dof = max(2 * N - df, 1.0)
    s2x = float(np.sum(wx * ex ** 2)) / (dof / 2)
    s2y = float(np.sum(wy * ey ** 2)) / (dof / 2)
    cov = Ainv @ (s2x * Bx + s2y * By) @ Ainv
    G = n_grid(sensor)
    names = ["dx", "dy", "pitch", "yaw", "roll"]
    idx_full = np.flatnonzero(free)
    se = {}
    for k, name in enumerate(names):
        j = 2 * G + k
        if free[j]:
            p = int(np.flatnonzero(idx_full == j)[0])
            se[name] = float(np.sqrt(max(cov[p, p], 0.0)))
    return {"df_eff": df, "n_params_free": int(free.sum()), "n_residuals": 2 * N,
            "sigma_ex": float(np.sqrt(s2x)), "sigma_ey": float(np.sqrt(s2y)), "stderr": se}
