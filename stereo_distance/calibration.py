"""Gradient-based calibration of the gamma grids and the baseline.

Loss per calibration point ``(uL, vL, uR, vR, Z_true)``:

    disp_pred = r_Lx - r_Rx            disp_true = dx / Z_true
    e_x = disp_pred / disp_true - 1    L_1D = 1/2 e_x^2     (relative disparity)
    e_y = (r_Ly - r_Ry) - dy / Z_true  L_Y  = 1/2 e_y^2     (epipolar alignment)

The relative formulation treats a 5 % depth error at 1 m like one at 10 m,
avoiding the ``Z^2`` gradient blow-up of a depth-space loss.  All gradients
are derived by hand with the chain rule (``dr_x/dGamma_k = u_n w_k``) - there
is no autograd anywhere in the project.

The original trainer is plain batch gradient descent with gradient clipping,
a Laplacian smoothness term and a frozen centre node.  Two options described
in the author's derivation document are available as opt-ins and are off by
default so the shipped behaviour is unchanged: a Huber loss on ``e_x, e_y``
(``huber_delta``) and the Adam optimiser (``optimizer="adam"``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_RIG, RigConfig
from .lens import build_ray, build_rays_batch, default_params, laplacian_grad
from .triangulation import triangulate_batch

CalPoint = tuple[float, float, float, float, float]
"""``(uL, vL, uR, vR, Z_true_m)``."""


def _huber_grad(e: np.ndarray, delta: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(loss, dloss/de)`` for a squared or Huber penalty."""
    if delta is None:
        return 0.5 * e * e, e
    a = np.abs(e)
    loss = np.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    grad = np.where(a <= delta, e, delta * np.sign(e))
    return loss, grad


def forward_grads(
    u_l: float,
    v_l: float,
    u_r: float,
    v_r: float,
    z_true: float,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
    huber_delta: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Loss and analytic gradients for one calibration point.

    Returns:
        ``(loss, grad_gamma_l, grad_gamma_r, grad_d)`` with the grid gradients
        shaped like the grids and ``grad_d`` of shape ``(3,)`` (``dz`` is never
        trained, its gradient is 0).
    """
    ray_l, info_l = build_ray(u_l, v_l, gamma_l, rig)
    ray_r, info_r = build_ray(u_r, v_r, gamma_r, rig)

    denom = float(ray_l[0]) - float(ray_r[0])
    if abs(denom) < 1e-12:
        return 0.0, np.zeros(rig.grid_shape), np.zeros(rig.grid_shape), np.zeros(3)

    dx = max(float(d[0]), rig.dx_min)

    disp_true = dx / z_true
    rel_err = (denom / disp_true) - 1.0
    loss_1d, g_1d = _huber_grad(np.array(rel_err), huber_delta)
    dl_ddenom = float(g_1d) * (z_true / dx)
    dl_dvlx, dl_dvrx = dl_ddenom, -dl_ddenom

    e_y = (float(ray_l[1]) - float(ray_r[1])) - (float(d[1]) / z_true)
    loss_y, g_y = _huber_grad(np.array(e_y), huber_delta)
    dl_dvly, dl_dvry = float(g_y), -float(g_y)

    loss = float(loss_1d) + float(loss_y)

    grad_d = np.zeros(3)
    grad_d[0] = float(g_1d) * (-denom * z_true / (dx * dx))
    grad_d[1] = -float(g_y) / z_true

    grad_gl = np.zeros(rig.grid_shape)
    s_l = dl_dvlx * info_l["un"] + dl_dvly * info_l["vn"]
    for k in range(4):
        grad_gl[info_l["idx"][k]] += s_l * info_l["w"][k]

    grad_gr = np.zeros(rig.grid_shape)
    s_r = dl_dvrx * info_r["un"] + dl_dvry * info_r["vn"]
    for k in range(4):
        grad_gr[info_r["idx"][k]] += s_r * info_r["w"][k]

    return loss, grad_gl, grad_gr, grad_d


def forward_grads_batch(
    pts: np.ndarray,
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
    huber_delta: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`forward_grads` summed over ``N`` points.

    ``pts`` is an ``(N, 5)`` array of ``(uL, vL, uR, vR, Z)`` rows.  The result
    equals the sum of the per-point values to floating-point precision; the
    trainer uses it so that 9000 epochs on a few dozen points take about a
    second instead of a minute.
    """
    pts = np.asarray(pts, dtype=np.float64)
    rays_l, un_l, vn_l, w_l, rows_l, cols_l = build_rays_batch(pts[:, 0], pts[:, 1], gamma_l, rig)
    rays_r, un_r, vn_r, w_r, rows_r, cols_r = build_rays_batch(pts[:, 2], pts[:, 3], gamma_r, rig)
    z = pts[:, 4]

    denom = rays_l[:, 0] - rays_r[:, 0]
    valid = np.abs(denom) >= 1e-12
    dx = max(float(d[0]), rig.dx_min)

    rel_err = denom * z / dx - 1.0
    loss_1d, g_1d = _huber_grad(rel_err, huber_delta)
    dl_ddenom = g_1d * (z / dx)

    e_y = (rays_l[:, 1] - rays_r[:, 1]) - float(d[1]) / z
    loss_y, g_y = _huber_grad(e_y, huber_delta)

    loss_1d = np.where(valid, loss_1d, 0.0)
    loss_y = np.where(valid, loss_y, 0.0)
    dl_ddenom = np.where(valid, dl_ddenom, 0.0)
    g_1d = np.where(valid, g_1d, 0.0)
    g_y = np.where(valid, g_y, 0.0)

    grad_d = np.zeros(3)
    grad_d[0] = float((g_1d * (-denom * z / (dx * dx))).sum())
    grad_d[1] = float((-g_y / z).sum())

    s_l = dl_ddenom * un_l + g_y * vn_l
    s_r = -dl_ddenom * un_r - g_y * vn_r
    grad_gl = np.zeros(rig.grid_shape)
    grad_gr = np.zeros(rig.grid_shape)
    np.add.at(grad_gl, (rows_l.ravel(), cols_l.ravel()), (s_l[:, None] * w_l).ravel())
    np.add.at(grad_gr, (rows_r.ravel(), cols_r.ravel()), (s_r[:, None] * w_r).ravel())

    return float(loss_1d.sum() + loss_y.sum()), grad_gl, grad_gr, grad_d


@dataclass
class TrainConfig:
    """Hyper-parameters of :func:`train`.  Defaults reproduce ``run_math.py``."""

    epochs: int = 9000
    lr_gamma: float = 1e-1
    lr_d: float = 1e-3
    clip_norm: float = 1e3
    optimizer: str = "sgd"  # "sgd" | "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    huber_delta: float | None = None
    verbose_every: int = 300


@dataclass
class CalibrationResult:
    """Output of :func:`train`."""

    gamma_l: np.ndarray
    gamma_r: np.ndarray
    d: np.ndarray
    best_loss: float
    best_epoch: int
    history: list[tuple[int, float, float]] = field(default_factory=list)
    """``(epoch, loss, depth_MAE_m)`` rows, every ``verbose_every`` epochs."""

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.gamma_l, self.gamma_r, self.d


def depth_mae(
    cal_pts: Sequence[CalPoint],
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
) -> float:
    """Mean absolute depth error (m) over the valid points; ``nan`` if none."""
    pts = np.asarray(cal_pts, dtype=np.float64)
    if pts.size == 0:
        return float("nan")
    z = triangulate_batch(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], gamma_l, gamma_r, d, rig)
    ok = ~np.isnan(z)
    if not ok.any():
        return float("nan")
    return float(np.abs(z[ok] - pts[ok, 4]).mean())


def warm_start_baseline(cal_pts: Sequence[CalPoint], rig: RigConfig = DEFAULT_RIG) -> float:
    """Initial ``dx`` from the mean pixel disparity: ``dx ~ mean(Z) * mean(disp) * g0 / cx``."""
    pts = np.asarray(cal_pts, dtype=np.float64)
    mean_disp_ray = float(np.abs(pts[:, 0] - pts[:, 2]).mean()) * (rig.g0 / rig.cx)
    mean_z = float(pts[:, 4].mean())
    return max(mean_z * mean_disp_ray, rig.dx_min)


def train(
    cal_pts: Sequence[CalPoint],
    cfg: TrainConfig | None = None,
    rig: RigConfig = DEFAULT_RIG,
    log: Callable[[str], None] | None = print,
    init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CalibrationResult:
    """Fit ``GammaL, GammaR, d`` to labelled point pairs by gradient descent.

    Each epoch: average the analytic data gradient over all points, add the
    Laplacian smoothness gradient, zero the centre-node gradient (the frozen
    node anchors the absolute focal scale and removes the ``dx``/``Gamma``
    degeneracy), clip the gradient norm, step, then clamp ``dx >= dx_min``,
    ``g_min <= Gamma <= g_max`` and re-pin the centre node.  The parameters of
    the lowest-loss epoch are returned.
    """
    cfg = cfg or TrainConfig()
    pts = np.asarray(cal_pts, dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("train() needs at least one calibration point")
    ci, cj = rig.centre

    if init is None:
        gamma_l, gamma_r, d = default_params(rig)
        d[0] = warm_start_baseline(cal_pts, rig)
    else:
        gamma_l, gamma_r, d = (np.array(p, dtype=np.float64) for p in init)

    if log:
        log(
            f"[TRAIN] N={n} epochs={cfg.epochs} grid={rig.grid_rows}x{rig.grid_cols} "
            f"smooth_w={rig.smooth_w} optimizer={cfg.optimizer} huber={cfg.huber_delta}"
        )
        log(f"[TRAIN] warm-start dx={d[0]:.4f} m  mean_Z={pts[:, 4].mean():.2f} m  g0={rig.g0:.4f}")

    params = [gamma_l, gamma_r, d]
    lrs = [cfg.lr_gamma, cfg.lr_gamma, cfg.lr_d]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    best_loss, best_epoch, best_state = float("inf"), -1, None
    history: list[tuple[int, float, float]] = []

    for ep in range(cfg.epochs):
        total, g_l, g_r, g_d = forward_grads_batch(pts, gamma_l, gamma_r, d, rig, cfg.huber_delta)
        g_l /= n
        g_r /= n
        g_d /= n

        g_l += rig.smooth_w * laplacian_grad(gamma_l)
        g_r += rig.smooth_w * laplacian_grad(gamma_r)

        g_l[ci, cj] = 0.0
        g_r[ci, cj] = 0.0

        grads = [g_l, g_r, g_d]
        for g in grads:
            nrm = float(np.linalg.norm(g))
            if nrm > cfg.clip_norm:
                g *= cfg.clip_norm / nrm

        if cfg.optimizer == "adam":
            t = ep + 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = cfg.adam_beta1 * m[i] + (1 - cfg.adam_beta1) * g
                v[i] = cfg.adam_beta2 * v[i] + (1 - cfg.adam_beta2) * g * g
                m_hat = m[i] / (1 - cfg.adam_beta1**t)
                v_hat = v[i] / (1 - cfg.adam_beta2**t)
                p -= lrs[i] * m_hat / (np.sqrt(v_hat) + cfg.adam_eps)
        elif cfg.optimizer == "sgd":
            for p, g, lr in zip(params, grads, lrs):
                p -= lr * g
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

        d[2] = 0.0
        d[0] = max(d[0], rig.dx_min)
        np.clip(gamma_l, rig.g_min, rig.g_max, out=gamma_l)
        np.clip(gamma_r, rig.g_min, rig.g_max, out=gamma_r)
        gamma_l[ci, cj] = rig.g0
        gamma_r[ci, cj] = rig.g0

        if total < best_loss:
            best_loss, best_epoch = total, ep
            best_state = (gamma_l.copy(), gamma_r.copy(), d.copy())

        if cfg.verbose_every and (ep % cfg.verbose_every == 0 or ep == cfg.epochs - 1):
            mae = depth_mae(cal_pts, gamma_l, gamma_r, d, rig)
            history.append((ep, total, mae))
            if log:
                log(
                    f"  ep {ep:5d}  L={total:9.4f}  MAE={mae:.3f}m  dx={d[0]:.4f}m  "
                    f"|GL|={np.linalg.norm(gamma_l):.4f}  |GR|={np.linalg.norm(gamma_r):.4f}"
                )

    assert best_state is not None
    if log:
        log(f"[TRAIN] best L = {best_loss:.6f} at epoch {best_epoch}")
    return CalibrationResult(*best_state, best_loss, best_epoch, history)


def loss_only(
    cal_pts: Sequence[CalPoint],
    gamma_l: np.ndarray,
    gamma_r: np.ndarray,
    d: np.ndarray,
    rig: RigConfig = DEFAULT_RIG,
    huber_delta: float | None = None,
) -> float:
    """Total data loss over the points (no smoothness term)."""
    total = 0.0
    for p in cal_pts:
        total += forward_grads(*p, gamma_l, gamma_r, d, rig, huber_delta)[0]
    return total if math.isfinite(total) else float("inf")
