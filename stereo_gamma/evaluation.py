"""
Evaluation protocol: accuracy metrics, leave-one-out cross-validation and
reference baselines used in the README ablation study.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import DEFAULT_SENSOR, Sensor, TrainConfig
from .training import train


def metrics(Z_pred, Z_true) -> dict:
    """Standard depth-estimation metrics (Eigen et al. 2014 style, plus MAE)."""
    Z_pred, Z_true = np.asarray(Z_pred, float), np.asarray(Z_true, float)
    ok = np.isfinite(Z_pred)
    p, t = Z_pred[ok], Z_true[ok]
    ae, are = np.abs(p - t), np.abs(p / t - 1)
    ratio = np.maximum(p / t, t / p)
    return {
        "n": int(len(Z_true)),
        "n_valid": int(ok.sum()),
        "mae_m": float(ae.mean()),
        "rmse_m": float(np.sqrt((ae ** 2).mean())),
        "mape_pct": float(100 * are.mean()),
        "median_ape_pct": float(100 * np.median(are)),
        "max_ape_pct": float(100 * are.max()),
        "delta_1.05_pct": float(100 * (ratio < 1.05).mean()),
        "delta_1.10_pct": float(100 * (ratio < 1.10).mean()),
    }


def loocv(points, cfg: TrainConfig | None = None, sensor: Sensor = DEFAULT_SENSOR):
    """
    Leave-one-out cross-validation: train on N−1 points, predict the held-out
    one, repeat N times.  Returns the out-of-sample predictions ``(N,)``.
    """
    cfg = replace(cfg or TrainConfig(), log_every=0)
    pts = np.asarray(points, float)
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        model, _ = train(np.delete(pts, i, axis=0), cfg, sensor, log=None)
        out[i] = model.triangulate(*pts[i:i + 1, :4].T)[0][0]
    return out


# ─── REFERENCE BASELINE ───────────────────────────────────────────────────────
def fit_disparity_offset(points):
    """
    Classical 2-parameter baseline: horizontal pixel disparity
    Δu = uL − uR = a/Z + b   (pinhole with a constant offset from camera yaw).
    Returns ``(a, b)``.
    """
    pts = np.asarray(points, float)
    du, Z = pts[:, 0] - pts[:, 2], pts[:, 4]
    A = np.stack([1.0 / Z, np.ones_like(Z)], axis=1)
    return np.linalg.lstsq(A * Z[:, None], du * Z, rcond=None)[0]  # relative weighting


def loocv_disparity_offset(points):
    pts = np.asarray(points, float)
    out = np.empty(len(pts))
    for i in range(len(pts)):
        a, b = fit_disparity_offset(np.delete(pts, i, axis=0))
        du = pts[i, 0] - pts[i, 2]
        out[i] = a / (du - b) if du > b else np.nan
    return out
