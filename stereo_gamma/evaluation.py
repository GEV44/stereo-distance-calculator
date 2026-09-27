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


def loocv(points, cfg: TrainConfig | None = None, sensor: Sensor = DEFAULT_SENSOR,
          with_details: bool = False):
    """
    Leave-one-out cross-validation: train on N−1 points, predict the held-out
    one, repeat N times.  Every hyper-parameter choice (noise model, λ_s) is
    made inside ``train`` from the N−1 training points only, so the estimate
    is *nested* and free of selection bias.

    Returns the out-of-sample predictions ``(N,)``; with ``with_details`` also
    a dict of per-point arrays: ``sigma`` (the model's predicted ±1σ),
    ``z_gamma`` and ``z_companion`` (the two ensemble members).
    """
    cfg = replace(cfg or TrainConfig(), log_every=0)
    pts = np.asarray(points, float)
    n = len(pts)
    out = np.full(n, np.nan)
    det = {k: np.full(n, np.nan) for k in ("sigma", "z_gamma", "z_companion")}
    for i in range(n):
        model, _ = train(np.delete(pts, i, axis=0), cfg, sensor, log=None)
        q = pts[i:i + 1, :4].T
        out[i] = model.triangulate(*q)[0][0]
        if with_details:
            det["sigma"][i] = model.depth_uncertainty(*pts[i, :4])[1]
            det["z_gamma"][i] = model.triangulate_gamma(*q)[0][0]
            if model.companion is not None:
                det["z_companion"][i] = model.companion_model().triangulate(*q)[0]
    return (out, det) if with_details else out


def coverage(Z_pred, sigma, Z_true) -> dict:
    """
    Calibration of the predicted uncertainty: fraction of |error| within
    1σ / 2σ (Gaussian ideal: 68.3 % / 95.4 %) and the RMS of the z-scores
    (ideal 1).
    """
    z = (np.asarray(Z_pred) - np.asarray(Z_true)) / np.asarray(sigma)
    z = z[np.isfinite(z)]
    return {"within_1sigma_pct": float(100 * np.mean(np.abs(z) <= 1)),
            "within_2sigma_pct": float(100 * np.mean(np.abs(z) <= 2)),
            "rms_z": float(np.sqrt(np.mean(z * z))),
            "median_abs_z": float(np.median(np.abs(z)))}


def signflip_test(err_a, err_b, n: int = 20000, seed: int = 0) -> dict:
    """
    Paired two-sided sign-flip permutation test on per-point absolute errors
    (exact under the null of exchangeable errors; no normality assumption).
    Returns the mean difference ``a − b`` and its p-value.
    """
    d = np.abs(np.asarray(err_a)) - np.abs(np.asarray(err_b))
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], (n, len(d))) * d).mean(1)
    return {"mean_diff": float(d.mean()), "p_value": float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())}


def bootstrap_ci(values, n: int = 4000, seed: int = 0, level: float = 95.0):
    """Percentile bootstrap CI of the mean."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    means = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    a = (100 - level) / 2
    return float(np.percentile(means, a)), float(np.percentile(means, 100 - a))


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
