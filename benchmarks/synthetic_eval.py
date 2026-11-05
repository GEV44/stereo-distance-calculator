"""Depth-accuracy benchmark for the pure-gamma stereo model.

Everything here is reproducible from fixed seeds; nothing is hand-tuned per
result.  Three experiments:

1. **Error vs distance** (synthetic).  A rig shaped like the author's
   (2592x1944 px, f = 2880 px, 0.27 m baseline, ~8 % smooth lens distortion).
   Test points at 0.5-10 m, Gaussian pixel noise on all four click coordinates.
   Two parameter sets are compared:
     - *oracle*      : the true gamma grids and baseline;
     - *calibrated*  : grids + baseline learned by ``train()`` from 40 noisy
                       synthetic calibration points at 0.8-6 m (default config,
                       9000 epochs, plain gradient descent).
   The first-order stereo bound  sigma_Z = Z^2 * sqrt(2) * sigma_px / (f * b)
   is drawn for reference.
2. **Error vs pixel noise** (synthetic).  Same rig, sigma = 0..3 px; for every
   noise level a fresh calibration is trained on noisy points and evaluated on
   noisy held-out points at 0.8-6 m.
3. **Real data, leave-one-out** (``cal_pts.json``, 27 hand-clicked pairs with
   tape-measured distances, 0.41-4.74 m).  For each point the model is trained
   on the other 26 and asked for the held-out depth.  No real left/right
   images are in the repository, so this is the only real-data number.

Outputs: ``docs/img/error_vs_distance.png``, ``docs/img/error_vs_noise.png``,
``docs/img/hero.png``, ``benchmarks/results.md`` and ``benchmarks/results.json``.

Usage::

    python benchmarks/synthetic_eval.py            # full run, a few minutes
    python benchmarks/synthetic_eval.py --quick    # fewer epochs/points, smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from stereo_distance import (  # noqa: E402
    DEFAULT_RIG,
    SyntheticRig,
    TrainConfig,
    __version__,
    load_cal_pts,
    load_calibration,
    train,
    triangulate_batch,
)

# Reference palette (dataviz skill, light mode, first three categorical slots)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

BASELINE_M = 0.27
FOCAL_PX = DEFAULT_RIG.f_init


def make_truth(seed: int = 11) -> SyntheticRig:
    return SyntheticRig.distorted(amplitude=0.08, dx=BASELINE_M, dy=0.003, seed=seed)


def test_points(truth: SyntheticRig, z: float, n: int, rng: np.random.Generator):
    """``n`` visible points at depth ``z`` spread over the left image."""
    rig = truth.rig
    u = rng.uniform(150, rig.img_w - 150, 3 * n)
    v = rng.uniform(150, rig.img_h - 150, 3 * n)
    x = (u - rig.cx) / rig.cx * rig.g0 * z
    y = (v - rig.cy) / rig.cy * rig.g0 * z
    pts = np.column_stack([x, y, np.full_like(x, z)])
    left, right = truth.project(pts)
    ok = truth.in_frame(left, right, margin=2.0)
    left, right = left[ok][:n], right[ok][:n]
    return left, right


def depth_errors(left, right, params, sigma, rng, z_true):
    noisy_l = left + rng.normal(0, sigma, left.shape) if sigma > 0 else left
    noisy_r = right + rng.normal(0, sigma, right.shape) if sigma > 0 else right
    z = triangulate_batch(noisy_l[:, 0], noisy_l[:, 1], noisy_r[:, 0], noisy_r[:, 1], *params)
    z_true = np.broadcast_to(z_true, z.shape)
    ok = ~np.isnan(z)
    abs_err = np.abs(z[ok] - z_true[ok])
    return abs_err, abs_err / z_true[ok], int((~ok).sum())


def calibrate(truth: SyntheticRig, sigma: float, n_cal: int, epochs: int, seed: int):
    pts = truth.make_cal_pts(n_cal, z_range=(0.8, 6.0), noise_px=sigma, seed=seed)
    res = train(pts, TrainConfig(epochs=epochs, verbose_every=0), log=None)
    return res


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


# --------------------------------------------------------------------------
def exp_distance(truth, calib_params, sigma, n_per_z, seed):
    rng = np.random.default_rng(seed)
    oracle = (truth.gamma_l, truth.gamma_r, truth.d)
    rows = []
    for z in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        left, right = test_points(truth, z, n_per_z, rng)
        noise_rng = np.random.default_rng(seed + int(z * 100))
        o_abs, o_rel, o_fail = depth_errors(left, right, oracle, sigma, noise_rng, z)
        noise_rng = np.random.default_rng(seed + int(z * 100))  # identical noise for both
        c_abs, c_rel, c_fail = depth_errors(left, right, calib_params, sigma, noise_rng, z)
        rows.append({
            "z_m": z,
            "n": int(len(left)),
            "theory_sigma_m": z * z * np.sqrt(2) * sigma / (FOCAL_PX * BASELINE_M),
            "oracle_mae_m": float(o_abs.mean()),
            "oracle_rel_pct": float(100 * o_rel.mean()),
            "calib_mae_m": float(c_abs.mean()),
            "calib_rel_pct": float(100 * c_rel.mean()),
            "calib_p90_rel_pct": float(100 * np.percentile(c_rel, 90)),
            "failed": o_fail + c_fail,
        })
    return rows


def exp_noise(truth, sigmas, n_cal, epochs, n_test, seed):
    oracle = (truth.gamma_l, truth.gamma_r, truth.d)
    rows = []
    for k, sigma in enumerate(sigmas):
        t0 = time.perf_counter()
        res = calibrate(truth, sigma, n_cal, epochs, seed=seed + k)
        t_train = time.perf_counter() - t0
        test = truth.make_cal_pts(n_test, z_range=(0.8, 6.0), noise_px=0.0, seed=seed + 100 + k)
        left, right, z_true = test[:, 0:2], test[:, 2:4], test[:, 4]
        o_abs, o_rel, _ = depth_errors(left, right, oracle, sigma, np.random.default_rng(seed + k), z_true)
        c_abs, c_rel, _ = depth_errors(left, right, res.params, sigma, np.random.default_rng(seed + k), z_true)
        rows.append({
            "sigma_px": sigma,
            "oracle_mae_m": float(o_abs.mean()),
            "oracle_rel_pct": float(100 * o_rel.mean()),
            "calib_mae_m": float(c_abs.mean()),
            "calib_rel_pct": float(100 * c_rel.mean()),
            "calib_median_rel_pct": float(100 * np.median(c_rel)),
            "learned_dx_m": float(res.d[0]),
            "train_seconds": round(t_train, 2),
        })
        print(f"  sigma={sigma:4.2f}px  calib MAE={c_abs.mean():.4f} m  rel={100 * c_rel.mean():.2f}%  "
              f"dx={res.d[0]:.4f}  ({t_train:.1f}s)")
    return rows


def exp_real_loo(epochs):
    pts = load_cal_pts(REPO / "cal_pts.json")
    pts_arr = np.asarray(pts)
    preds = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        train_pts = [p for j, p in enumerate(pts) if j != i]
        res = train(train_pts, TrainConfig(epochs=epochs, verbose_every=0), log=None)
        p = pts_arr[i]
        preds[i] = triangulate_batch(p[[0]], p[[1]], p[[2]], p[[3]], *res.params)[0]
    full = train(pts, TrainConfig(epochs=epochs, verbose_every=0), log=None)
    fit = triangulate_batch(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], pts_arr[:, 3], *full.params)
    gl, gr, d = load_calibration(REPO / "stereo_calibration.json")
    shipped = triangulate_batch(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], pts_arr[:, 3], gl, gr, d)
    z = pts_arr[:, 4]
    ok = ~np.isnan(preds)
    rel = np.abs(preds[ok] - z[ok]) / z[ok]
    return {
        "n": len(pts),
        "z_range_m": [float(z.min()), float(z.max())],
        "loo_valid": int(ok.sum()),
        "loo_mae_m": float(np.abs(preds[ok] - z[ok]).mean()),
        "loo_median_abs_m": float(np.median(np.abs(preds[ok] - z[ok]))),
        "loo_rel_pct": float(100 * rel.mean()),
        "loo_median_rel_pct": float(100 * np.median(rel)),
        "loo_within_5pct": int((rel <= 0.05).sum()),
        "train_fit_mae_m": float(np.nanmean(np.abs(fit - z))),
        "shipped_calib_mae_m": float(np.nanmean(np.abs(shipped - z))),
        "learned_dx_m": float(full.d[0]),
        "z_true": z.tolist(),
        "z_loo": preds.tolist(),
        "gamma_l": full.gamma_l.tolist(),
        "gamma_r": full.gamma_r.tolist(),
    }


# --------------------------------------------------------------------------
def plot_distance(rows, sigma, out):
    import matplotlib.pyplot as plt

    z = np.array([r["z_m"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.plot(z, [100 * r["theory_sigma_m"] for r in rows], color=INK_2, lw=1.5, ls="--",
            label=r"first-order bound $\sigma_Z = Z^2\sqrt{2}\,\sigma_{px}/(f\,b)$")
    ax.plot(z, [100 * r["oracle_mae_m"] for r in rows], color=BLUE, lw=2, marker="o", ms=5,
            label="true parameters (oracle)")
    ax.plot(z, [100 * r["calib_mae_m"] for r in rows], color=ORANGE, lw=2, marker="s", ms=5,
            label="calibrated from 40 noisy points")
    ax.axvspan(0.8, 6.0, color=AQUA, alpha=0.08, lw=0)
    ax.set_xlabel("true distance Z (m)")
    ax.set_ylabel("mean absolute depth error (cm)")
    ax.set_yscale("log")
    ax.set_title(f"Depth error vs distance  -  synthetic rig, {sigma} px click noise",
                 fontsize=11, loc="left")
    ax.text(0.83, 0.02, "calibration range 0.8-6 m", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK_2)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_noise(rows, out):
    import matplotlib.pyplot as plt

    s = np.array([r["sigma_px"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.plot(s, [r["oracle_rel_pct"] for r in rows], color=BLUE, lw=2, marker="o", ms=5,
            label="true parameters (oracle)")
    ax.plot(s, [r["calib_rel_pct"] for r in rows], color=ORANGE, lw=2, marker="s", ms=5,
            label="calibrated on 40 points with the same noise")
    ax.set_xlabel(r"pixel noise $\sigma$ on every click coordinate (px)")
    ax.set_ylabel("mean relative depth error (%)")
    ax.set_ylim(0, 3.6)
    ax.set_title("Depth error vs pixel noise  -  synthetic rig, test points 0.8-6 m",
                 fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.33))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_hero(real, rig, out):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 3.9), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.28, 1.15], wspace=0.12)
    fl = rig.cx / np.asarray(real["gamma_l"])
    fr = rig.cx / np.asarray(real["gamma_r"])
    vmin, vmax = min(fl.min(), fr.min()), max(fl.max(), fr.max())
    im = None
    for k, (f, name) in enumerate(((fl, "Left camera"), (fr, "Right camera"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(f, cmap="Blues", vmin=vmin, vmax=vmax, interpolation="bilinear",
                       extent=(0, rig.img_w, rig.img_h, 0))
        ax.set_title(f"{name}: learned focal map", fontsize=10, color=INK, loc="left")
        ax.set_xticks([0, rig.img_w])
        ax.set_yticks([0, rig.img_h] if k == 0 else [])
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ys, xs = np.mgrid[0:5, 0:5]
        ax.scatter(xs.ravel() * rig.cell_w, ys.ravel() * rig.cell_h, s=10, c=SURFACE,
                   edgecolors=INK_2, linewidths=0.6, zorder=3, clip_on=False)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cx / gamma (px)", color=INK, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 4])
    style_axes(ax)
    z = np.asarray(real["z_true"])
    p = np.asarray(real["z_loo"])
    lim = (0, max(z.max(), np.nanmax(p)) * 1.08)
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
    ax.scatter(z, p, s=28, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("tape-measured distance (m)", fontsize=9)
    ax.set_ylabel("predicted, point held out (m)", fontsize=9)
    ax.set_title(f"Real rig, leave-one-out (N={real['n']}): median error "
                 f"{real['loo_median_rel_pct']:.1f} %", fontsize=10, loc="left")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def write_markdown(results, out):
    d = results["distance"]
    nz = results["noise"]
    real = results["real_loo"]
    sigma = results["config"]["sigma_distance_px"]
    lines = [
        "# Benchmark results",
        "",
        f"Generated by `benchmarks/synthetic_eval.py` (stereo_distance {results['config']['version']}, "
        f"{results['config']['epochs']} epochs, numpy {results['config']['numpy']}). "
        "Synthetic rig: 2592x1944 px, f = 2880 px, baseline 0.27 m, ~8 % smooth gamma distortion.",
        "",
        f"## Error vs distance ({sigma} px noise on every click coordinate)",
        "",
        "| Z (m) | bound sigma_Z (cm) | oracle MAE (cm) | calibrated MAE (cm) | calibrated rel. (%) | p90 rel. (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in d:
        lines.append(
            f"| {r['z_m']:g} | {100 * r['theory_sigma_m']:.2f} | {100 * r['oracle_mae_m']:.2f} | "
            f"{100 * r['calib_mae_m']:.2f} | {r['calib_rel_pct']:.2f} | {r['calib_p90_rel_pct']:.2f} |"
        )
    lines += [
        "",
        "## Error vs pixel noise (test points 0.8-6 m)",
        "",
        "| sigma (px) | oracle rel. (%) | calibrated MAE (cm) | calibrated rel. (%) "
        "| median rel. (%) | learned dx (m) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in nz:
        lines.append(
            f"| {r['sigma_px']:g} | {r['oracle_rel_pct']:.2f} | {100 * r['calib_mae_m']:.2f} | "
            f"{r['calib_rel_pct']:.2f} | {r['calib_median_rel_pct']:.2f} | {r['learned_dx_m']:.4f} |"
        )
    lines += [
        "",
        "## Real rig, leave-one-out (`cal_pts.json`)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| points (tape-measured) | {real['n']} ({real['z_range_m'][0]:.2f}-{real['z_range_m'][1]:.2f} m) |",
        f"| LOO mean abs. error | {100 * real['loo_mae_m']:.1f} cm |",
        f"| LOO median abs. error | {100 * real['loo_median_abs_m']:.1f} cm |",
        f"| LOO mean rel. error | {real['loo_rel_pct']:.1f} % |",
        f"| LOO median rel. error | {real['loo_median_rel_pct']:.1f} % |",
        f"| held-out points within 5 % | {real['loo_within_5pct']} / {real['loo_valid']} |",
        f"| training-set fit MAE (all 27) | {100 * real['train_fit_mae_m']:.1f} cm |",
        f"| shipped `stereo_calibration.json` MAE | {100 * real['shipped_calib_mae_m']:.1f} cm |",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer epochs and points (smoke test)")
    ap.add_argument("--no-real", action="store_true", help="skip the leave-one-out real-data run")
    ap.add_argument("--replot", action="store_true",
                    help="only redraw figures and results.md from benchmarks/results.json")
    args = ap.parse_args(argv)

    if args.replot:
        results = json.loads((REPO / "benchmarks" / "results.json").read_text(encoding="utf-8"))
        img = REPO / "docs" / "img"
        plot_distance(results["distance"], results["config"]["sigma_distance_px"], img / "error_vs_distance.png")
        plot_noise(results["noise"], img / "error_vs_noise.png")
        if results["real_loo"] is not None:
            plot_hero(results["real_loo"], DEFAULT_RIG, img / "hero.png")
            write_markdown(results, REPO / "benchmarks" / "results.md")
        print("replotted from benchmarks/results.json")
        return

    epochs = 1500 if args.quick else TrainConfig.epochs
    n_cal, n_test, n_per_z = 40, (300 if args.quick else 2000), (100 if args.quick else 400)
    sigma_d = 0.5
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    img = REPO / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    truth = make_truth()
    print(f"[1/3] calibrating on {n_cal} points with {sigma_d} px noise ({epochs} epochs)")
    res = calibrate(truth, sigma_d, n_cal, epochs, seed=7)
    print(f"      learned dx = {res.d[0]:.4f} m (truth {BASELINE_M})")
    dist_rows = exp_distance(truth, res.params, sigma_d, n_per_z, seed=21)
    for r in dist_rows:
        print(f"      Z={r['z_m']:5.2f} m  oracle {100 * r['oracle_mae_m']:6.2f} cm  "
              f"calibrated {100 * r['calib_mae_m']:6.2f} cm")

    print("[2/3] noise sweep")
    noise_rows = exp_noise(truth, sigmas, n_cal, epochs, n_test, seed=31)

    real = None
    if not args.no_real:
        print("[3/3] real data, leave-one-out over cal_pts.json")
        real = exp_real_loo(epochs)
        print(f"      LOO MAE = {100 * real['loo_mae_m']:.1f} cm, median rel = "
              f"{real['loo_median_rel_pct']:.1f} %")

    results = {
        "config": {
            "version": __version__, "epochs": epochs, "n_cal": n_cal, "n_test": n_test,
            "n_per_distance": n_per_z, "sigma_distance_px": sigma_d, "baseline_m": BASELINE_M,
            "focal_px": FOCAL_PX, "numpy": np.__version__, "python": platform.python_version(),
            "runtime_s": None,
        },
        "distance": dist_rows,
        "noise": noise_rows,
        "real_loo": real,
    }
    plot_distance(dist_rows, sigma_d, img / "error_vs_distance.png")
    plot_noise(noise_rows, img / "error_vs_noise.png")
    if real is not None:
        plot_hero(real, DEFAULT_RIG, img / "hero.png")
        write_markdown(results, REPO / "benchmarks" / "results.md")
    results["config"]["runtime_s"] = round(time.perf_counter() - t_start, 1)
    (REPO / "benchmarks" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"done in {results['config']['runtime_s']} s -> docs/img/*.png, benchmarks/results.*")


if __name__ == "__main__":
    main()
