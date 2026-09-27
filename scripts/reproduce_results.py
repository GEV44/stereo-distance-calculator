"""
Reproduce every number and figure in the README and the PDF.

    python scripts/reproduce_results.py            # full run (~20 min on a laptop CPU)
    python scripts/reproduce_results.py --fast     # smoke run (fewer seeds, no rendering)

Outputs
    results/results.json            all metrics + provenance (machine-readable)
    results/RESULTS.md              the tables used in the README / PDF
    docs/figures/*.png              figures

Protocol.  Real-data numbers are leave-one-out cross-validation in which every
data-dependent choice (noise model, regularisation strength λ_s) is made
inside ``train`` from the N−1 training points only (nested CV).  Synthetic
numbers are held-out errors on 500 fresh points per seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from stereo_gamma import __version__  # noqa: E402
from stereo_gamma.baselines import BrownConradyStereo  # noqa: E402
from stereo_gamma.config import TrainConfig  # noqa: E402
from stereo_gamma.data import load_points  # noqa: E402
from stereo_gamma.evaluation import (  # noqa: E402
    bootstrap_ci,
    coverage,
    fit_disparity_offset,
    loocv,
    loocv_disparity_offset,
    metrics,
    signflip_test,
)
from stereo_gamma.legacy import loocv_legacy, train_legacy  # noqa: E402
from stereo_gamma.solver import diagnostics  # noqa: E402
from stereo_gamma.synthetic import SyntheticRig  # noqa: E402
from stereo_gamma.training import train  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
RES = os.path.join(ROOT, "results")

# reference palette (validated with the dataviz skill's validator; see README "Figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, LGRAY, INK, INK2, GRID, SURFACE = "#9a9994", "#c9c8c3", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
    })
    return plt


def rel(z, Z):
    return np.asarray(z) / Z - 1


# ─── 1. REAL DATA ─────────────────────────────────────────────────────────────
LADDER = [
    ("legacy", "v1 original (reproduced exactly)"),
    ("offset", "Pinhole + disparity offset (2 params)"),
    ("brown", "Brown–Conrady, 18 params (same loss, FGLS)"),
    ("g_v20", "Γ grid + rotation, v2.0 loss"),
    ("g_fgls", "  + estimated noise model (FGLS)"),
    ("gamma", "  + radial prior  = Γ grid alone (v2.1)"),
    ("v21", "  + Brown–Conrady companion = **v2.1 default**"),
    ("v21_norot", "v2.1 without the rig rotation"),
]


def real_experiments(pts):
    Zt = pts[:, 4]
    base = TrainConfig()
    preds, t0 = {}, time.time()

    def run(key, fn):
        t = time.time()
        preds[key] = fn()
        print(f"  {key:10s} MAPE {100 * np.nanmean(np.abs(rel(preds[key], Zt))):5.2f}%  ({time.time() - t:.0f}s)",
              flush=True)

    run("legacy", lambda: loocv_legacy(pts))
    run("offset", lambda: loocv_disparity_offset(pts))
    run("g_v20", lambda: loocv(pts, replace(base, noise_model="relative", radial_prior=False, ensemble=False)))
    run("g_fgls", lambda: loocv(pts, replace(base, radial_prior=False, ensemble=False)))
    z, det = loocv(pts, base, with_details=True)
    preds["v21"], preds["gamma"], preds["brown"] = z, det["z_gamma"], det["z_companion"]
    print(f"  v21/gamma/brown from one nested LOOCV  ({time.time() - t0:.0f}s so far)", flush=True)
    run("v21_norot", lambda: loocv(pts, replace(base, learn_rotation=False)))

    rows = {}
    for key, name in LADDER:
        m = metrics(preds[key], Zt)
        m["mape_ci95"] = bootstrap_ci(100 * np.abs(rel(preds[key], Zt)))
        m["name"], m["pred"] = name, [float(v) for v in preds[key]]
        rows[key] = m

    # paired sign-flip permutation tests, Holm–Bonferroni corrected
    comps = [("v21", "legacy"), ("v21", "brown"), ("v21", "gamma"), ("gamma", "brown"), ("g_fgls", "g_v20")]
    tests = []
    for a, b in comps:
        t = signflip_test(rel(preds[a], Zt), rel(preds[b], Zt))
        tests.append({"a": a, "b": b, "mean_diff_pp": 100 * t["mean_diff"], "p": t["p_value"]})
    order = np.argsort([t["p"] for t in tests])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running

    cov = coverage(preds["v21"], det["sigma"], Zt)
    corr = float(np.corrcoef(rel(preds["gamma"], Zt), rel(preds["brown"], Zt))[0, 1])
    return rows, tests, cov, {"sigma": det["sigma"].tolist(), "error_correlation_gamma_brown": corr}


def full_model_analysis(pts):
    """Fit on all points: noise model, error budget, df_eff, parameter uncertainty (Laplace vs bootstrap)."""
    model, _ = train(pts, TrainConfig(), log=None)
    Zt = pts[:, 4]
    click_only = model.copy()
    click_only.meta["rel_model_error"] = 0.0
    s_click = np.array([click_only.depth_uncertainty(*p[:4])[2] for p in pts])
    pred_click_mape = 100 * np.sqrt(2 / np.pi) * float(np.mean(s_click / Zt))
    sl = model.meta["noise_model"]["sigma_label"]
    pred_total_mape = 100 * np.sqrt(2 / np.pi) * float(np.mean(np.sqrt((s_click / Zt) ** 2 + sl ** 2)))

    # bootstrap of the rig parameters (Γ-only, fixed λ: isolates estimation noise)
    cfg_fixed = TrainConfig(auto_smooth=False, smooth=model.meta["smooth_selected"], ensemble=False)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):
        m, _ = train(pts[rng.integers(0, len(pts), len(pts))], cfg_fixed, log=None)
        boot.append([m.baseline[0], m.baseline[1], *np.degrees(m.rotation)])
    boot = np.array(boot)
    m0, _ = train(pts, cfg_fixed, log=None)
    a = np.r_[m0.meta["radial_prior"]["left"], m0.meta["radial_prior"]["right"]]
    th = {"gL": m0.gamma_L, "gR": m0.gamma_R, "d": m0.baseline[:2], "w": m0.rotation, "a": a}
    from stereo_gamma.training import noise_scales

    nm = m0.meta["noise_model"]
    w = np.c_[pts, noise_scales(Zt, nm["kappa"], float(np.sqrt(np.mean(Zt ** 2))))]
    diag = diagnostics(th, w, cfg_fixed, model.sensor)
    names = ["dx", "dy", "pitch", "yaw", "roll"]
    vals = [m0.baseline[0], m0.baseline[1], *np.degrees(m0.rotation)]
    params = {}
    for k, name in enumerate(names):
        lap = diag["stderr"][name] * (1 if k < 2 else np.degrees(1))
        params[name] = {"value": float(vals[k]), "se_laplace": float(lap), "se_bootstrap": float(boot[:, k].std(ddof=1)),
                        "unit": "m" if k < 2 else "deg"}
    return model, {
        "noise_model": model.meta["noise_model"], "smooth_selected": model.meta["smooth_selected"],
        "smooth_cv_mape": model.meta["smooth_cv_mape"], "df_eff": diag["df_eff"],
        "n_params": diag["n_params_free"] + 2, "n_params_free": diag["n_params_free"],
        "n_residuals": diag["n_residuals"], "params": params,
        "error_budget": {"predicted_mape_click_only_pct": pred_click_mape,
                         "predicted_mape_click_and_label_pct": pred_total_mape},
        "radial_prior": model.meta["radial_prior"],
    }


# ─── 2. SYNTHETIC BENCHMARK ───────────────────────────────────────────────────
SYN_MODELS = ["oracle", "offset", "legacy", "pinrot", "brown", "gamma", "v21"]


def synthetic_benchmark(seeds, Ns, sigmas):
    rigs = {"brown": SyntheticRig(), "freeform": SyntheticRig.freeform()}
    res = {r: {str(s): {m: [] for m in SYN_MODELS} for s in sigmas} for r in rigs}
    for rname, rig in rigs.items():
        for s in sigmas:
            for N in Ns:
                vals = {m: [] for m in SYN_MODELS}
                t = time.time()
                for seed in range(seeds):
                    rng = np.random.default_rng(1000 * seed + N)
                    tr, _ = rig.sample(N, rng, click_sigma=s, label_sigma=0.01)
                    te, _ = rig.sample(500, rng, click_sigma=s)
                    q, Zt = te[:, :4].T, te[:, 4]
                    a, b = fit_disparity_offset(tr)
                    du = te[:, 0] - te[:, 2]
                    model, _ = train(tr, TrainConfig(), log=None)
                    pin, _ = train(tr, TrainConfig(learn_gamma=False, ensemble=False), log=None)
                    z = {"oracle": rig.triangulate(*q), "offset": np.where(du > b, a / (du - b), np.nan),
                         "legacy": train_legacy(tr).triangulate(*q)[0], "pinrot": pin.triangulate(*q)[0],
                         "brown": (model.companion_model() if model.companion is not None
                                   else BrownConradyStereo().fit(tr)).triangulate(*q),
                         "gamma": model.triangulate_gamma(*q)[0],
                         "v21": model.triangulate(*q)[0]}
                    for m in SYN_MODELS:
                        vals[m].append(metrics(z[m], Zt)["mape_pct"])
                for m in SYN_MODELS:
                    res[rname][str(s)][m].append([float(np.mean(vals[m])), float(np.std(vals[m]))])
                print(f"  {rname:8s} σ={s} N={N:3d}  " + "  ".join(f"{m}={np.mean(vals[m]):.3f}" for m in SYN_MODELS)
                      + f"  ({time.time() - t:.0f}s)", flush=True)
    return {"N": Ns, "sigmas": sigmas, "seeds": seeds, "results": res}


# ─── 3. MATCHING ──────────────────────────────────────────────────────────────
def matching_benchmark(n_clicks=200):
    from stereo_gamma.matching import clahe, epipolar_match, lk_refine
    from stereo_gamma.synthetic import render, scene_points

    rig = SyntheticRig()
    left, depth = render(rig, "left")
    right, _ = render(rig, "right")
    model, _ = train(scene_points(rig, 40, np.random.default_rng(0)), log=None)
    Lg, Rg = clahe(left), clahe(right)
    rng = np.random.default_rng(3)
    rows, t_total = [], 0.0
    while len(rows) < n_clicks:
        u, v = int(rng.uniform(60, 2530)), int(rng.uniform(60, 1880))
        if not np.isfinite(depth[v, u]):
            continue
        t = time.time()
        m = epipolar_match(model, Lg, Rg, u, v)
        if m is not None:
            lk = lk_refine(Lg, Rg, u, v, m.u, m.v)
            if lk is not None and np.hypot(lk[0] - m.u, lk[1] - m.v) < 3:
                m.u, m.v = lk
        t_total += time.time() - t
        z = model.depth(u, v, m.u, m.v) if m is not None else np.nan
        rows.append((depth[v, u], z, bool(m is not None and m.reliable)))
    a = np.array(rows, float)
    e = np.abs(a[:, 1] / a[:, 0] - 1)
    acc = a[:, 2] > 0
    out = {"clicks": n_clicks, "accepted_pct": float(100 * acc.mean()),
           "precision_pct": float(100 * (e[acc] < 0.03).mean()),
           "median_err_pct_accepted": float(100 * np.median(e[acc])),
           "mape_pct_accepted": float(100 * np.mean(e[acc])),
           "ms_per_click": float(1000 * t_total / n_clicks)}
    print(f"  matching: {out}")
    return out, (rig, left, right, model)


# ─── FIGURES ──────────────────────────────────────────────────────────────────
def fig_real(plt, rows, pts):
    Zt = pts[:, 4]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    lim = [0, 5.2]
    a.fill_between(lim, [x * 0.95 for x in lim], [x * 1.05 for x in lim], color=GRID, alpha=0.8, lw=0,
                   label="±5 % band")
    a.plot(lim, lim, color=INK2, lw=0.8)
    for key, col, lab in (("legacy", ORANGE, "v1 original"), ("v21", BLUE, "v2.1")):
        a.scatter(Zt, rows[key]["pred"], s=34, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                  label=f"{lab} — MAPE {rows[key]['mape_pct']:.1f} %")
    a.set(xlim=lim, ylim=[0, 6.2], xlabel="true distance (m)", ylabel="predicted distance (m)",
          title="Nested leave-one-out predictions (27 real points)")
    a.legend(loc="upper left")

    keys = [k for k, _ in LADDER]
    names = [rows[k]["name"].replace("**", "") for k in keys]
    vals = np.array([rows[k]["mape_pct"] for k in keys])
    lo = vals - np.array([rows[k]["mape_ci95"][0] for k in keys])
    hi = np.array([rows[k]["mape_ci95"][1] for k in keys]) - vals
    y = np.arange(len(keys))[::-1]
    b.barh(y, vals, height=0.62, color=[BLUE if k == "v21" else GRAY for k in keys], zorder=2)
    b.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for yi, v in zip(y, vals):
        b.text(0.12, yi, f"{v:.2f} %", va="center", ha="left", color=SURFACE, fontsize=8.5, weight="bold", zorder=4)
    b.set_yticks(y, names, fontsize=8.5)
    b.grid(axis="y", visible=False)
    b.set(xlabel="LOOCV mean absolute % error  (bars: 95 % bootstrap CI)", title="Ablation ladder")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_loocv.png"))
    plt.close(fig)


def fig_error_vs_distance(plt, rows, extra, pts):
    Zt = pts[:, 4]
    e = 100 * rel(rows["v21"]["pred"], Zt)
    s = 100 * np.asarray(extra["sigma"]) / Zt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.errorbar(Zt, e, yerr=2 * s, fmt="none", ecolor=LGRAY, elinewidth=5, alpha=0.9, zorder=2,
                label="predicted ±2σ (out-of-sample)")
    inside = np.abs(e) <= 2 * s
    ax.scatter(Zt[inside], e[inside], s=30, color=BLUE, edgecolor=SURFACE, lw=1.2, zorder=3, label="error inside 2σ")
    ax.scatter(Zt[~inside], e[~inside], s=30, color=ORANGE, edgecolor=SURFACE, lw=1.2, zorder=3,
               label="error outside 2σ")
    ax.set(xlabel="true distance (m)", ylabel="relative error (%)",
           title="v2.1 out-of-sample error vs distance, with the model's own uncertainty")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "error_vs_distance.png"))
    plt.close(fig)


def fig_synthetic(plt, syn):
    rigs = [("brown", "Brown–Conrady lens"), ("freeform", "free-form lens (+ decentering, waviness)")]
    fig, axes = plt.subplots(2, len(syn["sigmas"]), figsize=(11, 7), sharey=True, sharex=True, squeeze=False)
    series = [("oracle", "oracle (true cameras)", INK2, 1.2), ("offset", "pinhole + offset", LGRAY, 1.5),
              ("legacy", "v1 original", LGRAY, 1.5), ("pinrot", "pinhole + rotation", GRAY, 1.5),
              ("brown", "Brown–Conrady", ORANGE, 2), ("gamma", "Γ grid (v2.1)", AQUA, 2),
              ("v21", "Γ + Brown ensemble (v2.1)", BLUE, 2.2)]
    N = syn["N"]
    for i, (rk, rtitle) in enumerate(rigs):
        for j, s in enumerate(syn["sigmas"]):
            ax = axes[i, j]
            for key, lab, col, lw in series:
                r = np.array(syn["results"][rk][str(s)][key])
                ax.plot(N, r[:, 0], color=col, lw=lw, marker="o" if lw >= 2 else None, ms=4, mec=SURFACE,
                        mew=1, label=lab, zorder=3 if lw >= 2 else 2)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(N, [str(n) for n in N])
            ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
            ax.set_title(f"{rtitle}, σ = {s} px", fontsize=10)
            if i == 1:
                ax.set_xlabel("calibration points N")
            if j == 0:
                ax.set_ylabel("held-out MAPE (%) — log")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "synthetic_benchmark.png"))
    plt.close(fig)


def fig_gamma(plt, model):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a8f", "#b92f2f"])
    g0 = model.sensor.gamma0
    grids = [(model.gamma_L[..., 0] / g0[0] - 1, "left γx"), (model.gamma_L[..., 1] / g0[1] - 1, "left γy"),
             (model.gamma_R[..., 0] / g0[0] - 1, "right γx"), (model.gamma_R[..., 1] / g0[1] - 1, "right γy")]
    vmax = max(np.abs(g).max() for g, _ in grids) * 100
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, (g, name) in zip(axes, grids):
        im = ax.imshow(100 * g, cmap=cmap, vmin=-vmax, vmax=vmax)
        for (i, j), val in np.ndenumerate(100 * g):
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7.5, color=INK)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("deviation from pinhole Γ₀ (%)")
    cb.outline.set_visible(False)
    fig.suptitle("Learned Γ grids on the real rig (5×5 nodes over the image; left centre node = scale gauge)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.savefig(os.path.join(FIG, "gamma_grids.png"), bbox_inches="tight")
    plt.close(fig)


def fig_convergence(plt, pts):
    """LM and Adam on the identical objective: both reach the same minimum."""
    fixed = dict(auto_smooth=False, ensemble=False, noise_model="relative")
    _, h_lm = train(pts, TrainConfig(solver="lm", **fixed), log=None)
    _, h_ad = train(pts, TrainConfig(solver="adam", log_every=5, **fixed), log=None)
    lm, ad = np.array(h_lm), np.array(h_ad)
    L_star = min(lm[:, 1].min(), ad[:, 1].min())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for h, col, lab in ((ad, GRAY, "Adam (cosine LR)"), (lm, BLUE, "Levenberg–Marquardt (IRLS)")):
        gap = np.maximum(h[:, 1] - L_star, 1e-16) / L_star
        ax.plot(np.maximum(h[:, 0], 1), gap, color=col, marker="o" if col == BLUE else None, ms=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="iteration / epoch", ylabel="relative sub-optimality (L − L*) / L*",
           title="Two independent optimisers converge to the same minimum")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"))
    plt.close(fig)


def app_screenshot(scene):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    from stereo_gamma.app import App

    rig, left, right, model = scene
    tmp = os.path.join(RES, "_tmp_demo_cal.json")
    model.save(tmp)
    pygame.init()
    surf = [pygame.surfarray.make_surface(np.repeat(im[..., None], 3, 2).transpose(1, 0, 2)) for im in (left, right)]
    app = App(None, None, tmp, os.path.join(RES, "_tmp_pts.json"), surf[0], surf[1])
    for u, v in [(700, 900), (1850, 500), (1300, 1700), (2300, 1200), (1000, 400)]:
        app.click_left(u, v)
    app.render()
    pygame.image.save(app.screen, os.path.join(FIG, "app_demo.png"))
    pygame.quit()
    for p in (tmp, os.path.join(RES, "_tmp_pts.json")):
        if os.path.exists(p):
            os.remove(p)


# ─── REPORT ───────────────────────────────────────────────────────────────────
NAMES = {"legacy": "v1", "offset": "offset", "brown": "Brown–Conrady", "gamma": "Γ alone", "v21": "v2.1",
         "g_v20": "Γ v2.0 loss", "g_fgls": "Γ + FGLS"}


def write_markdown(rows, tests, cov, extra, full, syn, match):
    L = ["# Results", "", f"_Generated by `scripts/reproduce_results.py` (stereo_gamma {__version__})._", "",
         "## Real dataset — nested leave-one-out cross-validation (27 points)", "",
         "| Model | MAE (m) | RMSE (m) | MAPE | 95 % CI | median | δ<1.05 | δ<1.10 | worst |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, _ in LADDER:
        m = rows[k]
        L.append(f"| {m['name']} | {m['mae_m']:.3f} | {m['rmse_m']:.3f} | {m['mape_pct']:.2f} % | "
                 f"{m['mape_ci95'][0]:.1f}–{m['mape_ci95'][1]:.1f} | {m['median_ape_pct']:.2f} % | "
                 f"{m['delta_1.05_pct']:.0f} % | {m['delta_1.10_pct']:.0f} % | {m['max_ape_pct']:.1f} % |")
    L += ["", "### Paired significance tests (sign-flip permutation, 20 000 draws, Holm-corrected)", "",
          "| Comparison | Δ MAPE (pp) | p | p (Holm) |", "|---|---|---|---|"]
    for t in tests:
        L.append(f"| {NAMES[t['a']]} vs {NAMES[t['b']]} | {t['mean_diff_pp']:+.2f} | {t['p']:.4f} | {t['p_holm']:.4f} |")
    L += ["", "### Uncertainty calibration (v2.1, out-of-sample)", "",
          f"- |error| ≤ 1σ: **{cov['within_1sigma_pct']:.0f} %** (Gaussian ideal 68 %); "
          f"≤ 2σ: **{cov['within_2sigma_pct']:.0f} %** (ideal 95 %); RMS z-score {cov['rms_z']:.2f} (ideal 1)",
          f"- correlation of Γ and Brown–Conrady out-of-sample errors: {extra['error_correlation_gamma_brown']:.2f}", "",
          "### Fit on all 27 points", "",
          f"- selected λ_s = {full['smooth_selected']}; estimated noise: label σ = "
          f"{100 * full['noise_model']['sigma_label']:.1f} % of depth, click σ = {full['noise_model']['sigma_px']:.1f} px, "
          f"κ = {full['noise_model']['kappa']:.2f} m",
          f"- parameters: {full['n_params']} ({full['n_params_free']} free); **effective degrees of freedom "
          f"{full['df_eff']:.1f}**; residuals {full['n_residuals']}",
          f"- error budget: click noise alone predicts {full['error_budget']['predicted_mape_click_only_pct']:.2f} % MAPE, "
          f"click + label noise {full['error_budget']['predicted_mape_click_and_label_pct']:.2f} %; "
          f"observed (nested LOOCV) {rows['v21']['mape_pct']:.2f} %", "",
          "| Rig parameter | value | s.e. (sandwich) | s.e. (bootstrap, 200) |", "|---|---|---|---|"]
    for k, p in full["params"].items():
        f = "{:.4f}" if p["unit"] == "m" else "{:.3f}"
        L.append(f"| {k} ({p['unit']}) | {f.format(p['value'])} | {f.format(p['se_laplace'])} | "
                 f"{f.format(p['se_bootstrap'])} |")
    names = {"oracle": "Oracle (true cameras) — noise floor", "offset": "Pinhole + offset", "legacy": "v1 original",
             "pinrot": "Pinhole + rotation", "brown": "Brown–Conrady (18 params)", "gamma": "Γ grid alone",
             "v21": "**Γ + Brown ensemble (v2.1)**"}
    L += ["", f"## Synthetic rigs — held-out MAPE (%), mean ± sd over {syn['seeds']} seeds", ""]
    for rk, rt in (("brown", "Brown–Conrady lens"), ("freeform", "Free-form lens")):
        for s in syn["sigmas"]:
            L += [f"**{rt}, click noise σ = {s} px**", "", "| Model | " + " | ".join(f"N={n}" for n in syn["N"]) + " |",
                  "|---|" + "---|" * len(syn["N"])]
            for k, nm in names.items():
                L.append(f"| {nm} | " + " | ".join(f"{m:.3f} ± {sd:.3f}" for m, sd in syn["results"][rk][str(s)][k]) + " |")
            L.append("")
    if match:
        L += ["## Automatic matching on the ray-traced scene", "",
              f"- clicks: {match['clicks']}, accepted by ZNCC + left-right check: {match['accepted_pct']:.0f} %",
              f"- precision of accepted matches (depth error < 3 %): **{match['precision_pct']:.1f} %**",
              f"- median / mean depth error of accepted matches: {match['median_err_pct_accepted']:.2f} % / "
              f"{match['mape_pct_accepted']:.2f} %",
              f"- time per click: {match['ms_per_click']:.0f} ms", ""]
    with open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    return {"date": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
            "stereo_gamma": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    plt = style()
    pts = load_points(os.path.join(ROOT, "cal_pts.json"))
    t0 = time.time()

    print("[1/5] real data: nested LOOCV ablation ladder")
    rows, tests, cov, extra = real_experiments(pts)
    print("[2/5] real data: full-data fit, error budget, parameter uncertainty")
    model, full = full_model_analysis(pts)
    print("[3/5] synthetic benchmark")
    syn = (synthetic_benchmark(1, [30, 120], [0.5]) if a.fast
           else synthetic_benchmark(3, [15, 30, 60, 120, 480], [0.5, 2.0]))
    print("[4/5] matching benchmark")
    match, scene = (None, None) if a.fast else matching_benchmark()
    print("[5/5] figures")
    fig_real(plt, rows, pts)
    fig_error_vs_distance(plt, rows, extra, pts)
    fig_synthetic(plt, syn)
    fig_gamma(plt, model)
    fig_convergence(plt, pts)
    if scene is not None:
        app_screenshot(scene)
    out = {"provenance": provenance(), "real_loocv": rows, "significance": tests, "coverage": cov,
           "real_extra": extra, "full_fit": full, "synthetic": syn, "matching": match}
    with open(os.path.join(RES, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    write_markdown(rows, tests, cov, extra, full, syn, match)
    print(f"done in {time.time() - t0:.0f}s → {RES}/ and {FIG}/")


if __name__ == "__main__":
    main()
