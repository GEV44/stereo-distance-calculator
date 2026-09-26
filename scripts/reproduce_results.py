"""
Reproduce every number and figure in the README.

    python scripts/reproduce_results.py            # full run (~4 min)
    python scripts/reproduce_results.py --fast     # fewer seeds / skips the app screenshot

Outputs
    results/results.json            all metrics (machine-readable)
    results/RESULTS.md              the tables used in the README
    docs/figures/*.png              figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stereo_gamma.config import TrainConfig  # noqa: E402
from stereo_gamma.data import load_points  # noqa: E402
from stereo_gamma.evaluation import (  # noqa: E402
    fit_disparity_offset,
    loocv,
    loocv_disparity_offset,
    metrics,
)
from stereo_gamma.legacy import loocv_legacy, train_legacy  # noqa: E402
from stereo_gamma.synthetic import SyntheticRig  # noqa: E402
from stereo_gamma.training import train  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
RES = os.path.join(ROOT, "results")

# reference palette (validated: see README "Figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID, SURFACE = "#9a9994", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


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


def bootstrap_ci(ape, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ape = ape[np.isfinite(ape)]
    means = rng.choice(ape, (n, len(ape))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ─── 1. REAL DATA: LOOCV ABLATION ─────────────────────────────────────────────
def real_ablation(pts):
    Zt = pts[:, 4]
    base = TrainConfig()
    runs = [
        ("v1 original (reproduced)", "legacy", lambda: loocv_legacy(pts)),
        ("Pinhole + disparity offset (2 params)", "offset", lambda: loocv_disparity_offset(pts)),
        ("Pinhole + rotation (Γ frozen)", "pinrot", lambda: loocv(pts, replace(base, learn_gamma=False))),
        ("Γ grid, no rotation", "norot", lambda: loocv(pts, replace(base, learn_rotation=False))),
        ("Γ grid + rotation (v2, default)", "v2", lambda: loocv(pts, base)),
        ("v2 + quadratic post-correction", "v2pc", lambda: loocv(pts, replace(base, postcorrection=True))),
    ]
    out = {}
    for name, key, fn in runs:
        t = time.time()
        z = fn()
        m = metrics(z, Zt)
        ape = np.abs(z / Zt - 1) * 100
        m["mape_ci95"] = bootstrap_ci(ape)
        m["name"], m["pred"] = name, z.tolist()
        out[key] = m
        print(f"  {name:42s} MAPE {m['mape_pct']:5.2f}%  [{m['mape_ci95'][0]:.1f}, {m['mape_ci95'][1]:.1f}]"
              f"  ({time.time() - t:.0f}s)", flush=True)
    # paired bootstrap: is v2 better than v1?
    a = np.abs(np.asarray(out["legacy"]["pred"]) / Zt - 1)
    b = np.abs(np.asarray(out["v2"]["pred"]) / Zt - 1)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(Zt), (4000, len(Zt)))
    diff = (a[idx] - b[idx]).mean(1) * 100
    out["paired_v1_minus_v2"] = {"mean_pct": float(diff.mean()),
                                 "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                 "p_v2_not_better": float((diff <= 0).mean())}
    return out


# ─── 2. SYNTHETIC BENCHMARK ───────────────────────────────────────────────────
def synthetic_benchmark(seeds):
    rig = SyntheticRig()
    Ns, sigmas = [15, 30, 60, 120], [0.5, 2.0]
    models = ["offset", "legacy", "pinrot", "v2"]
    res = {f"{s}": {m: [] for m in models} for s in sigmas}
    for s in sigmas:
        for N in Ns:
            vals = {m: [] for m in models}
            for seed in range(seeds):
                rng = np.random.default_rng(100 + seed)
                tr, _ = rig.sample(N, rng, click_sigma=s, label_sigma=0.01)
                te, _ = rig.sample(500, rng, click_sigma=s)
                a, b = fit_disparity_offset(tr)
                du = te[:, 0] - te[:, 2]
                vals["offset"].append(metrics(np.where(du > b, a / (du - b), np.nan), te[:, 4])["mape_pct"])
                vals["legacy"].append(metrics(train_legacy(tr).triangulate(*te[:, :4].T)[0], te[:, 4])["mape_pct"])
                for key, cfg in (("pinrot", TrainConfig(learn_gamma=False)), ("v2", TrainConfig())):
                    m, _ = train(tr, cfg, log=None)
                    vals[key].append(metrics(m.triangulate(*te[:, :4].T)[0], te[:, 4])["mape_pct"])
            for m in models:
                res[f"{s}"][m].append([float(np.mean(vals[m])), float(np.std(vals[m]))])
            print(f"  σ={s} N={N:3d}  " + "  ".join(f"{m}={np.mean(vals[m]):.2f}%" for m in models), flush=True)
    return {"N": Ns, "sigmas": sigmas, "results": res}


# ─── 3. MATCHING BENCHMARK ON THE RAY-TRACED SCENE ────────────────────────────
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
    rel = np.abs(a[:, 1] / a[:, 0] - 1)
    acc = a[:, 2] > 0
    out = {
        "clicks": n_clicks,
        "accepted_pct": float(100 * acc.mean()),
        "precision_pct": float(100 * (rel[acc] < 0.03).mean()),
        "median_err_pct_accepted": float(100 * np.median(rel[acc])),
        "mape_pct_accepted": float(100 * np.mean(rel[acc])),
        "wrong_among_rejected_pct": float(100 * (~(rel[~acc] < 0.03)).mean()) if (~acc).any() else 0.0,
        "ms_per_click": float(1000 * t_total / n_clicks),
    }
    print(f"  matching: {out}")
    return out, (rig, left, right, model)


# ─── FIGURES ──────────────────────────────────────────────────────────────────
def fig_real(plt, real, pts):
    Zt = pts[:, 4]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    lim = [0, 5.2]
    a.fill_between(lim, [x * 0.95 for x in lim], [x * 1.05 for x in lim], color=GRID, alpha=0.7, lw=0,
                   label="±5 % band")
    a.plot(lim, lim, color=INK2, lw=0.8)
    for key, col, lab in (("legacy", ORANGE, "v1 original"), ("v2", BLUE, "v2 (this work)")):
        z = np.asarray(real[key]["pred"])
        a.scatter(Zt, z, s=34, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                  label=f"{lab} — MAPE {real[key]['mape_pct']:.1f} %")
    a.set(xlim=lim, ylim=[0, 6.2], xlabel="true distance (m)", ylabel="predicted distance (m)",
          title="Leave-one-out predictions, 27 real points")
    a.legend(loc="upper left")

    keys = ["legacy", "offset", "pinrot", "norot", "v2", "v2pc"]
    names = [real[k]["name"] for k in keys]
    vals = [real[k]["mape_pct"] for k in keys]
    lo = [v - real[k]["mape_ci95"][0] for v, k in zip(vals, keys)]
    hi = [real[k]["mape_ci95"][1] - v for v, k in zip(vals, keys)]
    y = np.arange(len(keys))[::-1]
    cols = [BLUE if k == "v2" else GRAY for k in keys]
    b.barh(y, vals, height=0.62, color=cols, zorder=2)
    b.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for yi, v in zip(y, vals):
        b.text(0.15, yi, f"{v:.1f} %", va="center", ha="left", color=SURFACE, fontsize=9, weight="bold", zorder=4)
    b.set_yticks(y, names)
    b.grid(axis="y", visible=False)
    b.set(xlabel="LOOCV mean absolute % error  (bars: 95 % bootstrap CI)", title="Ablation on the real dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_loocv.png"))
    plt.close(fig)


def fig_synthetic(plt, syn):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    series = [("offset", "Pinhole + offset", AQUA), ("legacy", "v1 original", ORANGE),
              ("pinrot", "Pinhole + rotation", YELLOW), ("v2", "Γ grid + rotation (v2)", BLUE)]
    N = syn["N"]
    for ax, s in zip(axes, syn["sigmas"]):
        for key, lab, col in series:
            r = np.array(syn["results"][f"{s}"][key])
            ax.plot(N, r[:, 0], color=col, marker="o", ms=5, mec=SURFACE, mew=1.2, label=lab)
            ax.text(N[-1] * 1.08, r[-1, 0], lab, color=INK2, va="center", fontsize=8.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(N, [str(n) for n in N])
        ax.set_yticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
        ax.set_xlim(12, 400)
        ax.set(xlabel="calibration points N", title=f"click noise σ = {s} px")
    axes[0].set_ylabel("held-out MAPE (%)  — log scale")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Synthetic rig with Brown–Conrady distortion (500 test points, mean of seeds)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
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
    fig.suptitle("Learned Γ grids (5×5 nodes over the image); left centre node frozen = gauge",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.savefig(os.path.join(FIG, "gamma_grids.png"), bbox_inches="tight")
    plt.close(fig)


def fig_training(plt, hist):
    h = np.array(hist)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    a.plot(h[:, 0], 1e3 * h[:, 1], color=BLUE)
    a.set(xlabel="epoch", ylabel="objective L  (×10⁻³)", title="Training objective (Adam, cosine LR)")
    b.plot(h[:, 0], 100 * h[:, 2], color=BLUE)
    b.set(xlabel="epoch", ylabel="in-sample MAPE (%)", title="In-sample error")
    b.text(h[0, 0] + 20, 100 * h[0, 2], " epoch 0 = closed-form linear warm start", color=INK2,
           va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curve.png"))
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


def write_markdown(real, syn, match):
    L = ["## Real dataset — leave-one-out cross-validation (27 points)", "",
         "| Model | MAE (m) | MAPE | 95 % CI | median APE | δ<1.05 | worst |",
         "|---|---|---|---|---|---|---|"]
    for k in ["legacy", "offset", "pinrot", "norot", "v2", "v2pc"]:
        m = real[k]
        L.append(f"| {m['name']} | {m['mae_m']:.3f} | **{m['mape_pct']:.2f} %** | "
                 f"{m['mape_ci95'][0]:.1f}–{m['mape_ci95'][1]:.1f} % | {m['median_ape_pct']:.2f} % | "
                 f"{m['delta_1.05_pct']:.0f} % | {m['max_ape_pct']:.1f} % |")
    p = real["paired_v1_minus_v2"]
    L += ["", f"Paired bootstrap, v1 − v2 MAPE: **{p['mean_pct']:.2f} pp** "
              f"(95 % CI {p['ci95'][0]:.2f} – {p['ci95'][1]:.2f}), P(v2 not better) = {p['p_v2_not_better']:.3f}", ""]
    L += ["## Synthetic rig — held-out MAPE (%)", ""]
    names = {"offset": "Pinhole + offset", "legacy": "v1 original", "pinrot": "Pinhole + rotation",
             "v2": "**Γ grid + rotation (v2)**"}
    for s in syn["sigmas"]:
        L += [f"Click noise σ = {s} px", "", "| Model | " + " | ".join(f"N={n}" for n in syn["N"]) + " |",
              "|---|" + "---|" * len(syn["N"])]
        for k, nm in names.items():
            L.append(f"| {nm} | " + " | ".join(f"{m:.2f} ± {sd:.2f}" for m, sd in syn["results"][f"{s}"][k]) + " |")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    plt = style()
    pts = load_points(os.path.join(ROOT, "cal_pts.json"))

    print("[1/4] real-data ablation (LOOCV)")
    real = real_ablation(pts)
    print("[2/4] synthetic benchmark")
    syn = synthetic_benchmark(2 if a.fast else 5)
    print("[3/4] matching benchmark")
    match, scene = (None, None) if a.fast else matching_benchmark()
    print("[4/4] figures")
    model, hist = train(pts, replace(TrainConfig(), log_every=10), log=None)
    fig_real(plt, real, pts)
    fig_synthetic(plt, syn)
    fig_gamma(plt, model)
    fig_training(plt, hist)
    if scene is not None:
        app_screenshot(scene)
    with open(os.path.join(RES, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"real_loocv": real, "synthetic": syn, "matching": match}, f, indent=1)
    write_markdown(real, syn, match)
    print(f"done → {RES}/ and {FIG}/")


if __name__ == "__main__":
    main()
