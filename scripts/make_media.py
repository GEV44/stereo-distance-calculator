"""
Regenerate the README / portfolio media in docs/media/.

    python scripts/make_media.py            # everything
    python scripts/make_media.py plots      # accuracy, lens grids, depth curve
    python scripts/make_media.py math       # PDF page previews (needs pdftoppm)
    python scripts/make_media.py demo       # synthetic scene + UI recording (needs ffmpeg)
    python scripts/make_media.py diagram    # pipeline.svg
    python scripts/make_media.py loo        # leave-one-out check (slow, prints only)

Extra dependencies: matplotlib and Pillow (pip install matplotlib).

What is real and what is synthetic
----------------------------------
* The plots use only the shipped data: cal_pts.json (27 measured points) and
  stereo_calibration.json (learned Γ grids + baseline), evaluated with the
  functions in run_math.py.
* The demo video drives the real run_math.run() pygame UI headlessly, but the
  stereo pair it shows is a SYNTHETIC SCENE: textured boxes, a sphere, a floor
  and a wall at known depths, ray-traced through the shipped Γ model so the
  geometry is self-consistent. The repo ships no real camera photos.
"""

import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_math as rm  # noqa: E402

OUT = ROOT / "docs" / "media"
SCRATCH = Path(os.environ.get("MEDIA_SCRATCH", Path(tempfile.gettempdir()) / "stereo-media"))

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
BLUE, ORANGE, GREEN = "#2f6feb", "#e8590c", "#1a7f37"


# ─── shared helpers ───────────────────────────────────────────────────────────
def load_real():
    GL, GR, d = rm.load_calibration(str(ROOT / rm.CALIB_PATH))
    pts = np.array(rm.load_cal_pts(str(ROOT / rm.CAL_PTS_PATH)), dtype=float)
    pred = np.array([rm.triangulate(*p[:4], GL, GR, d)[0] for p in pts])
    return GL, GR, d, pts, pred


def error_stats(pts, pred):
    err = pred - pts[:, 4]
    a = np.abs(err)
    return {"n": len(pts), "mae": a.mean(), "max": a.max(), "median": np.median(a),
            "rmse": math.sqrt((err ** 2).mean()), "mape": (a / pts[:, 4]).mean() * 100}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 12,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titleweight": "semibold", "axes.titlesize": 13,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def save_png(fig, path, palette=True):
    """Save a matplotlib figure as an optimised PNG (palette unless it has gradients)."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if palette:
        img = img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(path, optimize=True)
    print(f"  wrote {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")


def gamma_field(u, v, G):
    """Vectorised copy of run_math.bilinear (same clamping) for arrays u, v >= 0."""
    fx, fy = u / rm.CELL_W, v / rm.CELL_H
    ix = np.clip(np.floor(fx).astype(int), 0, rm.GRID_COLS - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, rm.GRID_ROWS - 2)
    tx, ty = fx - ix, fy - iy
    return ((1 - tx) * (1 - ty) * G[iy, ix] + tx * (1 - ty) * G[iy, ix + 1]
            + (1 - tx) * ty * G[iy + 1, ix] + tx * ty * G[iy + 1, ix + 1])


# ─── 1. accuracy.png ──────────────────────────────────────────────────────────
def fig_accuracy():
    plt = setup_mpl()
    _, _, _, pts, pred = load_real()
    s = error_stats(pts, pred)
    true = pts[:, 4]
    err = pred - true

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150,
                                 gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
    lim = (0, 5.2)
    xs = np.linspace(*lim, 50)
    a1.fill_between(xs, xs * 0.95, xs * 1.05, color=BLUE, alpha=0.08, lw=0, label="±5 %")
    a1.plot(lim, lim, color=MUTED, lw=1.2, ls="--", label="perfect (y = x)")
    a1.scatter(true, pred, s=46, color=BLUE, edgecolor="white", lw=0.8, zorder=3,
               label="real point")
    a1.set(xlim=lim, ylim=lim, xlabel="measured distance (m)",
           ylabel="predicted distance (m)", title="Predicted vs measured")
    a1.set_aspect("equal")
    a1.legend(loc="upper left", frameon=False, fontsize=11)
    w = int(np.argmax(np.abs(err)))
    a1.annotate(f"worst: {true[w]:.2f} → {pred[w]:.2f} m", (true[w], pred[w]),
                xytext=(2.45, 1.0), fontsize=11, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))

    order = np.argsort(true)
    a2.axhspan(-s["mae"] * 100, s["mae"] * 100, color=BLUE, alpha=0.08, lw=0)
    a2.axhline(0, color=MUTED, lw=1.2)
    a2.vlines(true[order], 0, err[order] * 100, color=BLUE, lw=1.6, alpha=0.6)
    big = np.abs(err) > 0.2
    a2.scatter(true[~big], err[~big] * 100, s=40, color=BLUE, edgecolor="white", lw=0.8,
               zorder=3)
    a2.scatter(true[big], err[big] * 100, s=46, color=ORANGE, edgecolor="white", lw=0.8,
               zorder=3, label="|error| > 0.2 m")
    a2.text(0.2, s["mae"] * 100 + 2, f"± MAE ({s['mae'] * 100:.1f} cm)", color=BLUE,
            fontsize=11)
    a2.set(xlim=lim, xlabel="measured distance (m)", ylabel="predicted − measured (cm)",
           title="Residual per point")
    a2.legend(loc="lower left", frameon=False, fontsize=11)

    fig.suptitle(f"Shipped calibration on {s['n']} real measured points:  "
                 f"MAE {s['mae']:.3f} m", fontsize=16, fontweight="bold", x=0.06,
                 ha="left", y=0.995)
    fig.text(0.06, 0.905,
             f"max |error| {s['max']:.2f} m  ·  median {s['median'] * 100:.1f} cm  ·  "
             f"RMSE {s['rmse']:.2f} m  ·  mean relative error {s['mape']:.1f} %",
             fontsize=12, color=INK)
    fig.text(0.06, 0.87,
             "run_math.triangulate on cal_pts.json, the point set the calibration is fitted "
             "to: this is fit error, not a held-out test.", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.11, top=0.79)
    save_png(fig, OUT / "accuracy.png")
    plt.close(fig)
    return s


# ─── 2. lens-grids.png ────────────────────────────────────────────────────────
def fig_lens_grids():
    plt = setup_mpl()
    from matplotlib.colors import TwoSlopeNorm
    GL, GR, d, pts, _ = load_real()
    g0 = rm.G0_INIT
    lo = min(GL.min(), GR.min()) - 0.01
    norm = TwoSlopeNorm(vcenter=g0, vmin=min(lo, g0 - 0.02),
                        vmax=max(GL.max(), GR.max()) + 0.01)
    uu, vv = np.meshgrid(np.linspace(0, rm.IMG_W, 260), np.linspace(0, rm.IMG_H, 196))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=150)
    for ax, G, name, cols in ((axes[0], GL, "Left camera  Γ_L", (0, 1)),
                              (axes[1], GR, "Right camera  Γ_R", (2, 3))):
        im = ax.imshow(gamma_field(uu, vv, G), cmap="RdBu_r", norm=norm,
                       extent=(0, rm.IMG_W, rm.IMG_H, 0), interpolation="bilinear")
        # how much calibration data touches each node (sum of bilinear weights)
        support = np.zeros_like(G)
        for p in pts:
            _, w, idx = rm.bilinear(p[cols[0]], p[cols[1]], G)
            for k in range(4):
                support[idx[k]] += w[k]
        ax.scatter(pts[:, cols[0]], pts[:, cols[1]], s=16, color="white",
                   edgecolor=INK, lw=0.7, zorder=3)
        for iy in range(rm.GRID_ROWS):
            for ix in range(rm.GRID_COLS):
                x, y = ix * rm.CELL_W, iy * rm.CELL_H
                val = G[iy, ix]
                frozen = (iy, ix) == (2, 2)
                untouched = support[iy, ix] < 1e-9
                ax.plot(x, y, "o", ms=5, color=INK, zorder=4, clip_on=False)
                label = f"{val:.3f}"
                if frozen:
                    label += "\nfrozen"
                elif untouched:
                    label += "\nno data"
                ax.text(x, y + 70, label, ha="center", va="top", fontsize=10.5,
                        fontweight="semibold", color=INK, zorder=5, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=INK if frozen else "none", alpha=0.85, lw=0.8))
        ax.set_xlim(-120, rm.IMG_W + 120)
        ax.set_ylim(rm.IMG_H + 260, -120)
        ax.set_xticks(np.arange(0, rm.IMG_W + 1, rm.CELL_W))
        ax.set_yticks(np.arange(0, rm.IMG_H + 1, rm.CELL_H))
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("u (px)")
    axes[0].set_ylabel("v (px)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f"Γ   (nominal {g0:.2f} = {rm.F_INIT:.0f} px focal length;\n"
                 "red = wider ray / shorter effective focal length)", fontsize=10.5)
    cb.ax.tick_params(labelsize=10)
    fig.suptitle("What the lens model learned: 5×5 Γ grids per camera", fontsize=16,
                 fontweight="bold", x=0.05, ha="left")
    fig.text(0.05, 0.895,
             f"Ray = [u_n·Γ, v_n·Γ, 1], Γ bilinearly interpolated between nodes.  "
             f"Learned baseline dx = {d[0] * 100:.1f} cm, dy = {d[1] * 1000:.1f} mm.",
             fontsize=11, color=INK)
    fig.text(0.05, 0.86,
             "White dots: the 27 real calibration clicks in each image.  "
             "Edge nodes that no click touches keep the initial 0.45.",
             fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.08, top=0.83, wspace=0.12)
    save_png(fig, OUT / "lens-grids.png", palette=False)
    plt.close(fig)


# ─── 3. depth-curve.png ───────────────────────────────────────────────────────
def fig_depth_curve():
    plt = setup_mpl()
    GL, GR, d, pts, _ = load_real()
    Zs = np.geomspace(0.33, 6.5, 160)

    def curve(uL, vL):
        out = []
        for Z in Zs:
            pr = rm.predict_right(uL, vL, Z, GL, GR, d)
            if pr is not None and 0 < pr[0] < rm.IMG_W - 1:
                out.append((uL - pr[0], Z))
        return np.array(out)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    for p in pts:                       # the model curve through each real point's left pixel
        c = curve(p[0], p[1])
        ax.plot(c[:, 0], c[:, 1], color=BLUE, alpha=0.22, lw=1)
    c = curve(rm.CX, rm.CY)
    ax.plot([], [], color=BLUE, alpha=0.4, lw=1.5,
            label="model curve at each point's left pixel")
    ax.plot(c[:, 0], c[:, 1], color=BLUE, lw=2.6, label="model at the image centre")
    disp = pts[:, 0] - pts[:, 2]
    ax.scatter(disp, pts[:, 4], s=52, color=ORANGE, edgecolor="white", lw=0.9, zorder=4,
               label=f"{len(pts)} real points (measured disparity, true distance)")
    ax.set(xlim=(0, 1550), ylim=(0, 6.2), xlabel="disparity  u_L − u_R  (px)",
           ylabel="depth Z (m)")
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    k = rm.F_INIT * d[0]                # Z ≈ k / disparity  →  dZ/d(disp) ≈ Z² / k
    ax.text(640, 3.1, f"Z ≈ f·dx / disparity  (f·dx ≈ {k:.0f} px·m)\n"
            f"1 px of disparity ≈ {0.25 / k * 1000:.1f} mm at 0.5 m,  "
            f"{25 / k * 100:.1f} cm at 5 m", fontsize=11, color=MUTED, linespacing=1.6)
    fig.suptitle("Depth vs disparity across the image (shipped calibration)",
                 fontsize=16, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.885, "Thin curves: run_math.predict_right swept over Z at each real "
             "point's left pixel. Their spread is the learned lens variation.",
             fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.85)
    save_png(fig, OUT / "depth-curve.png")
    plt.close(fig)


def make_plots():
    s = fig_accuracy()
    fig_lens_grids()
    fig_depth_curve()
    print("  real-data stats: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))
    return s


# ─── 4. PDF page previews ─────────────────────────────────────────────────────
def make_math_pages(pages=(1, 2, 3)):
    from PIL import Image, ImageOps
    tmp = Path(tempfile.mkdtemp())
    pdf = ROOT / "docs" / "MATHEMATICAL_FOUNDATION.pdf"
    for p in pages:
        subprocess.run(["pdftoppm", "-r", "110", "-png", "-f", str(p), "-l", str(p),
                        "-singlefile", str(pdf), str(tmp / f"p{p}")], check=True)
        img = Image.open(tmp / f"p{p}.png").convert("RGB")
        ink = ImageOps.invert(img.convert("L")).point(lambda v: 255 if v > 12 else 0)
        x0, y0, x1, y1 = ink.getbbox()
        pad = 36
        img = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                        min(img.width, x1 + pad), min(img.height, y1 + pad)))
        img = img.quantize(32, method=Image.Quantize.MEDIANCUT)   # text + a few colours
        out = OUT / f"math-p{p}.png"
        img.save(out, optimize=True)
        print(f"  wrote {out.relative_to(ROOT)}  {img.size}  {out.stat().st_size / 1024:.0f} KB")
    shutil.rmtree(tmp)


# ─── 5. synthetic scene + headless UI recording ───────────────────────────────
def _texture(w_m, h_m, ppm, base, palette, density, rmin, rmax, seed, tiles=None):
    """Terrazzo-like texture: random flat shapes on a base colour (unique, matchable)."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = int(w_m * ppm), int(h_m * ppm)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = base
    if tiles:
        step = int(tiles[0] * ppm)
        for y in range(0, H, step):
            for x in range(0, W, step):
                if ((x // step) + (y // step)) % 2:
                    img[y:y + step, x:x + step] = tiles[1]
    for _ in range(int(density * w_m * h_m)):
        col = tuple(int(c) for c in palette[rng.integers(len(palette))])
        x, y = int(rng.integers(W)), int(rng.integers(H))
        r = int(rng.uniform(rmin, rmax) * ppm)
        k = rng.integers(3)
        if k == 0:
            cv2.circle(img, (x, y), r, col, -1, cv2.LINE_AA)
        elif k == 1:
            box = cv2.boxPoints(((x, y), (2 * r, r * rng.uniform(0.4, 1.4)),
                                 float(rng.uniform(0, 180))))
            cv2.fillPoly(img, [box.astype(np.int32)], col, cv2.LINE_AA)
        else:
            ang = rng.uniform(0, 2 * np.pi) + np.array([0, 2.1, 4.2])
            tri = np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], 1)
            cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)
    return img, ppm


def _scene():
    """Objects in the LEFT camera frame (metres, x right, y down, z forward)."""
    return [
        dict(kind="plane", axis=2, at=5.0,                                   # back wall
             tex=_texture(6, 5, 260, (206, 201, 191),
                          [(172, 163, 152), (228, 223, 214), (74, 92, 112), (192, 112, 82)],
                          60, 0.02, 0.07, 1)),
        dict(kind="plane", axis=1, at=1.0, zmax=5.0,                         # floor
             tex=_texture(6, 6, 300, (128, 120, 112),
                          [(96, 90, 85), (162, 154, 144), (204, 194, 172)], 50, 0.01, 0.05, 2,
                          tiles=(0.5, (116, 109, 102)))),
        dict(kind="box", lo=(-0.62, 0.12, 1.40), hi=(-0.08, 1.0, 1.90),       # blue box
             tex=_texture(2, 2, 900, (52, 98, 150),
                          [(35, 70, 115), (92, 142, 192), (236, 231, 220), (240, 180, 70)],
                          420, 0.006, 0.026, 3)),
        dict(kind="box", lo=(0.22, 0.05, 2.50), hi=(0.95, 1.0, 3.05),         # yellow box
             tex=_texture(2, 2, 700, (226, 186, 80),
                          [(188, 148, 52), (246, 216, 132), (60, 62, 72), (200, 92, 62)],
                          330, 0.008, 0.035, 4)),
        dict(kind="box", lo=(-0.30, -0.70, 3.60), hi=(0.20, 1.0, 4.10),       # red box
             tex=_texture(2, 2, 500, (160, 72, 64),
                          [(120, 50, 45), (202, 112, 96), (240, 230, 215), (52, 42, 42)],
                          260, 0.01, 0.045, 5)),
        dict(kind="sphere", c=(-1.15, 0.05, 4.10), r=0.45,                   # green sphere
             tex=_texture(3, 1.5, 500, (70, 140, 105),
                          [(45, 100, 75), (122, 186, 150), (236, 236, 222)], 260, 0.01, 0.045,
                          6)),
    ]


def _render(objects, origin, G):
    """Ray-trace one camera of the Γ model: ray(u,v) = [u_n·Γ, v_n·Γ, 1] from origin."""
    import cv2
    W, H = rm.IMG_W, rm.IMG_H
    img = np.zeros((H, W, 3), np.float32)
    depth = np.full((H, W), np.inf, np.float32)
    light = np.array([-0.45, -0.8, -0.55]); light /= np.linalg.norm(light)
    o = np.asarray(origin, float)
    for r0 in range(0, H, 243):
        v, u = np.mgrid[r0:min(H, r0 + 243), 0:W].astype(np.float64)
        g = gamma_field(u, v, G)
        D = np.stack([(u - rm.CX) / rm.CX * g, (v - rm.CY) / rm.CY * g, np.ones_like(u)], -1)
        best = np.full(u.shape, np.inf)
        col = np.zeros(u.shape + (3,), np.float32)
        for ob in objects:
            t = np.full(u.shape, np.inf)
            if ob["kind"] == "plane":
                ax = ob["axis"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (ob["at"] - o[ax]) / D[..., ax]
                t[~(t > 0)] = np.inf
                if "zmax" in ob:
                    t[o[2] + t * D[..., 2] > ob["zmax"]] = np.inf
                n = np.zeros(3); n[ax] = -1.0
                P = o + np.where(np.isinf(t), 0, t)[..., None] * D
                a, b = (P[..., 0] + 3, P[..., 1] + 2.5) if ax == 2 else (P[..., 0] + 3, P[..., 2])
                normal = np.broadcast_to(n, D.shape)
            elif ob["kind"] == "box":
                lo, hi = np.array(ob["lo"]), np.array(ob["hi"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1, t2 = (lo - o) / D, (hi - o) / D
                tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
                tn = np.nan_to_num(tn, nan=-np.inf); tf = np.nan_to_num(tf, nan=np.inf)
                face = np.argmax(tn, -1)
                t0, t3 = tn.max(-1), tf.min(-1)
                t = np.where((t3 >= t0) & (t0 > 0), t0, np.inf)
                P = o + np.where(np.isinf(t), 0, t)[..., None] * D
                normal = np.zeros(D.shape)
                sgn = -np.sign(np.take_along_axis(D, face[..., None], -1)[..., 0])
                np.put_along_axis(normal, face[..., None], sgn[..., None], -1)
                q = P - lo
                a = np.where(face == 0, q[..., 2] + 0.9, q[..., 0])
                b = np.where(face == 1, q[..., 2] + 1.0, q[..., 1])
            else:  # sphere
                c, r = np.array(ob["c"]), ob["r"]
                oc = o - c
                A = (D * D).sum(-1); B = 2 * (D * oc).sum(-1); C = oc @ oc - r * r
                disc = B * B - 4 * A * C
                t = np.where(disc > 0, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
                t[~(t > 0)] = np.inf
                P = o + np.where(np.isinf(t), 0, t)[..., None] * D
                normal = (P - c) / r
                a = (np.arctan2(normal[..., 0], -normal[..., 2]) + np.pi) * r
                b = np.arccos(np.clip(normal[..., 1], -1, 1)) * r
            hit = t < best
            if not hit.any():
                continue
            tex, ppm = ob["tex"]
            mx = np.where(hit, a * ppm, 0).astype(np.float32)
            my = np.where(hit, b * ppm, 0).astype(np.float32)
            samp = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            shade = 0.58 + 0.42 * np.clip((normal * light).sum(-1), 0, 1)
            col[hit] = samp[hit] * shade[hit, None]
            best[hit] = t[hit]
        img[r0:r0 + u.shape[0]] = col
        depth[r0:r0 + u.shape[0]] = best          # dir.z == 1, so t is depth Z
    img = cv2.GaussianBlur(img, (0, 0), 0.6)      # mild optical blur / anti-aliasing
    return np.clip(img, 0, 255).astype(np.uint8), depth


def _right_pixel(uL, vL, Z, GL, GR, d, iters=60):
    """Exact inverse of the right-camera Γ ray (fixed point, run to convergence)."""
    vL_d, _ = rm.build_ray(uL, vL, GL)
    tgt = vL_d - d / Z
    uR, vR = rm.CX + rm.CX * tgt[0] / rm.G0_INIT, rm.CY + rm.CY * tgt[1] / rm.G0_INIT
    for _ in range(iters):
        g, _, _ = rm.bilinear(uR, vR, GR)
        uR, vR = rm.CX + rm.CX * tgt[0] / g, rm.CY + rm.CY * tgt[1] / g
    return uR, vR


def make_scene(workdir):
    import cv2
    GL, GR, d = rm.load_calibration(str(ROOT / rm.CALIB_PATH))
    objects = _scene()
    print("  rendering synthetic left/right pair (2592×1944) …")
    left, depth_L = _render(objects, (0, 0, 0), GL)
    right, depth_R = _render(objects, d, GR)
    cv2.imwrite(str(workdir / "left.png"), left[..., ::-1])
    cv2.imwrite(str(workdir / "right.png"), right[..., ::-1])
    return GL, GR, d, depth_L, depth_R


def _ease(a, b, n):
    s = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    return [(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k) for k in s]


def record_demo():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame

    work = SCRATCH / "demo"
    if work.exists():
        shutil.rmtree(work)
    (work / "frames").mkdir(parents=True)
    GL, GR, d, depth_L, depth_R = make_scene(work)
    # run() reads stereo_calibration.json / cal_pts.json from the cwd: give it the
    # shipped calibration and no calibration points (those belong to real photos).
    shutil.copy(ROOT / rm.CALIB_PATH, work / rm.CALIB_PATH)

    SX, SY = rm.DISP_W / rm.IMG_W, rm.DISP_H / rm.IMG_H
    Wd, Hd = rm.DISP_W, rm.DISP_H

    def to_screen(u, v, p, z=1.0, pan=(0, 0)):           # mirrors run().img_to_screen
        return int(round(u * SX * z + pan[0])) + p * Wd, int(round(v * SY * z + pan[1]))

    def to_img(mx, my, p, z=1.0, pan=(0, 0)):           # mirrors run().screen_to_img
        return (int(round(((mx - p * Wd) - pan[0]) / (z * SX))),
                int(round((my - pan[1]) / (z * SY))))

    def wheel(z, pan, mx, my):                          # mirrors run()'s MOUSEWHEEL zoom
        new = max(1.0, min(8.0, z * 1.15))
        lx = mx - Wd
        px, py = int(lx - (lx - pan[0]) * new / z), int(my - (my - pan[1]) * new / z)
        px = min(0, max(px, Wd - int(Wd * new)))
        py = min(0, max(py, Hd - int(Hd * new)))
        return new, (px, py)

    # Click targets (left-image pixels) on textured surfaces at known depths; the last
    # one is clicked after zooming the right panel with the mouse wheel.
    targets = [("blue box", 900, 1330, 0), ("yellow box", 1950, 1160, 0),
               ("red box", 1180, 820, 0), ("back wall", 2050, 470, 8)]
    shots = []
    for name, u, v, n_wheel in targets:
        sL = to_screen(u, v, 0)
        uL, vL = to_img(*sL, 0)                     # the pixel run() will actually use
        Z_true = float(depth_L[vL, uL])
        uR, vR = _right_pixel(uL, vL, Z_true, GL, GR, d)
        assert abs(depth_R[int(round(vR)), int(round(uR))] - Z_true) < 0.02, name  # visible
        z, pan = 1.0, (0, 0)
        sR0 = to_screen(uR, vR, 1)
        for _ in range(n_wheel):
            z, pan = wheel(z, pan, *sR0)
        sR = to_screen(uR, vR, 1, z, pan)
        uRi, vRi = to_img(*sR, 1, z, pan)
        Z_meas = rm.triangulate(uL, vL, uRi, vRi, GL, GR, d)[0]
        shots.append(dict(name=name, sL=sL, sR0=sR0, sR=sR, wheel=n_wheel, zoom=z,
                          Z_true=Z_true, Z=Z_meas))
        print(f"  {name:10s} L=({uL},{vL}) R=({uRi},{vRi}) zoom {z:.2f}x  true {Z_true:.3f} m"
              f"  measured {Z_meas:.3f} m  (exact R=({uR:.1f},{vR:.1f}))")

    # Timeline: one entry per UI frame = (events, cursor, caption, result)
    FPS = 15
    tl, cur = [], (Wd // 2, Hd // 2)
    MB = pygame.MOUSEBUTTONDOWN
    away = lambda p: (p[0] + 34, p[1] + 40)     # park the cursor off the marker it placed

    def add(events, cursor, caption, res=None, n=1):
        tl.extend([(events, cursor, caption, res)] + [([], cursor, caption, res)] * (n - 1))

    def move(a, b, caption, res=None, n=8):
        for c in _ease(a, b, n):
            add([], c, caption, res)

    add([], cur, "Real run_math.py UI, driven headlessly on a synthetic stereo pair", n=12)
    for i, s in enumerate(shots):
        step = f"{i + 1}/{len(shots)}  {s['name']}"
        move(cur, s["sL"], f"{step}: click a point in the LEFT image")
        cap = f"{step}: Γ-ray → epipolar curve, 1/2/5 m rings, ZNCC hint (+) on the right"
        add([pygame.event.Event(MB, pos=s["sL"], button=1)], s["sL"], cap)
        move(s["sL"], away(s["sL"]), cap, n=4)
        add([], away(s["sL"]), cap, n=7)
        if s["wheel"]:
            cap = f"{step}: mouse wheel zooms the right panel for a precise click"
            move(away(s["sL"]), s["sR0"], cap)
            for _ in range(s["wheel"]):
                add([pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1)], s["sR0"], cap)
            add([], s["sR0"], cap, n=3)
            move(s["sR0"], s["sR"], f"{step}: click the matching point in the RIGHT image", n=5)
        else:
            move(away(s["sL"]), s["sR"], f"{step}: click the matching point in the RIGHT image")
        cap = f"{step}: OLS triangulation of the two Γ-rays"
        add([pygame.event.Event(MB, pos=s["sR"], button=1)], s["sR"], cap, s)
        move(s["sR"], away(s["sR"]), cap, s, n=4)
        add([], away(s["sR"]), cap, s, n=10 if i < len(shots) - 1 else 22)
        cur = away(s["sR"])
        if i == 1:
            poster = len(tl) - 1        # poster: finished, unzoomed second measurement

    state = {"i": 0, "frame": 0, "fonts": None}
    real_get, real_flip, real_clock = pygame.event.get, pygame.display.flip, pygame.time.Clock
    real_pos = pygame.mouse.get_pos

    def fake_pos():
        return tuple(int(c) for c in tl[min(state["i"], len(tl) - 1)][1])

    def fake_get(*a, **k):
        if state["i"] >= len(tl):
            return [pygame.event.Event(pygame.QUIT)]
        return list(tl[state["i"]][0])

    def fake_flip():
        if state["i"] >= len(tl):
            return
        _, cursor, caption, res = tl[state["i"]]
        state["i"] += 1
        screen = pygame.display.get_surface()
        if state["fonts"] is None:
            f = lambda name, size: pygame.font.Font(
                pygame.font.match_font(name) or pygame.font.get_default_font(), size)
            state["fonts"] = (f("inter,dejavusans", 27), f("inter,dejavusans", 21),
                              f("jetbrainsmono,dejavusansmono", 30))
        big, small, mono = state["fonts"]
        CAP = 96
        W, H = screen.get_size()
        canvas = pygame.Surface((W, H + CAP))
        canvas.blit(screen, (0, 0))
        _draw_cursor(pygame, canvas, cursor)
        canvas.fill((13, 17, 23), pygame.Rect(0, H, W, CAP))
        pygame.draw.line(canvas, (48, 54, 61), (0, H), (W, H), 2)
        canvas.blit(big.render(caption, True, (230, 237, 243)), (20, H + 14))
        canvas.blit(small.render(
            "Synthetic scene ray-traced through the shipped Γ calibration · cursor drawn "
            "for clarity", True, (139, 148, 158)), (20, H + 56))
        badge = small.render("SYNTHETIC SCENE", True, (255, 166, 87))
        bx = W - badge.get_width() - 34
        pygame.draw.rect(canvas, (255, 166, 87),
                         pygame.Rect(bx - 12, H + 52, badge.get_width() + 24, 32), 2,
                         border_radius=6)
        canvas.blit(badge, (bx, H + 56))
        if res is not None:
            txt = mono.render(f"Z = {res['Z']:.3f} m   true {res['Z_true']:.3f} m   "
                              f"Δ {100 * (res['Z'] - res['Z_true']):+.1f} cm",
                              True, (126, 231, 135))
            canvas.blit(txt, (W - txt.get_width() - 22, H + 12))
        pygame.image.save(canvas, str(work / "frames" / f"f_{state['frame']:04d}.png"))
        state["frame"] += 1

    class _Clock:
        def tick(self, *a):
            return 0

    cwd = os.getcwd()
    try:
        os.chdir(work)
        pygame.event.get, pygame.display.flip, pygame.time.Clock = fake_get, fake_flip, _Clock
        pygame.mouse.get_pos = fake_pos
        with contextlib.redirect_stdout(io.StringIO()) as log:
            rm.run(str(work / "left.png"), str(work / "right.png"))
    finally:
        pygame.event.get, pygame.display.flip, pygame.time.Clock = real_get, real_flip, real_clock
        pygame.mouse.get_pos = real_pos
        os.chdir(cwd)
    print("  " + "\n  ".join(l for l in log.getvalue().splitlines() if "INFO-ONLY" in l))
    print(f"  captured {state['frame']} frames")
    _encode(work / "frames", FPS, poster=poster)
    return shots


def _draw_cursor(pygame, surf, pos):
    x, y = int(pos[0]), int(pos[1])
    pts = [(0, 0), (0, 22), (6, 17), (10, 26), (14, 24), (10, 16), (17, 16)]
    poly = [(x + px, y + py) for px, py in pts]
    pygame.draw.polygon(surf, (255, 255, 255), poly)
    pygame.draw.polygon(surf, (0, 0, 0), poly, 2)


def _encode(frames, fps, poster):
    from PIL import Image
    pattern = str(frames / "f_%04d.png")
    mp4, gif = OUT / "demo.mp4", OUT / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-vf", "scale=1280:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "21", "-movflags", "+faststart", "-an", str(mp4)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-vf", "scale=960:-1:flags=lanczos,split[a][b];"
                           "[a]palettegen=max_colors=224:stats_mode=diff[p];"
                           "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    str(gif)], check=True)
    img = Image.open(frames / f"f_{poster:04d}.png").convert("RGB")
    img = img.resize((1280, round(img.height * 1280 / img.width)), Image.LANCZOS)
    img.quantize(256, method=Image.Quantize.MEDIANCUT).save(OUT / "demo-poster.png",
                                                            optimize=True)
    for p in (mp4, gif, OUT / "demo-poster.png"):
        print(f"  wrote {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


# ─── 6. pipeline.svg ──────────────────────────────────────────────────────────
def make_diagram():
    W, H = 1180, 470
    bg, card, edge, txt, sub = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
    acc = {"in": "#58a6ff", "model": "#d2a8ff", "match": "#3fb950", "solve": "#ffa657",
           "info": "#8b949e"}

    def box(x, y, w, h, title, lines, kind, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{card}" '
             f'stroke="{acc[kind]}" stroke-width="1.6"{dash}/>',
             f'<rect x="{x}" y="{y + 14}" width="4" height="{h - 28}" rx="2" fill="{acc[kind]}"/>',
             f'<text x="{x + 18}" y="{y + 30}" class="t">{title}</text>']
        for i, line in enumerate(lines):
            s.append(f'<text x="{x + 18}" y="{y + 54 + i * 20}" class="s">{line}</text>')
        return "\n".join(s)

    def arrow(x1, y1, x2, y2, label=None, dashed=False):
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        s = f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{sub}" stroke-width="1.8" ' \
            f'fill="none" marker-end="url(#a)"{dash}/>'
        if label and x1 == x2:          # vertical arrow: label to the right of it
            s += f'<text x="{x1 + 10}" y="{(y1 + y2) / 2 + 4}" class="l">{label}</text>'
        elif label:
            s += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 8}" class="l" ' \
                 f'text-anchor="middle">{label}</text>'
        return s

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="Stereo distance pipeline">',
        "<title>Stereo distance pipeline: Γ-grid ray model, ZNCC hint, OLS triangulation"
        "</title>",
        "<defs><marker id=\"a\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" "
        f"markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0,0 L10,5 L0,10 z\" "
        f"fill=\"{sub}\"/></marker>",
        "<style>text{font-family:Inter,'Segoe UI',Helvetica,Arial,sans-serif}"
        f".t{{fill:{txt};font-size:16px;font-weight:600}}"
        f".s{{fill:{sub};font-size:13px}}.l{{fill:{sub};font-size:12px}}"
        f".h{{fill:{txt};font-size:20px;font-weight:700}}"
        f".c{{fill:{sub};font-size:12.5px;font-family:'JetBrains Mono',Consolas,monospace}}"
        "</style></defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{bg}"/>',
        '<text x="32" y="44" class="h">run_math.py: from two clicks to metres</text>',
        '<text x="32" y="68" class="s">Measurement path (solid) and offline calibration '
        'path (dashed). Pure NumPy + Pygame.</text>',
        # row 1: measurement
        box(32, 100, 200, 130, "Stereo pair", ["left + right image", "2592 × 1944 px",
                                               "CLAHE grayscale copy"], "in"),
        box(282, 100, 236, 130, "Γ-grid ray model", ["ray = [u_n·Γ, v_n·Γ, 1]",
                                                     "Γ: 5×5 bilinear grid", "per camera"],
            "model"),
        box(568, 100, 250, 130, "Right-image guidance", ["epipolar curve + 1/2/5 m rings",
                                                         "ZNCC row search → hint (+)",
                                                         "user clicks the match"], "match"),
        box(868, 100, 280, 130, "OLS triangulation", ["A = [v_L, −v_R],  A x = d",
                                                      "x = (AᵀA)⁻¹ Aᵀ d,  Z = x₀",
                                                      "→ depth Z in metres"], "solve"),
        arrow(232, 165, 280, 165), arrow(518, 165, 566, 165), arrow(818, 165, 866, 165),
        box(568, 268, 250, 92, "Sub-pixel refinement", ["SSD parabola + Lucas–Kanade",
                                                       "printed, info only"], "info",
            dashed=True),
        arrow(693, 230, 693, 266, dashed=True),
        # row 2: calibration
        box(32, 330, 200, 108, "Calibration points", ["≥ 15 clicks with a", "measured distance"],
            "in", dashed=True),
        box(282, 330, 236, 108, "train()", ["batch gradient descent,", "analytic gradients"],
            "model", dashed=True),
        arrow(232, 384, 280, 384, dashed=True),
        arrow(400, 328, 400, 232, "fits Γ_L, Γ_R, dx, dy", dashed=True),
        box(868, 268, 280, 170, "Loss (per point)", ["e_x = (v_Lx − v_Rx)·Z / dx − 1",
                                                     "e_y = (v_Ly − v_Ry) − dy / Z",
                                                     "+ Laplacian smoothness on Γ",
                                                     "centre node frozen as scale",
                                                     "anchor; 9,000 epochs"], "info",
            dashed=True),
        arrow(518, 384, 866, 384, "minimises", dashed=True),
        "</svg>",
    ]
    out = OUT / "pipeline.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")


# ─── leave-one-out (optional, slow) ───────────────────────────────────────────
def _loo_one(i):
    pts = rm.load_cal_pts(str(ROOT / rm.CAL_PTS_PATH))
    train_set = pts[:i] + pts[i + 1:]
    with contextlib.redirect_stdout(io.StringIO()):
        GL, GR, d = rm.train(train_set, verbose=10 ** 9)
    return rm.triangulate(*pts[i][:4], GL, GR, d)[0] - pts[i][4]


def loo():
    from multiprocessing import Pool
    n = len(rm.load_cal_pts(str(ROOT / rm.CAL_PTS_PATH)))
    with Pool(os.cpu_count()) as pool:
        err = np.array(pool.map(_loo_one, range(n)))
    a = np.abs(err[~np.isnan(err)])
    print(f"  leave-one-out over {n} points: MAE {a.mean():.3f} m, median {np.median(a):.3f} m,"
          f" max {a.max():.3f} m, failed {int(np.isnan(err).sum())}")
    return err


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sys.argv[1:] or ["plots", "math", "diagram", "demo"]
    for job in jobs:
        print(f"[{job}]")
        {"plots": make_plots, "math": make_math_pages, "diagram": make_diagram,
         "demo": record_demo, "loo": loo}[job]()
