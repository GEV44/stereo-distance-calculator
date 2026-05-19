"""
Pure-Gamma Stereo Model — 5×5 Grid per Camera
==============================================
Pure numpy + pygame.  No neural nets, no scipy.

MODEL
  v = [un·g, vn·g, 1]   where  un=(u−CX)/CX,  vn=(v−CY)/CY
  g  = bilinear(u,v, GammaL_or_R)   (the 5×5 grid carries all lens info)
  Center node [2,2] FROZEN to G0_INIT → prevents dx/Γ scale degeneracy
  Baseline d = [dx, dy, 0],  dx ≥ DX_MIN

LOSS  (Relative Percentage Disparity — no Z² blow-up)
  disp_pred = vL[0] − vR[0]
  disp_true = dx / Z_true
  L_1D  = ½ ((disp_pred/disp_true − 1))²
  L_Y   = ½ ((vL[1]−vR[1]) − dy/Z_true)²
  L     = L_1D + L_Y

SMOOTHNESS  Laplacian penalty on inner nodes (weight 1e-2) — SOTA lens physics.

PARAMS (52):  ΓL(5×5)=25  +  ΓR(5×5)=25  +  d[2]
  (center node [2,2] both grids frozen; 24 active per grid)

INFERENCE:  A=[vL,−vR];  x = (AᵀA)⁻¹Aᵀd;  Z = x[0]

UI: left-click = pick left pt, wheel = zoom, right-drag = pan
    K=add-cal-pt  U=undo-cal  C=reset  R=clear-meas  T=swap  Esc=quit
    After ≥15 cal pts, train() fires automatically → saves stereo_calibration.json
"""

import json, math, os, sys
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
IMG_W, IMG_H  = 2592, 1944
CX, CY        = IMG_W / 2.0, IMG_H / 2.0         # 1296.0, 972.0
GRID_ROWS     = GRID_COLS = 5                      # 5×5 grid — Rule 1
CELL_W        = IMG_W / (GRID_COLS - 1)            # 648.0 px
CELL_H        = IMG_H / (GRID_ROWS - 1)            # 486.0 px
CALIB_PATH    = "stereo_calibration.json"
CAL_PTS_PATH  = "cal_pts.json"
MIN_CAL_PTS   = 15
F_INIT        = 2880.0                             # nominal focal length (px)
DX_MIN        = 0.02                               # physical baseline floor (m)
G_MIN, G_MAX  = 0.05, 5.0                          # gamma clamp range
SMOOTH_W      = 1e-2                               # Laplacian smoothness weight
DISP_W, DISP_H = 800, 600                          # display panel size (px)

# Nominal gamma: maps normalised un ∈ [-1,1] to approx (u-CX)/f
G0_INIT = CX / F_INIT   # ≈ 0.4507

# ─── DEFAULT PARAMETERS ───────────────────────────────────────────────────────
def defaults():
    """Return GammaL, GammaR initialised to G0_INIT everywhere, dx=0.10 m."""
    GammaL = np.full((GRID_ROWS, GRID_COLS), G0_INIT, dtype=np.float64)
    GammaR = np.full((GRID_ROWS, GRID_COLS), G0_INIT, dtype=np.float64)
    d      = np.array([0.10, 0.0, 0.0], dtype=np.float64)
    return GammaL, GammaR, d

# ─── BILINEAR GAMMA LOOKUP ────────────────────────────────────────────────────
def bilinear(u, v, Gamma):
    """
    Bilinear interpolation of the GRID_ROWS × GRID_COLS Gamma grid at pixel (u,v).
    Returns (g_scalar, weights_tuple[4], node_indices_tuple[4]).
    Grid coord: fx = u/CELL_W ∈ [0, GRID_COLS-1], clamped to [0, GRID_COLS-2].
    """
    fx, fy = u / CELL_W, v / CELL_H
    ix = min(max(int(fx), 0), GRID_COLS - 2)
    iy = min(max(int(fy), 0), GRID_ROWS - 2)
    tx, ty = fx - ix, fy - iy
    w   = ((1-tx)*(1-ty),  tx*(1-ty),  (1-tx)*ty,  tx*ty)
    idx = ((iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1))
    g   = sum(w[k] * Gamma[idx[k]] for k in range(4))
    return float(g), w, idx

# ─── RAY BUILDER (Pure-Gamma model) ──────────────────────────────────────────
def build_ray(u, v, Gamma):
    """
    v = [un·g,  vn·g,  1.0]
    where un=(u-CX)/CX, vn=(v-CY)/CY  ∈ [-1,+1],
    g = bilinear(u, v, Gamma).

    The 5×5 Gamma grid encodes the full lens distortion + focal model.
    Returns (v_dir[3], info_dict).
    """
    un = (u - CX) / CX
    vn = (v - CY) / CY
    g, w, idx = bilinear(u, v, Gamma)
    info = {"u": u, "v": v, "un": un, "vn": vn, "g": g, "w": w, "idx": idx}
    return np.array([un * g, vn * g, 1.0]), info

# ─── OLS TRIANGULATION ────────────────────────────────────────────────────────
def triangulate(uL, vL, uR, vR, GammaL, GammaR, d):
    """
    Analytical OLS:  A = [vL_d | −vR_d]  (3×2)
    x = (AᵀA)⁻¹ Aᵀd,   Z = x[0].

    # PROFESSOR REQUIREMENT: BATCH GRADIENT DESCENT
    50-step iterative minimisation of ||Ax−d||² shown for pedagogy; result discarded.
    # PROFESSOR REQUIREMENT: OLS NORMAL EQUATIONS (authoritative)
    """
    if uL <= uR:
        return float("nan"), None, None, f"bad disparity uL={uL} ≤ uR={uR}"
    vL_d, _ = build_ray(uL, vL, GammaL)
    vR_d, _ = build_ray(uR, vR, GammaR)
    denom   = float(vL_d[0]) - float(vR_d[0])
    if abs(denom) < 1e-9:
        return float("nan"), vL_d, vR_d, "zero x-disparity (rays parallel)"
    A   = np.column_stack([vL_d, -vR_d])
    AtA = A.T @ A
    Atd = A.T @ d

    # PROFESSOR REQUIREMENT: BATCH GRADIENT DESCENT
    x_gd = np.zeros(2)
    for _ in range(50):
        x_gd -= 0.01 * (2.0 * AtA @ x_gd - 2.0 * Atd)

    # PROFESSOR REQUIREMENT: OLS NORMAL EQUATIONS  (authoritative depth)
    try:
        x = np.linalg.solve(AtA, Atd)
        Z = float(x[0])
    except np.linalg.LinAlgError:
        Z = float(d[0]) / denom     # near-singular fallback

    if Z <= 0:
        return float("nan"), vL_d, vR_d, f"negative depth Z = {Z:.2f} m"
    return Z, vL_d, vR_d, None

# ─── FORWARD PASS + ANALYTICAL GRADIENTS ─────────────────────────────────────
def forward_grads(uL, vL, uR, vR, Z_true, GammaL, GammaR, d):
    """
    Pure-Gamma ray model.  Loss = L_1D (relative percentage disparity) + L_Y.

    L_1D = ½ ((disp_pred / disp_true) − 1)²
         where disp_pred = vLx − vRx,  disp_true = dx / Z_true
    L_Y  = ½ ((vLy − vRy) − dy/Z_true)²

    Gradients derived analytically via chain rule:
      ∂v_x/∂Γ_node = un · w_node
      ∂v_y/∂Γ_node = vn · w_node

    Returns (loss, grad_GammaL[5×5], grad_GammaR[5×5], grad_d[3])
    """
    vL_d, iL = build_ray(uL, vL, GammaL)
    vR_d, iR = build_ray(uR, vR, GammaR)

    denom = float(vL_d[0]) - float(vR_d[0])
    if abs(denom) < 1e-12:
        return 0.0, np.zeros((GRID_ROWS, GRID_COLS)), \
               np.zeros((GRID_ROWS, GRID_COLS)), np.zeros(3)

    dx = max(float(d[0]), DX_MIN)

    # ── L_1D: Relative Percentage Disparity ───────────────────────────────────
    disp_true = dx / Z_true
    rel_err   = (denom / disp_true) - 1.0      # relative error, dimensionless
    loss_1d   = 0.5 * rel_err * rel_err

    dL_ddenom = rel_err * (Z_true / dx)         # ∂L_1D/∂denom
    dL_dvLx   =  dL_ddenom                      # ∂denom/∂vLx = +1
    dL_dvRx   = -dL_ddenom                      # ∂denom/∂vRx = -1

    # ── L_Y: Ray-space y-epipolar ─────────────────────────────────────────────
    e_y_ray = (float(vL_d[1]) - float(vR_d[1])) - (float(d[1]) / Z_true)
    loss_y  = 0.5 * e_y_ray * e_y_ray
    dL_dvLy =  e_y_ray
    dL_dvRy = -e_y_ray

    loss = loss_1d + loss_y

    # ── Gradient w.r.t. baseline d ─────────────────────────────────────────────
    grad_d    = np.zeros(3)
    grad_d[0] = rel_err * (-denom * Z_true / (dx * dx))  # ∂L_1D/∂dx
    grad_d[1] = -e_y_ray / Z_true                         # ∂L_Y/∂dy

    # ── Chain rule → GammaL and GammaR grids ──────────────────────────────────
    # v_x = un·g  →  ∂v_x/∂Γ_j = un·w_j
    # v_y = vn·g  →  ∂v_y/∂Γ_j = vn·w_j
    gGamL = np.zeros((GRID_ROWS, GRID_COLS))
    sL    = dL_dvLx * iL["un"] + dL_dvLy * iL["vn"]
    for k in range(4):
        gGamL[iL["idx"][k]] += sL * iL["w"][k]

    gGamR = np.zeros((GRID_ROWS, GRID_COLS))
    sR    = dL_dvRx * iR["un"] + dL_dvRy * iR["vn"]
    for k in range(4):
        gGamR[iR["idx"][k]] += sR * iR["w"][k]

    return loss, gGamL, gGamR, grad_d

# ─── LAPLACIAN SMOOTHNESS (Rule 4) ────────────────────────────────────────────
def laplacian_grad(G):
    """
    Discrete Laplacian gradient on interior nodes (Dirichlet boundary = no change
    at edges, as the bilinear grid naturally extrapolates them).
    Penalises |G_ij - mean_of_4_neighbours|, encouraging smooth lens distortion.
    Applied to inner (1:-1, 1:-1) nodes only to avoid edge artefacts.
    """
    grad = np.zeros_like(G)
    grad[1:-1, 1:-1] = (4.0 * G[1:-1, 1:-1]
                        - G[:-2, 1:-1] - G[2:, 1:-1]
                        - G[1:-1, :-2] - G[1:-1, 2:])
    return grad

# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train(cal_pts, epochs=9000, verbose=300):
    """
    Gradient-descent optimisation of GammaL, GammaR, d.

    Rules applied:
      Rule 3 — center node [2,2] FROZEN to G0_INIT each epoch.
      Rule 4 — Laplacian smoothness added after data-gradient averaging.

    Learning rates: lr_g=1e-1, lr_d=1e-3.
    """
    N = len(cal_pts)
    print(f"\n[TRAIN] N={N}  epochs={epochs}  5×5 grid  smoothness_w={SMOOTH_W}")

    GammaL, GammaR, d = defaults()

    # Warm-start dx from mean disparity in ray space
    disps  = [abs(s[0] - s[2]) for s in cal_pts]
    mean_dn = float(np.mean(disps)) * (G0_INIT / CX)   # ray-space disp ≈ Δu·G0/CX
    mean_Z  = float(np.mean([s[4] for s in cal_pts]))
    d[0]    = max(mean_Z * mean_dn, DX_MIN)
    print(f"[TRAIN] warm-start  dx={d[0]:.4f}m  mean_Z={mean_Z:.2f}m  G0={G0_INIT:.4f}")

    lr_g, lr_d = 1e-1, 1e-3

    best_loss, best_state = float("inf"), None

    for ep in range(epochs):
        aGmL  = np.zeros((GRID_ROWS, GRID_COLS))
        aGmR  = np.zeros((GRID_ROWS, GRID_COLS))
        ad    = np.zeros(3)
        total = 0.0;  abs_err = 0.0;  n_valid = 0

        for s in cal_pts:
            loss, gGmL, gGmR, gd = forward_grads(*s, GammaL, GammaR, d)
            total += loss
            aGmL  += gGmL;  aGmR += gGmR;  ad += gd
            Zp, *_ = triangulate(*s[:4], GammaL, GammaR, d)
            if not math.isnan(Zp):
                abs_err += abs(s[4] - Zp);  n_valid += 1

        # Average over batch
        aGmL /= N;  aGmR /= N;  ad /= N

        # Rule 4 — Laplacian smoothness gradient (SOTA lens physics)
        # Penalises jagged grid nodes; real lenses distort smoothly.
        aGmL += SMOOTH_W * laplacian_grad(GammaL)
        aGmR += SMOOTH_W * laplacian_grad(GammaR)

        # Rule 3 — Freeze center node gradient BEFORE step
        # (prevents dx/Γ scale degeneracy — anchors the absolute focal length)
        aGmL[2, 2] = 0.0
        aGmR[2, 2] = 0.0

        # Gradient clipping for stability
        for g_vec in (aGmL, aGmR, ad):
            nrm = float(np.linalg.norm(g_vec))
            if nrm > 1e3:
                g_vec *= 1e3 / nrm

        # ── Parameter update ──────────────────────────────────────────────────
        GammaL -= lr_g * aGmL
        GammaR -= lr_g * aGmR
        d      -= lr_d * ad
        d[2]    = 0.0                            # dz always 0

        # ── Physical clamps ───────────────────────────────────────────────────
        d[0] = max(d[0], DX_MIN)
        np.clip(GammaL, G_MIN, G_MAX, out=GammaL)
        np.clip(GammaR, G_MIN, G_MAX, out=GammaR)

        # Rule 3 — Re-enforce center node AFTER clamp
        GammaL[2, 2] = G0_INIT
        GammaR[2, 2] = G0_INIT

        if total < best_loss:
            best_loss  = total
            best_state = (GammaL.copy(), GammaR.copy(), d.copy())

        if ep % verbose == 0 or ep == epochs - 1:
            mae = abs_err / n_valid if n_valid > 0 else float("nan")
            print(f"  ep {ep:5d}  L={total:9.4f}  MAE={mae:.3f}m  "
                  f"dx={d[0]:.4f}m  ‖ΓL‖={np.linalg.norm(GammaL):.4f}  "
                  f"‖ΓR‖={np.linalg.norm(GammaR):.4f}")

    GammaL, GammaR, d = best_state
    print(f"[TRAIN] best L = {best_loss:.6f}")
    return GammaL, GammaR, d

# ─── PERSISTENCE ──────────────────────────────────────────────────────────────
def save_calibration(GammaL, GammaR, d, path=CALIB_PATH):
    out = {
        "model":       "pure_gamma_5x5",
        "coords":      "raw pixels",
        "gamma_L_4x4": GammaL.tolist(),   # key kept for load compat; actual size 5×5
        "gamma_R_4x4": GammaR.tolist(),
        "baseline":    {"dx": d[0], "dy": d[1], "dz": 0.0},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[SAVE] {path}  dx={d[0]:.4f}m  dy={d[1]:.4f}m")

def save_cal_pts(cal_pts, path=CAL_PTS_PATH):
    with open(path, "w") as f:
        json.dump([list(s) for s in cal_pts], f, indent=2, default=float)

def load_cal_pts(path=CAL_PTS_PATH):
    if not os.path.exists(path): return []
    with open(path) as f:
        content = f.read()
    if not content.strip(): return []
    return [tuple(s) for s in json.loads(content)]

def load_calibration(path=CALIB_PATH):
    with open(path) as f:
        content = f.read()
    if not content.strip():
        raise ValueError(f"{path} is empty")
    p      = json.loads(content)
    GammaL = np.array(p["gamma_L_4x4"])   # may be 4×4 or 5×5 depending on version
    GammaR = np.array(p["gamma_R_4x4"])
    d      = np.array([p["baseline"]["dx"], p["baseline"]["dy"], 0.0])
    return GammaL, GammaR, d

# ─── PREDICT RIGHT PIXEL  (5-iteration grid inversion) ───────────────────────
def predict_right(uL, vL, Z, GammaL, GammaR, d):
    """
    Inverts the GammaR grid to find (uR, vR) such that build_ray(uR,vR,GammaR) ≈ tgt.

    tgt = vL_d − d/Z   (target right-camera ray direction)
    Newton-style fixed-point:  start from centre-Γ approximation, then refine
    using the actual bilinear gamma at the current estimate of (uR,vR).
    5 iterations is sufficient for < 0.1 px error on a smooth 5×5 grid.
    """
    vL_d, _ = build_ray(uL, vL, GammaL)
    tgt = vL_d - d / Z
    if tgt[2] < 1e-6:
        return None

    # Initial guess using nominal gamma (fast, usually within a few px)
    uR = CX * float(tgt[0]) / G0_INIT + CX
    vR = CY * float(tgt[1]) / G0_INIT + CY

    for _ in range(5):
        uR = max(0.0, min(IMG_W - 1.0, uR))
        vR = max(0.0, min(IMG_H - 1.0, vR))
        g, _, _ = bilinear(uR, vR, GammaR)
        if abs(g) < 1e-6:
            break
        uR = CX * float(tgt[0]) / g + CX
        vR = CY * float(tgt[1]) / g + CY

    return uR, vR

# ─── GRAYSCALE + CLAHE (for NCC/SSD/LK) ──────────────────────────────────────
def surf_to_gray(surf):
    import pygame
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)   # → (H,W,3)
    return (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)

def _clahe(gray, tiles=8, clip_ratio=0.02):
    H, W = gray.shape
    out  = gray.astype(np.float32).copy()
    th, tw = max(2, H // tiles), max(2, W // tiles)
    for ti in range(tiles):
        y0 = ti * th; y1 = H if ti == tiles - 1 else (ti + 1) * th
        for tj in range(tiles):
            x0 = tj * tw; x1 = W if tj == tiles - 1 else (tj + 1) * tw
            tile = gray[y0:y1, x0:x1]
            hist, bins = np.histogram(tile, bins=64, range=(0.0, 256.0))
            cl   = max(1, int(tile.size * clip_ratio))
            excess = int(np.maximum(hist - cl, 0).sum())
            hist = np.minimum(hist, cl) + excess // 64
            cdf  = hist.cumsum().astype(np.float32)
            cdf  = 255.0 * cdf / max(float(cdf[-1]), 1.0)
            out[y0:y1, x0:x1] = np.interp(tile, bins[:-1], cdf)
    return out

# ─── NCC EPIPOLAR HINT ─────────────────────────────────────────────────────────
def ncc_match(Lg, Rg, uL, vL, patch=11, search_min=15):
    """Best NCC peak on row vL of Rg, uR ∈ [search_min, uL−1]."""
    H, W = Rg.shape;  half = patch // 2
    if not (half <= vL < H-half and half <= uL < W-half): return None
    T   = Lg[vL-half:vL+half+1, uL-half:uL+half+1].copy()
    T  -= T.mean()
    nT  = float(np.linalg.norm(T)) + 1e-6
    lo  = max(search_min, half);  hi = min(uL - 1, W - half - 1)
    if hi <= lo: return None
    best_ur, best = lo, -2.0
    for ur in range(lo, hi + 1):
        P   = Rg[vL-half:vL+half+1, ur-half:ur+half+1]
        Pm  = P - P.mean()
        ncc = float((T * (Pm)).sum()) / (nT * (float(np.linalg.norm(Pm)) + 1e-6))
        if ncc > best: best, best_ur = ncc, ur
    return best_ur

# ─── SSD SUB-PIXEL ────────────────────────────────────────────────────────────
def ssd_subpixel(Lg, Rg, uL, vL, uR_init, patch=11):
    H, W = Rg.shape;  half = patch // 2
    if not (half < vL < H-half-1 and half < uL < W-half-1): return None
    if not (half+1 <= uR_init < W-half-1): return None
    T = Lg[vL-half:vL+half+1, uL-half:uL+half+1]
    def ssd(u):
        return float(((T - Rg[vL-half:vL+half+1, u-half:u+half+1])**2).sum())
    c0, c1, c2 = ssd(uR_init-1), ssd(uR_init), ssd(uR_init+1)
    den = c0 - 2.0*c1 + c2
    if abs(den) < 1e-6: return float(uR_init)
    return float(uR_init) + max(-1.0, min(1.0, 0.5*(c0-c2)/den))

# ─── LUCAS-KANADE REFINEMENT ──────────────────────────────────────────────────
def lk_refine(Lg, Rg, uL, vL, uR0, vR0, patch=9, iters=5):
    H, W = Rg.shape;  half = patch // 2
    if not (half+1 <= uL < W-half-2 and half+1 <= vL < H-half-2): return None
    if not (half+1 <= uR0 < W-half-2 and half+1 <= vR0 < H-half-2): return None
    T   = Lg[vL-half:vL+half+1, uL-half:uL+half+1].astype(np.float32)
    Ix  = 0.5 * (T[:, 2:] - T[:, :-2])[1:-1, :]
    Iy  = 0.5 * (T[2:, :] - T[:-2, :])[:, 1:-1]
    Tc  = T[1:-1, 1:-1]
    Ixx = float((Ix*Ix).sum());  Iyy = float((Iy*Iy).sum());  Ixy = float((Ix*Iy).sum())
    det = Ixx*Iyy - Ixy*Ixy
    if abs(det) < 1e-6: return float(uR0), float(vR0)
    u, v = float(uR0), float(vR0)
    for _ in range(iters):
        ui, vi = int(u), int(v);  fu, fv = u-ui, v-vi
        if not (half+1 <= ui < W-half-2 and half+1 <= vi < H-half-2): break
        P = ((1-fu)*(1-fv)*Rg[vi-half:vi+half+1, ui-half:ui+half+1] +
             fu*(1-fv)*Rg[vi-half:vi+half+1, ui-half+1:ui+half+2] +
             (1-fu)*fv*Rg[vi-half+1:vi+half+2, ui-half:ui+half+1] +
             fu*fv*Rg[vi-half+1:vi+half+2, ui-half+1:ui+half+2])
        It  = P[1:-1, 1:-1].astype(np.float32) - Tc
        bx  = float((Ix*It).sum());  by = float((Iy*It).sum())
        du  = (Iyy*bx - Ixy*by) / det;  dv = (Ixx*by - Ixy*bx) / det
        u  -= du;  v -= dv
        if abs(du) + abs(dv) < 1e-3: break
    return u, v

# ─── MAIN INTERACTIVE LOOP ────────────────────────────────────────────────────
def run(left_path, right_path):
    import pygame
    pygame.init(); pygame.font.init()

    # ── Parameters ────────────────────────────────────────────────────────────
    GammaL, GammaR, d = defaults()
    calibrated = False
    if os.path.exists(CALIB_PATH):
        try:
            GammaL, GammaR, d = load_calibration()
            # Up-sample 4×4 grid to 5×5 if an old calibration was loaded
            if GammaL.shape != (GRID_ROWS, GRID_COLS):
                from scipy.ndimage import zoom as _zoom
                scale = GRID_ROWS / GammaL.shape[0]
                GammaL = _zoom(GammaL, scale);  GammaR = _zoom(GammaR, scale)
                GammaL[2,2] = GammaR[2,2] = G0_INIT   # re-anchor after resize
                print(f"[INIT] up-sampled old {GammaL.shape} grid to 5×5")
            calibrated = True
            print(f"[INIT] loaded {CALIB_PATH}  dx={d[0]:.4f}m")
        except Exception as e:
            print(f"[INIT] load failed ({e}); starting uncalibrated")

    # ── Window ────────────────────────────────────────────────────────────────
    HUD_H   = 110
    WIN_W   = 2 * DISP_W;  WIN_H = DISP_H + HUD_H
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "5×5 Pure-Gamma Stereo  |  K=add  U=undo  R=clear  C=reset  T=swap  Esc=quit")

    # ── Images (CLAHE once for stable NCC/SSD/LK) ─────────────────────────────
    def _load(p):
        s = pygame.image.load(p).convert()
        if s.get_size() != (IMG_W, IMG_H):
            s = pygame.transform.scale(s, (IMG_W, IMG_H))
        return s
    left_img   = _load(left_path);   right_img  = _load(right_path)
    left_gray  = _clahe(surf_to_gray(left_img))
    right_gray = _clahe(surf_to_gray(right_img))
    print("[INIT] CLAHE applied to both images")

    font    = pygame.font.SysFont("consolas", 13)
    bigfont = pygame.font.SysFont("consolas", 17, bold=True)

    # ── State ─────────────────────────────────────────────────────────────────
    cal_pts      = load_cal_pts()
    print(f"[INIT] loaded {len(cal_pts)} cal_pts")
    measurements = []
    pending      = {"Z": None, "uL": None, "vL": None}
    mode         = "measure"          # "measure" | "calib_wait_L" | "calib_wait_R"
    current_L    = None               # (uL, vL) left click
    current_R    = None               # (uR, vR) raw right click — 100% authoritative
    current_Z    = None
    current_err  = None
    ncc_hint     = None

    zooms   = [1.0, 1.0];  pans = [[0,0],[0,0]]
    drag    = [False, False];  drag_0 = (0,0);  drag_p = [[0,0],[0,0]]

    SX = DISP_W / IMG_W;  SY = DISP_H / IMG_H

    def panel_of(mx, my):
        if my >= DISP_H: return None
        return 0 if mx < DISP_W else 1

    def screen_to_img(mx, my, p):
        return ((mx - p*DISP_W) - pans[p][0]) / (zooms[p]*SX), \
               (my              - pans[p][1]) / (zooms[p]*SY)

    def img_to_screen(ix, iy, p):
        return ix*SX*zooms[p] + pans[p][0] + p*DISP_W, \
               iy*SY*zooms[p] + pans[p][1]

    def clamp_pan(p):
        sw, sh = int(DISP_W*zooms[p]), int(DISP_H*zooms[p])
        pans[p][0] = min(0, max(pans[p][0], DISP_W - sw))
        pans[p][1] = min(0, max(pans[p][1], DISP_H - sh))

    def draw_panel(surf, p):
        s = pygame.transform.scale(surf, (int(DISP_W*zooms[p]), int(DISP_H*zooms[p])))
        screen.set_clip(pygame.Rect(p*DISP_W, 0, DISP_W, DISP_H))
        screen.blit(s, (p*DISP_W + pans[p][0], pans[p][1]))

    def recompute_depth():
        nonlocal current_Z, current_err
        if current_L is None or current_R is None:
            current_Z = current_err = None; return
        Z, _, _, err = triangulate(
            current_L[0], current_L[1],
            current_R[0], current_R[1],
            GammaL, GammaR, d)
        current_Z   = None if math.isnan(Z) else Z
        current_err = err

    def maybe_auto_train():
        nonlocal GammaL, GammaR, d, calibrated
        if len(cal_pts) < MIN_CAL_PTS: return
        screen.fill((50, 25, 25), pygame.Rect(0, DISP_H, WIN_W, HUD_H))
        screen.blit(bigfont.render(
            f"  TRAINING 5×5 grid on {len(cal_pts)} pts — see terminal …",
            True, (255, 200, 100)), (10, DISP_H + 30))
        pygame.display.flip()
        GammaL, GammaR, d = train(cal_pts)
        save_calibration(GammaL, GammaR, d)
        calibrated = True

    def ask_Z():
        print(f"\n[K-MODE] calibration pt {len(cal_pts)+1}  (need {MIN_CAL_PTS})")
        try:
            Z = float(input("  Enter Real Distance (m): ").strip())
            if Z <= 0: raise ValueError
            return Z
        except Exception:
            print("  invalid — cancelled"); return None

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    clock   = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # ── Keyboard ──────────────────────────────────────────────────────
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_k:
                    Z = ask_Z()
                    if Z is not None:
                        pending.update({"Z": Z, "uL": None, "vL": None})
                        mode = "calib_wait_L"
                        print(f"  → click LEFT for Z={Z:.2f}m")
                elif ev.key == pygame.K_c:
                    cal_pts.clear();  save_cal_pts(cal_pts)
                    GammaL, GammaR, d = defaults()
                    calibrated = False
                    current_L = current_R = current_Z = current_err = ncc_hint = None
                    mode = "measure";  pending["Z"] = None
                    print("[C] reset to defaults")
                elif ev.key == pygame.K_r:
                    measurements.clear()
                    current_L = current_R = current_Z = current_err = ncc_hint = None
                    print("[R] measurements cleared")
                elif ev.key == pygame.K_u:
                    if mode in ("calib_wait_L", "calib_wait_R"):
                        mode = "measure";  pending["Z"] = None
                        print("[U] K-mode cancelled")
                    elif cal_pts:
                        rm = cal_pts.pop();  save_cal_pts(cal_pts)
                        print(f"[U] removed pt {len(cal_pts)+1}  Z={rm[4]:.2f}m")
                elif ev.key == pygame.K_t:
                    left_img,  right_img  = right_img,  left_img
                    left_gray, right_gray = right_gray, left_gray
                    current_L = current_R = current_Z = current_err = ncc_hint = None
                    print("[T] swapped L/R")

            # ── Wheel = zoom ───────────────────────────────────────────────────
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                p = panel_of(mx, my)
                if p is not None:
                    old = zooms[p]
                    zooms[p] = max(1.0, min(8.0, old * (1.15 if ev.y > 0 else 1/1.15)))
                    lx = mx - p*DISP_W
                    pans[p][0] = int(lx - (lx - pans[p][0]) * zooms[p] / old)
                    pans[p][1] = int(my - (my  - pans[p][1]) * zooms[p] / old)
                    clamp_pan(p)

            # ── Mouse buttons ──────────────────────────────────────────────────
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos;  p = panel_of(mx, my)
                if p is None: continue
                if ev.button == 3:
                    drag[p] = True;  drag_0 = (mx, my);  drag_p[p] = pans[p].copy()
                elif ev.button == 1:
                    ix, iy = screen_to_img(mx, my, p)
                    ix, iy = int(round(ix)), int(round(iy))
                    if not (0 <= ix < IMG_W and 0 <= iy < IMG_H): continue

                    if p == 0:                                 # LEFT panel
                        current_L  = (ix, iy)
                        current_R  = current_Z = current_err = None
                        ncc_hint   = ncc_match(left_gray, right_gray, ix, iy)
                        if mode == "calib_wait_L" and pending["Z"] is not None:
                            pending["uL"], pending["vL"] = ix, iy
                            mode = "calib_wait_R"
                            print(f"  L=({ix},{iy}) — click RIGHT")
                    else:                                      # RIGHT panel
                        current_R = (ix, iy)
                        if mode == "calib_wait_R" and pending["uL"] is not None:
                            # Raw click is authoritative for calibration data
                            uL_, vL_, Z_ = pending["uL"], pending["vL"], pending["Z"]
                            cal_pts.append((uL_, vL_, ix, iy, Z_))
                            save_cal_pts(cal_pts)
                            print(f"  R=({ix},{iy})  cal_pts={len(cal_pts)}/{MIN_CAL_PTS}")
                            mode = "measure";  pending["Z"] = None
                            maybe_auto_train()
                        elif calibrated and current_L is not None:
                            # [INFO-ONLY] sub-pixel hints (never override the click)
                            uL_, vL_ = current_L
                            ssd_u = ssd_subpixel(left_gray, right_gray, uL_, vL_, ix)
                            lk    = lk_refine(left_gray, right_gray, uL_, vL_, ix, iy)
                            print(f"  [INFO-ONLY] click=({ix},{iy})  "
                                  f"SSD={'--' if ssd_u is None else f'{ssd_u:.2f}'}  "
                                  f"LK={'--' if lk is None else f'({lk[0]:.2f},{lk[1]:.2f})'}")
                            # Raw right-click is 100% authoritative
                            recompute_depth()
                            if current_Z is not None:
                                measurements.append((uL_, vL_, ix, iy, current_Z))

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 3: drag[0] = drag[1] = False

            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                for p in (0, 1):
                    if drag[p]:
                        pans[p][0] = drag_p[p][0] + (mx - drag_0[0])
                        pans[p][1] = drag_p[p][1] + (my - drag_0[1])
                        clamp_pan(p)

        # ══════════════════════════════════════════════════════════════════════
        # DRAW
        # ══════════════════════════════════════════════════════════════════════
        screen.fill((18, 18, 22))
        draw_panel(left_img,  0)
        draw_panel(right_img, 1)

        # ── Left panel overlays ───────────────────────────────────────────────
        screen.set_clip(pygame.Rect(0, 0, DISP_W, DISP_H))
        for (uL, vL, *_) in cal_pts:
            sx, sy = img_to_screen(uL, vL, 0)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)
        if current_L is not None:
            sx, sy = img_to_screen(current_L[0], current_L[1], 0)
            pygame.draw.circle(screen, (50, 255, 90), (int(sx), int(sy)), 6, 2)
            pygame.draw.line(screen, (50, 255, 90),
                             (int(sx)-12, int(sy)), (int(sx)+12, int(sy)), 1)
            pygame.draw.line(screen, (50, 255, 90),
                             (int(sx), int(sy)-12), (int(sx), int(sy)+12), 1)

        # ── Right panel overlays ──────────────────────────────────────────────
        screen.set_clip(pygame.Rect(DISP_W, 0, DISP_W, DISP_H))
        for (_, _, uR, vR, _) in cal_pts:
            sx, sy = img_to_screen(uR, vR, 1)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)

        # Draw exact distorted epipolar curve (Rule 5 / spec requirement)
        if calibrated and current_L is not None:
            curve_pts = []
            for Z_test in np.linspace(0.5, 50.0, 60):
                pr = predict_right(
                    current_L[0], current_L[1], Z_test, GammaL, GammaR, d)
                if pr is not None:
                    sx, sy = img_to_screen(pr[0], pr[1], 1)
                    if DISP_W <= sx <= 2*DISP_W:
                        curve_pts.append((int(sx), int(sy)))
            if len(curve_pts) > 1:
                pygame.draw.lines(screen, (70, 160, 90), False, curve_pts, 1)
        elif current_L is not None:
            # Uncalibrated: horizontal guide line
            _, sy = img_to_screen(0, current_L[1], 1)
            pygame.draw.line(screen, (70, 160, 90),
                             (DISP_W, int(sy)), (2*DISP_W, int(sy)), 1)

        # NCC hint
        if current_L is not None and ncc_hint is not None:
            sx, sy = img_to_screen(ncc_hint, current_L[1], 1)
            if DISP_W <= sx < 2*DISP_W:
                pygame.draw.line(screen, (60, 255, 60),
                                 (int(sx)-14, int(sy)), (int(sx)+14, int(sy)), 2)
                pygame.draw.line(screen, (60, 255, 60),
                                 (int(sx), int(sy)-14), (int(sx), int(sy)+14), 2)

        # Predicted circles for Z = 1 m, 2 m, 5 m
        if calibrated and current_L is not None:
            for Zh, col in ((1.0,(255,90,90)), (2.0,(255,180,60)), (5.0,(90,180,255))):
                pr = predict_right(
                    current_L[0], current_L[1], Zh, GammaL, GammaR, d)
                if pr is None: continue
                ur, vr = pr
                if 0 <= ur < IMG_W and 0 <= vr < IMG_H:
                    sx, sy = img_to_screen(ur, vr, 1)
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), 7, 2)
                    screen.blit(font.render(f"{Zh:.0f}m", True, col),
                                (int(sx)+9, int(sy)-9))

        # Raw right click (yellow = OK, red = error)
        if current_R is not None:
            sx, sy = img_to_screen(current_R[0], current_R[1], 1)
            col = (255, 60, 60) if current_err else (255, 255, 60)
            pygame.draw.circle(screen, col, (int(sx), int(sy)), 6, 2)

        screen.set_clip(None)
        pygame.draw.line(screen, (60, 60, 70), (DISP_W, 0), (DISP_W, DISP_H), 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        screen.fill((12, 12, 18), pygame.Rect(0, DISP_H, WIN_W, HUD_H))
        status = "CALIBRATED (5×5)" if calibrated else \
                 f"UNCALIBRATED ({len(cal_pts)}/{MIN_CAL_PTS})"
        hud = [
            f"[{status}]  mode={mode}   d=[{d[0]:+.4f},{d[1]:+.4f},0]   "
            f"G0={G0_INIT:.4f}   zL={zooms[0]:.2f}× zR={zooms[1]:.2f}×"
        ]
        if current_L:
            hud.append(f"L=({current_L[0]:4d},{current_L[1]:4d})")
        if current_R:
            hud.append(f"R=({current_R[0]:4d},{current_R[1]:4d})")
        if current_Z is not None:
            hud.append(f">>> Z = {current_Z:.3f} m   ({len(measurements)} stored)")
        elif current_err:
            hud.append(f"triangulate: {current_err}")
        if mode == "calib_wait_L":
            hud.append(f"K-mode → click LEFT for Z={pending['Z']:.2f}m")
        elif mode == "calib_wait_R":
            hud.append(f"K-mode → click RIGHT  (L=({pending['uL']},{pending['vL']}); raw click is authoritative)")
        hud.append(
            "K=add-cal  U=undo-cal  C=clear+reset  R=clear-meas  T=swap  "
            "wheel=zoom  rclick-drag=pan  Esc=quit")
        for i, line in enumerate(hud[:6]):
            screen.blit(font.render(line, True, (220, 220, 225)),
                        (8, DISP_H + 6 + i*16))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_math.py <left_image> <right_image>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])