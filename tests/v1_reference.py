"""
Verbatim copy of the numerical core of the original v1 ``run_math.py``
(commit 7b6bee0) — used only by ``test_legacy.py`` to prove that
``stereo_gamma.legacy`` reproduces v1 exactly.  Do not edit.
"""
# flake8: noqa
# ruff: noqa
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
    pass  # print(f"\n[TRAIN] N={N}  epochs={epochs}  5×5 grid  smoothness_w={SMOOTH_W}")

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

