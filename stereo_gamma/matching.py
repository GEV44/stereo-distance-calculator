"""
Correspondence search: CLAHE pre-processing, epipolar-guided ZNCC with
parabolic sub-pixel refinement, and Lucas–Kanade 2-D refinement.

The right-image search is **1-D along the model's epipolar curve** (which is
bent by lens distortion and tilted by the rig's roll), not along the image
row: with ~1° of roll the true match can sit tens of pixels above or below
the clicked row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── PRE-PROCESSING ───────────────────────────────────────────────────────────
def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma of an ``(H, W, 3)`` array → float32 ``(H, W)``."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def clahe(gray: np.ndarray, tiles: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation (Zuiderveld, 1994).

    Each tile's histogram is clipped at ``clip_limit × mean bin height``, the
    excess is redistributed uniformly, and the per-tile mappings are blended
    **bilinearly between tile centres** so no seams appear at tile borders
    (seams would create false ZNCC texture).
    """
    g = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
    H, W = g.shape
    th, tw = -(-H // tiles), -(-W // tiles)
    luts = np.empty((tiles, tiles, 256), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            tile = g[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if tile.size == 0:
                luts[i, j] = np.arange(256)
                continue
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            limit = max(1.0, clip_limit * tile.size / 256.0)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            luts[i, j] = 255.0 * (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-9)

    def axis(n, t, size):
        f = (np.arange(n) + 0.5) / size - 0.5
        i0 = np.clip(np.floor(f).astype(int), 0, t - 1)
        i1 = np.minimum(i0 + 1, t - 1)
        return i0, i1, np.clip(f - i0, 0.0, 1.0).astype(np.float32)

    y0, y1, wy = axis(H, tiles, th)
    x0, x1, wx = axis(W, tiles, tw)
    wy, wx = wy[:, None], wx[None, :]
    Y0, Y1, X0, X1 = y0[:, None], y1[:, None], x0[None, :], x1[None, :]
    return ((1 - wy) * ((1 - wx) * luts[Y0, X0, g] + wx * luts[Y0, X1, g])
            + wy * ((1 - wx) * luts[Y1, X0, g] + wx * luts[Y1, X1, g])).astype(np.float32)


# ─── ZNCC ─────────────────────────────────────────────────────────────────────
def _patches(img: np.ndarray, us: np.ndarray, vs: np.ndarray, half: int) -> np.ndarray:
    o = np.arange(-half, half + 1)
    return img[vs[:, None, None] + o[None, :, None], us[:, None, None] + o[None, None, :]]


def zncc_scores(Lg, Rg, uL: int, vL: int, us, vs, patch: int = 31):
    """
    Zero-mean normalised cross-correlation between the left patch at (uL, vL)
    and right patches centred at integer positions ``(us, vs)``:

        ZNCC(A, B) = Σ(A−Ā)(B−B̄) / (‖A−Ā‖ ‖B−B̄‖)   ∈ [−1, 1]

    Invariant to affine brightness/contrast changes.  Returns ``(scores,
    template_std)``; out-of-image candidates score −1.
    """
    half = patch // 2
    H, W = Rg.shape
    us, vs = np.asarray(us, int), np.asarray(vs, int)
    T = Lg[vL - half:vL + half + 1, uL - half:uL + half + 1].astype(np.float64)
    T = T - T.mean()
    nT = float(np.linalg.norm(T))
    inside = (us >= half) & (us < W - half) & (vs >= half) & (vs < H - half)
    scores = np.full(len(us), -1.0)
    if inside.any() and nT > 1e-6:
        P = _patches(Rg, us[inside], vs[inside], half).astype(np.float64)
        P -= P.mean(axis=(1, 2), keepdims=True)
        num = np.einsum("nij,ij->n", P, T)
        den = np.sqrt(np.einsum("nij,nij->n", P, P)) * nT + 1e-9
        scores[inside] = num / den
    return scores, nT / patch


def parabolic_peak(s_m1: float, s_0: float, s_p1: float) -> float:
    """Sub-sample offset Δ ∈ [−½, ½] of the vertex of the parabola through 3 scores."""
    den = s_m1 - 2.0 * s_0 + s_p1
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (s_m1 - s_p1) / den, -0.5, 0.5))


@dataclass
class Match:
    u: float
    v: float
    score: float  # ZNCC at the peak
    uniqueness: float  # best / second-best peak (≥ 1; higher = less ambiguous)
    texture: float  # template standard deviation (grey levels)
    lr_error: float = float("nan")  # left-right consistency error (px), NaN = not checked

    @property
    def reliable(self) -> bool:
        """
        Accept a match only if it is well-textured, correlates strongly and —
        when checked — survives the left-right consistency test (searching back
        from the right match must land within 2 px of the original click).
        """
        if not (self.score >= 0.6 and self.texture >= 2.0):
            return False
        if np.isfinite(self.lr_error):
            return self.lr_error <= 2.0
        return self.uniqueness >= 1.1


def match_along_curve(Lg, Rg, uL: int, vL: int, cu, cv, patch: int = 31) -> Match | None:
    """
    Best ZNCC match among curve samples ``(cu, cv)`` (ordered along the curve,
    spaced ≤ 1 px) with parabolic sub-pixel refinement along the curve.
    """
    half = patch // 2
    H, W = Lg.shape
    if not (half <= uL < W - half and half <= vL < H - half) or len(cu) < 3:
        return None
    cu, cv = np.asarray(cu, float), np.asarray(cv, float)
    s, texture = zncc_scores(Lg, Rg, uL, vL, np.rint(cu), np.rint(cv), patch)
    k = int(np.argmax(s))
    if s[k] <= -1.0:
        return None
    delta = parabolic_peak(s[k - 1], s[k], s[k + 1]) if 0 < k < len(s) - 1 else 0.0
    j = k + (1 if delta > 0 else -1)
    j = min(max(j, 0), len(s) - 1)
    u = cu[k] + abs(delta) * (cu[j] - cu[k])
    v = cv[k] + abs(delta) * (cv[j] - cv[k])
    far = np.hypot(cu - cu[k], cv - cv[k]) > patch
    second = float(s[far].max()) if far.any() else -1.0
    uniq = float(s[k] / second) if second > 1e-6 else float("inf")
    return Match(float(u), float(v), float(s[k]), uniq, float(texture))


def _densify(cu, cv):
    """Resample a polyline so consecutive samples are ≤ 1 px apart."""
    if len(cu) < 2:
        return cu, cv
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cu), np.diff(cv)))])
    t = np.arange(0.0, arc[-1] + 1e-9, 1.0)
    return np.interp(t, arc, cu), np.interp(t, arc, cv)


def dense_curve(model, uL: float, vL: float, z_min: float = 0.3, z_max: float = 60.0):
    """Right-image epipolar curve of (uL, vL), sampled every ≤ 1 px."""
    cu, cv, _ = model.epipolar_curve(uL, vL, z_min, z_max, n=300)
    return _densify(cu, cv)


def epipolar_match(model, Lg, Rg, uL: int, vL: int, patch: int = 31,
                   z_min: float = 0.3, z_max: float = 60.0, lr_check: bool = True) -> Match | None:
    """
    Epipolar-guided ZNCC search for the right-image correspondence of (uL, vL),
    followed (optionally) by the left-right consistency check.
    """
    cu, cv = dense_curve(model, uL, vL, z_min, z_max)
    m = match_along_curve(Lg, Rg, int(uL), int(vL), cu, cv, patch)
    if m is None or not lr_check:
        return m
    bu, bv = model.epipolar_curve_left(m.u, m.v, z_min, z_max, n=300)
    bu, bv = _densify(bu, bv)
    back = match_along_curve(Rg, Lg, int(round(m.u)), int(round(m.v)), bu, bv, patch)
    if back is not None:
        m.lr_error = float(np.hypot(back.u - uL, back.v - vL))
    else:
        m.lr_error = float("inf")
    return m


# ─── LUCAS–KANADE REFINEMENT ──────────────────────────────────────────────────
def _bilinear_patch(img, u: float, v: float, half: int):
    ui, vi = int(np.floor(u)), int(np.floor(v))
    fu, fv = u - ui, v - vi
    a = img[vi - half:vi + half + 1, ui - half:ui + half + 1]
    b = img[vi - half:vi + half + 1, ui - half + 1:ui + half + 2]
    c = img[vi - half + 1:vi + half + 2, ui - half:ui + half + 1]
    d = img[vi - half + 1:vi + half + 2, ui - half + 1:ui + half + 2]
    return (1 - fu) * (1 - fv) * a + fu * (1 - fv) * b + (1 - fu) * fv * c + fu * fv * d


def lk_refine(Lg, Rg, uL: int, vL: int, uR0: float, vR0: float, patch: int = 21,
              iters: int = 10):
    """
    Inverse-compositional Lucas–Kanade (translation only): refines the right
    match to sub-pixel precision in **both** axes.  Returns ``(u, v)`` or
    ``None`` near the border or on texture-less patches.
    """
    H, W = Rg.shape
    half = patch // 2
    lim = half + 2
    if not (lim <= uL < W - lim and lim <= vL < H - lim):
        return None
    T = Lg[vL - half - 1:vL + half + 2, uL - half - 1:uL + half + 2].astype(np.float64)
    Ix = 0.5 * (T[1:-1, 2:] - T[1:-1, :-2])
    Iy = 0.5 * (T[2:, 1:-1] - T[:-2, 1:-1])
    Tc = T[1:-1, 1:-1]
    Ixx, Iyy, Ixy = (Ix * Ix).sum(), (Iy * Iy).sum(), (Ix * Iy).sum()
    det = Ixx * Iyy - Ixy * Ixy
    if det < 1e-6 * (Ixx + Iyy + 1e-12) ** 2:
        return None
    u, v = float(uR0), float(vR0)
    for _ in range(iters):
        if not (lim <= u < W - lim and lim <= v < H - lim):
            return None
        e = _bilinear_patch(Rg, u, v, half) - Tc
        bx, by = (Ix * e).sum(), (Iy * e).sum()
        du, dv = (Iyy * bx - Ixy * by) / det, (Ixx * by - Ixy * bx) / det
        u, v = u - du, v - dv
        if abs(du) + abs(dv) < 1e-3:
            break
    if abs(u - uR0) > patch or abs(v - vR0) > patch:
        return None  # diverged
    return float(u), float(v)
