"""Correspondence helpers: grayscale, CLAHE, ZNCC epipolar search, sub-pixel refinement.

All functions take plain ``float32`` ``(H, W)`` arrays, so they work on images
loaded with Pillow, Pygame or anything else.  Matching is only a *hint* in the
UI - the user's raw right click stays authoritative.
"""

from __future__ import annotations

import numpy as np


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luma grayscale ``0.299 R + 0.587 G + 0.114 B`` of an ``(H, W, 3)`` array."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def clahe(gray: np.ndarray, tiles: int = 8, clip_ratio: float = 0.02) -> np.ndarray:
    """Tile-wise contrast-limited histogram equalisation (64 bins, 0-255 range).

    Each of ``tiles x tiles`` tiles is equalised on its own clipped histogram;
    the redistributed excess keeps flat regions from being amplified into noise.
    Applied once to both images so ZNCC / SSD / LK see stable contrast.
    """
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    out = gray.copy()
    th, tw = max(2, h // tiles), max(2, w // tiles)
    for ti in range(tiles):
        y0 = ti * th
        y1 = h if ti == tiles - 1 else (ti + 1) * th
        for tj in range(tiles):
            x0 = tj * tw
            x1 = w if tj == tiles - 1 else (tj + 1) * tw
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            hist, bins = np.histogram(tile, bins=64, range=(0.0, 256.0))
            cl = max(1, int(tile.size * clip_ratio))
            excess = int(np.maximum(hist - cl, 0).sum())
            hist = np.minimum(hist, cl) + excess // 64
            cdf = hist.cumsum().astype(np.float32)
            cdf = 255.0 * cdf / max(float(cdf[-1]), 1.0)
            out[y0:y1, x0:x1] = np.interp(tile, bins[:-1], cdf)
    return out


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two equally shaped patches, in ``[-1, 1]``."""
    a = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    b = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    return float((a * b).sum() / ((np.linalg.norm(a) + 1e-6) * (np.linalg.norm(b) + 1e-6)))


def ncc_match(
    left: np.ndarray,
    right: np.ndarray,
    u_l: int,
    v_l: int,
    patch: int = 11,
    search_min: int = 15,
    return_score: bool = False,
) -> int | tuple[int, float] | None:
    """Best ZNCC match for left pixel ``(u_l, v_l)`` along row ``v_l`` of ``right``.

    The search is 1-D over ``u_r in [search_min, u_l - 1]`` (positive
    disparity only).  Zero-mean makes the score invariant to brightness
    offsets, the normalisation to contrast; both matter for two cameras with
    different exposure.

    Returns:
        ``u_r`` (or ``(u_r, score)`` with ``return_score``), ``None`` if the
        patch does not fit or the search range is empty.
    """
    h, w = right.shape
    half = patch // 2
    if not (half <= v_l < h - half and half <= u_l < w - half):
        return None
    t = left[v_l - half : v_l + half + 1, u_l - half : u_l + half + 1].astype(np.float64)
    t = t - t.mean()
    n_t = float(np.linalg.norm(t)) + 1e-6
    lo = max(search_min, half)
    hi = min(u_l - 1, w - half - 1)
    if hi <= lo:
        return None
    best_ur, best = lo, -2.0
    for ur in range(lo, hi + 1):
        p = right[v_l - half : v_l + half + 1, ur - half : ur + half + 1].astype(np.float64)
        pm = p - p.mean()
        score = float((t * pm).sum()) / (n_t * (float(np.linalg.norm(pm)) + 1e-6))
        if score > best:
            best, best_ur = score, ur
    return (best_ur, best) if return_score else best_ur


def ssd_subpixel(
    left: np.ndarray, right: np.ndarray, u_l: int, v_l: int, u_r_init: int, patch: int = 11
) -> float | None:
    """Parabolic sub-pixel refinement of an integer disparity on SSD cost.

    Fits a parabola through the SSD at ``u_r_init - 1, u_r_init, u_r_init + 1``
    and returns the vertex ``u_r_init + 0.5 (c0 - c2) / (c0 - 2 c1 + c2)``,
    clamped to +-1 px.
    """
    h, w = right.shape
    half = patch // 2
    if not (half < v_l < h - half - 1 and half < u_l < w - half - 1):
        return None
    if not (half + 1 <= u_r_init < w - half - 1):
        return None
    t = left[v_l - half : v_l + half + 1, u_l - half : u_l + half + 1].astype(np.float64)

    def ssd(u: int) -> float:
        return float(((t - right[v_l - half : v_l + half + 1, u - half : u + half + 1]) ** 2).sum())

    c0, c1, c2 = ssd(u_r_init - 1), ssd(u_r_init), ssd(u_r_init + 1)
    den = c0 - 2.0 * c1 + c2
    if abs(den) < 1e-6:
        return float(u_r_init)
    return float(u_r_init) + max(-1.0, min(1.0, 0.5 * (c0 - c2) / den))


def lk_refine(
    left: np.ndarray,
    right: np.ndarray,
    u_l: int,
    v_l: int,
    u_r0: float,
    v_r0: float,
    patch: int = 9,
    iters: int = 5,
) -> tuple[float, float] | None:
    """Lucas-Kanade 2-D translation refinement of a right-image match.

    Solves the 2x2 structure-tensor system of the left template for the
    translation that best explains the bilinearly sampled right patch, a few
    Gauss-Newton steps.  Returns the refined ``(u_r, v_r)`` or ``None`` if the
    window does not fit; a degenerate (textureless) tensor returns the input.
    """
    h, w = right.shape
    half = patch // 2
    if not (half + 1 <= u_l < w - half - 2 and half + 1 <= v_l < h - half - 2):
        return None
    if not (half + 1 <= u_r0 < w - half - 2 and half + 1 <= v_r0 < h - half - 2):
        return None
    t = left[v_l - half : v_l + half + 1, u_l - half : u_l + half + 1].astype(np.float32)
    ix = 0.5 * (t[:, 2:] - t[:, :-2])[1:-1, :]
    iy = 0.5 * (t[2:, :] - t[:-2, :])[:, 1:-1]
    tc = t[1:-1, 1:-1]
    ixx = float((ix * ix).sum())
    iyy = float((iy * iy).sum())
    ixy = float((ix * iy).sum())
    det = ixx * iyy - ixy * ixy
    if abs(det) < 1e-6:
        return float(u_r0), float(v_r0)
    u, v = float(u_r0), float(v_r0)
    for _ in range(iters):
        ui, vi = int(u), int(v)
        fu, fv = u - ui, v - vi
        if not (half + 1 <= ui < w - half - 2 and half + 1 <= vi < h - half - 2):
            break
        p = (
            (1 - fu) * (1 - fv) * right[vi - half : vi + half + 1, ui - half : ui + half + 1]
            + fu * (1 - fv) * right[vi - half : vi + half + 1, ui - half + 1 : ui + half + 2]
            + (1 - fu) * fv * right[vi - half + 1 : vi + half + 2, ui - half : ui + half + 1]
            + fu * fv * right[vi - half + 1 : vi + half + 2, ui - half + 1 : ui + half + 2]
        )
        it = p[1:-1, 1:-1].astype(np.float32) - tc
        bx = float((ix * it).sum())
        by = float((iy * it).sum())
        du = (iyy * bx - ixy * by) / det
        dv = (ixx * by - ixy * bx) / det
        u -= du
        v -= dv
        if abs(du) + abs(dv) < 1e-3:
            break
    return u, v
