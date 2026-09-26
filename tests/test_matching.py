import numpy as np
import pytest

from stereo_gamma.matching import (
    clahe,
    epipolar_match,
    lk_refine,
    match_along_curve,
    parabolic_peak,
    zncc_scores,
)


def _texture(rng, h, w):
    img = rng.uniform(0, 255, (h, w))
    k = np.ones(5) / 5
    for ax in (0, 1):
        img = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), ax, img)
    return img


def _shift(img, sx, sy):
    """R(u, v) = img(u − sx, v − sy)  (bilinear), cropped by 60 px."""
    h, w = img.shape
    yy, xx = np.mgrid[60:h - 60, 60:w - 60].astype(float)
    x, y = xx - sx, yy - sy
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def test_parabolic_peak_is_exact_for_parabolas():
    for true in (-0.4, 0.0, 0.25):
        s = [-(k - true) ** 2 for k in (-1, 0, 1)]
        assert parabolic_peak(*s) == pytest.approx(true)


def test_zncc_invariant_to_brightness_and_contrast(rng):
    L = _texture(rng, 100, 100)
    s1, _ = zncc_scores(L, L, 50, 50, [50], [50], 15)
    s2, _ = zncc_scores(L, 3.0 * L + 40, 50, 50, [50], [50], 15)
    assert s1[0] == pytest.approx(1.0) and s2[0] == pytest.approx(1.0)


def test_subpixel_match_along_curve_and_lk(rng):
    base = _texture(rng, 520, 720)
    L, R = base[60:-60, 60:-60], _shift(base, -37.3, 4.6)
    uL, vL = 300, 200
    cu = np.arange(150.0, 300.0)
    m = match_along_curve(L * 1.3 + 20, R, uL, vL, cu, np.full_like(cu, vL + 4.6))
    assert m.u == pytest.approx(uL - 37.3, abs=0.35) and m.score > 0.8
    u, v = lk_refine(L, R, uL, vL, m.u, round(m.v))
    assert u == pytest.approx(uL - 37.3, abs=0.1) and v == pytest.approx(vL + 4.6, abs=0.1)


def test_lk_rejects_textureless_patch():
    flat = np.full((100, 100), 128.0)
    assert lk_refine(flat, flat, 50, 50, 50.0, 50.0) is None


def test_clahe_range_and_no_tile_seams(rng):
    g = np.clip(rng.normal(60, 5, (400, 400)) + np.linspace(0, 120, 400)[None, :], 0, 255)
    out = clahe(g, tiles=4)
    assert out.min() >= 0 and out.max() <= 255.01
    # bilinear LUT blending: no jump at tile borders larger than inside tiles
    col = np.abs(np.diff(out.mean(0)))
    assert col[99] < 5 * np.median(col) + 1


def test_epipolar_match_on_ray_traced_scene(small_scene, rng):
    """End-to-end: click → epipolar ZNCC + LR check → depth vs ray-traced ground truth."""
    rig, left, right, depth, model = small_scene
    Lg, Rg = clahe(left), clahe(right)
    h, w = left.shape
    errs, n_rel = [], 0
    for _ in range(40):
        u, v = int(rng.uniform(40, w - 40)), int(rng.uniform(40, h - 40))
        if not np.isfinite(depth[v, u]) or depth[v, u] > 5.9:
            continue
        m = epipolar_match(model, Lg, Rg, u, v, patch=15)
        if m is None or not m.reliable:
            continue
        n_rel += 1
        errs.append(abs(model.depth(u, v, m.u, m.v) / depth[v, u] - 1))
    assert n_rel >= 10
    assert np.median(errs) < 0.02 and max(errs) < 0.08
