"""ZNCC / SSD / Lucas-Kanade matching on synthetic textures with planted shifts."""

import numpy as np
import pytest

from stereo_distance import (
    clahe,
    lk_refine,
    make_texture,
    ncc_match,
    shift_image,
    ssd_subpixel,
    to_gray,
    zncc,
)

H, W = 120, 400


@pytest.fixture(scope="module")
def pair():
    """Left texture and a right image shifted by exactly 37 px plus noise."""
    rng = np.random.default_rng(0)
    left = make_texture(H, W, seed=1)
    right = shift_image(left, -37.0) + rng.normal(0, 6.0, (H, W)).astype(np.float32)
    return left, right


def test_zncc_bounds_and_invariances():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 255, (11, 11))
    assert zncc(a, a) == pytest.approx(1.0, abs=1e-5)
    assert zncc(a, 1.7 * a + 40.0) == pytest.approx(1.0, abs=1e-5)  # contrast + brightness
    assert zncc(a, -a) == pytest.approx(-1.0, abs=1e-5)
    assert -1.0 <= zncc(a, rng.uniform(0, 255, (11, 11))) <= 1.0


@pytest.mark.parametrize("u_l", [120, 200, 333])
def test_ncc_finds_planted_shift(pair, u_l):
    left, right = pair
    for v in (20, 60, 100):
        assert ncc_match(left, right, u_l, v) == u_l - 37


def test_ncc_survives_brightness_and_contrast_change(pair):
    left, right = pair
    darker = 0.6 * right - 30.0
    u_r, score = ncc_match(left, darker, 250, 60, return_score=True)
    assert u_r == 213
    assert score > 0.8


def test_ncc_returns_none_at_borders_and_empty_range(pair):
    left, right = pair
    assert ncc_match(left, right, 3, 60) is None  # patch does not fit
    assert ncc_match(left, right, 200, 2) is None
    assert ncc_match(left, right, 16, 60, search_min=15) is None  # hi <= lo


def test_ssd_subpixel_recovers_fractional_shift():
    left = make_texture(H, W, seed=4)
    for frac in (0.3, -0.4):
        right = shift_image(left, -(40 + frac))
        est = ssd_subpixel(left, right, 220, 60, 220 - 40)
        assert est == pytest.approx(220 - 40 - frac, abs=0.15)


def test_ssd_subpixel_boundaries():
    left = make_texture(H, W, seed=4)
    assert ssd_subpixel(left, left, 5, 60, 3) is None
    assert ssd_subpixel(left, left, 220, 60, 2) is None
    # identical images: integer peak is exact, parabola adds < 0.1 px bias
    assert ssd_subpixel(left, left, 220, 60, 220) == pytest.approx(220.0, abs=0.1)


def test_lk_refine_converges_to_true_offset():
    left = make_texture(H, W, seed=9, scale=6)
    right = shift_image(left, -30.0, 2.0)  # content moves left 30 px, down 2 px
    # start 1.5 px off in both axes
    out = lk_refine(left, right, 200, 60, 200 - 30 + 1.5, 60 + 2 - 1.5, patch=15, iters=20)
    assert out is not None
    assert out[0] == pytest.approx(170.0, abs=0.2)
    assert out[1] == pytest.approx(62.0, abs=0.2)


def test_lk_refine_degenerate_and_border_cases():
    flat = np.zeros((H, W), dtype=np.float32)
    assert lk_refine(flat, flat, 200, 60, 150.0, 60.0) == (150.0, 60.0)
    tex = make_texture(H, W, seed=1)
    assert lk_refine(tex, tex, 2, 60, 150.0, 60.0) is None
    assert lk_refine(tex, tex, 200, 60, 1.0, 60.0) is None


def test_to_gray_weights_and_passthrough():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    assert to_gray(rgb)[0, 0] == pytest.approx(0.299 * 255)
    rgb[:] = 100
    assert to_gray(rgb)[1, 1] == pytest.approx(100.0)
    g = np.ones((3, 3), dtype=np.float32)
    assert to_gray(g) is g or np.array_equal(to_gray(g), g)


def test_clahe_output_range_and_contrast_boost():
    rng = np.random.default_rng(3)
    low_contrast = (100 + 10 * rng.standard_normal((64, 64))).astype(np.float32)
    out = clahe(low_contrast, tiles=4)
    assert out.shape == low_contrast.shape
    assert 0.0 <= out.min() and out.max() <= 255.0
    assert out.std() > 1.5 * low_contrast.std()
    # monotone within a tile: ordering of pixel intensities is preserved
    tile_in = low_contrast[:16, :16].ravel()
    tile_out = out[:16, :16].ravel()
    order = np.argsort(tile_in)
    assert np.all(np.diff(tile_out[order]) >= -1e-3)
