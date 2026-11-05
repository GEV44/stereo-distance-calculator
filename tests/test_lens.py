"""Lens model: bilinear grid, ray construction, Laplacian, inversion, resizing."""

import numpy as np
import pytest

from stereo_distance import (
    RigConfig,
    bilinear,
    bilinear_batch,
    build_ray,
    build_rays_batch,
    default_params,
    laplacian_grad,
    predict_right,
    ray_to_pixel,
    resize_grid,
)


def test_rig_config_derived_values(rig):
    assert (rig.cx, rig.cy) == (1296.0, 972.0)
    assert (rig.cell_w, rig.cell_h) == (648.0, 486.0)
    assert rig.g0 == pytest.approx(1296.0 / 2880.0)
    assert rig.centre == (2, 2)


def test_bilinear_weights_sum_to_one_and_hit_nodes(rig):
    gamma = np.arange(25, dtype=float).reshape(5, 5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        u, v = rng.uniform(0, rig.img_w), rng.uniform(0, rig.img_h)
        g, w, idx = bilinear(u, v, gamma)
        assert sum(w) == pytest.approx(1.0)
        assert all(0.0 <= wk <= 1.0 for wk in w)
    # exactly on a node -> the node value
    for iy in range(5):
        for ix in range(5):
            g, _, _ = bilinear(ix * rig.cell_w, iy * rig.cell_h, gamma)
            assert g == pytest.approx(gamma[iy, ix])


def test_bilinear_is_linear_inside_a_cell(rig):
    gamma = np.zeros((5, 5))
    gamma[0, 1] = 1.0  # node at (u=cell_w, v=0)
    g_half, _, _ = bilinear(rig.cell_w / 2, 0.0, gamma)
    g_quarter, _, _ = bilinear(rig.cell_w / 4, 0.0, gamma)
    assert g_half == pytest.approx(0.5)
    assert g_quarter == pytest.approx(0.25)


def test_bilinear_extrapolates_outside_image_from_border_cell(rig):
    gamma = np.full((5, 5), 0.45)
    g_in, _, idx_in = bilinear(rig.img_w - 1, rig.img_h - 1, gamma)
    g_out, _, idx_out = bilinear(rig.img_w + 500, rig.img_h + 500, gamma)
    assert g_in == pytest.approx(0.45)
    assert g_out == pytest.approx(0.45)
    assert idx_out == idx_in  # clamped to the last cell


def test_bilinear_batch_matches_scalar(rig):
    rng = np.random.default_rng(3)
    gamma = rng.uniform(0.3, 0.6, (5, 5))
    u = rng.uniform(-10, rig.img_w + 10, 200)
    v = rng.uniform(-10, rig.img_h + 10, 200)
    g_b, w_b, rows, cols = bilinear_batch(u, v, gamma)
    for i in range(200):
        g, w, idx = bilinear(u[i], v[i], gamma)
        assert g_b[i] == pytest.approx(g)
        assert np.allclose(w_b[i], w)
        assert [tuple(x) for x in zip(rows[i], cols[i])] == list(idx)
    rays, *_ = build_rays_batch(u, v, gamma)
    for i in range(0, 200, 17):
        assert np.allclose(rays[i], build_ray(u[i], v[i], gamma)[0])


def test_uniform_grid_is_a_pinhole_camera(rig):
    gamma_l, _, _ = default_params()
    f = rig.f_init
    for u, v in [(0, 0), (rig.cx, rig.cy), (2000, 300), (rig.img_w - 1, rig.img_h - 1)]:
        ray, info = build_ray(u, v, gamma_l)
        assert ray[2] == 1.0
        assert ray[0] == pytest.approx((u - rig.cx) / f)
        # v is normalised by cy but g = cx / f, so the model's implied
        # vertical focal is f * cy / cx (the author's convention)
        assert ray[1] == pytest.approx((v - rig.cy) / rig.cy * rig.g0)
        assert info["g"] == pytest.approx(rig.g0)
    centre_ray, _ = build_ray(rig.cx, rig.cy, gamma_l)
    assert np.allclose(centre_ray, [0.0, 0.0, 1.0])


def test_laplacian_is_zero_for_uniform_and_linear_grids():
    assert np.allclose(laplacian_grad(np.full((5, 5), 0.45)), 0.0, atol=1e-12)
    yy, xx = np.mgrid[0:5, 0:5]
    ramp = 0.4 + 0.01 * xx + 0.02 * yy
    assert np.allclose(laplacian_grad(ramp), 0.0)


def test_laplacian_penalises_a_bump_and_leaves_borders_alone():
    g = np.full((5, 5), 0.45)
    g[2, 2] += 0.1
    lap = laplacian_grad(g)
    assert lap[2, 2] == pytest.approx(0.4)  # 4 * 0.1
    assert lap[1, 2] == pytest.approx(-0.1)
    assert lap[2, 3] == pytest.approx(-0.1)
    assert np.all(lap[0, :] == 0) and np.all(lap[-1, :] == 0)
    assert np.all(lap[:, 0] == 0) and np.all(lap[:, -1] == 0)


def test_ray_to_pixel_inverts_build_ray_on_distorted_grid(distorted):
    rng = np.random.default_rng(7)
    for _ in range(40):
        u, v = rng.uniform(5, distorted.rig.img_w - 5), rng.uniform(5, distorted.rig.img_h - 5)
        ray, _ = build_ray(u, v, distorted.gamma_l)
        px = ray_to_pixel(ray, distorted.gamma_l, iters=8)
        assert px is not None
        assert px[0] == pytest.approx(u, abs=0.05)
        assert px[1] == pytest.approx(v, abs=0.05)


def test_ray_to_pixel_rejects_points_behind_camera(pinhole):
    assert ray_to_pixel(np.array([0.1, 0.1, -1.0]), pinhole.gamma_l) is None


def test_predict_right_matches_projection(distorted, scene_points):
    left, right = distorted.project(scene_points)
    ok = distorted.in_frame(left, right)
    assert ok.sum() >= 20
    for i in np.flatnonzero(ok)[:20]:
        pr = predict_right(left[i, 0], left[i, 1], scene_points[i, 2],
                           distorted.gamma_l, distorted.gamma_r, distorted.d)
        assert pr is not None
        assert pr[0] == pytest.approx(right[i, 0], abs=0.1)
        assert pr[1] == pytest.approx(right[i, 1], abs=0.1)


def test_predict_right_disparity_shrinks_with_depth(pinhole):
    u, v = 1296.0, 972.0
    disps = []
    for z in (1.0, 2.0, 4.0, 8.0):
        pr = predict_right(u, v, z, pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
        disps.append(u - pr[0])
    assert all(a > b > 0 for a, b in zip(disps, disps[1:]))
    # pinhole: disparity = f * dx / Z
    assert disps[0] == pytest.approx(2880.0 * 0.25 / 1.0, rel=1e-6)


def test_resize_grid_keeps_corners_and_interpolates():
    g4 = np.arange(16, dtype=float).reshape(4, 4)
    g5 = resize_grid(g4, (5, 5))
    assert g5.shape == (5, 5)
    assert g5[0, 0] == g4[0, 0] and g5[-1, -1] == g4[-1, -1]
    assert g5[0, -1] == g4[0, -1] and g5[-1, 0] == g4[-1, 0]
    assert g5[2, 2] == pytest.approx(g4.mean())  # centre of a bilinear ramp
    assert np.allclose(resize_grid(g4, (4, 4)), g4)


def test_custom_rig_changes_geometry():
    small = RigConfig(img_w=640, img_h=480, f_init=500.0)
    gamma_l, _, _ = default_params(small)
    ray, _ = build_ray(640.0, 240.0, gamma_l, small)
    assert ray[0] == pytest.approx((640 - 320) / 500.0)
    assert ray[1] == pytest.approx(0.0)
