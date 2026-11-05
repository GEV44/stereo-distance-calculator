"""OLS triangulation on synthetic rigs with known ground truth."""

import math

import numpy as np
import pytest

from stereo_distance import triangulate, triangulate_batch, triangulate_pair


def test_pinhole_roundtrip_is_exact(pinhole, scene_points):
    left, right = pinhole.project(scene_points)
    ok = pinhole.in_frame(left, right)
    assert ok.sum() >= 30
    for i in np.flatnonzero(ok):
        t = triangulate_pair(*left[i], *right[i], pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
        assert t.ok
        assert t.z == pytest.approx(scene_points[i, 2], rel=1e-9)
        assert t.z_right == pytest.approx(scene_points[i, 2], rel=1e-9)  # d has no z component
        assert t.residual < 1e-9
        assert np.allclose(t.point, scene_points[i], atol=1e-9)


def test_distorted_roundtrip_within_tolerance(distorted, scene_points):
    left, right = distorted.project(scene_points)
    ok = distorted.in_frame(left, right)
    z = triangulate_batch(left[ok, 0], left[ok, 1], right[ok, 0], right[ok, 1],
                          distorted.gamma_l, distorted.gamma_r, distorted.d)
    rel = np.abs(z - scene_points[ok, 2]) / scene_points[ok, 2]
    assert np.all(np.isfinite(z))
    assert rel.max() < 1e-4  # only the fixed-point projection error remains


def test_batch_matches_scalar(distorted, scene_points):
    left, right = distorted.project(scene_points)
    ok = distorted.in_frame(left, right)
    z_b = triangulate_batch(left[ok, 0], left[ok, 1], right[ok, 0], right[ok, 1],
                            distorted.gamma_l, distorted.gamma_r, distorted.d)
    for i, k in enumerate(np.flatnonzero(ok)):
        z, *_ = triangulate(*left[k], *right[k], distorted.gamma_l, distorted.gamma_r, distorted.d)
        assert z_b[i] == pytest.approx(z, rel=1e-12)


def test_vertical_baseline_handled(pinhole):
    rig = pinhole.rig
    d = np.array([0.25, 0.03, 0.0])
    p = np.array([[0.2, -0.1, 3.0]])
    gl, gr = pinhole.gamma_l, pinhole.gamma_r
    from stereo_distance import SyntheticRig

    left, right = SyntheticRig(gl, gr, d, rig).project(p)
    assert right[0, 1] != pytest.approx(left[0, 1])  # dy shows up as a row offset
    t = triangulate_pair(*left[0], *right[0], gl, gr, d)
    assert t.z == pytest.approx(3.0, rel=1e-9)


def test_bad_disparity_returns_nan_with_reason(pinhole):
    t = triangulate_pair(1000, 900, 1200, 900, pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
    assert not t.ok and math.isnan(t.z)
    assert "bad disparity" in t.error
    z, rl, rr, err = triangulate(1000, 900, 1000, 900, pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
    assert math.isnan(z) and rl is None and "bad disparity" in err


def test_parallel_rays_detected():
    # Left grid gamma=1 and right grid gamma=2 at the same normalised u -> same ray x
    gl = np.full((5, 5), 0.45)
    gr = np.full((5, 5), 0.90)
    d = np.array([0.25, 0.0, 0.0])
    # choose uR so that (uR-cx)/cx * 0.9 == (uL-cx)/cx * 0.45  with uL > uR
    u_l = 1296 + 400.0
    u_r = 1296 + 200.0
    t = triangulate_pair(u_l, 972, u_r, 972, gl, gr, d)
    assert not t.ok and "parallel" in t.error


def test_negative_depth_rejected(pinhole):
    # Positive pixel disparity but a baseline pointing the other way -> Z < 0
    d = -pinhole.d
    t = triangulate_pair(1500, 972, 1300, 972, pinhole.gamma_l, pinhole.gamma_r, d)
    assert not t.ok and "negative depth" in t.error


def test_noise_error_grows_with_distance(pinhole):
    """Depth error of a stereo rig grows roughly like Z^2 for fixed pixel noise."""
    rng = np.random.default_rng(0)
    errs = []
    for z in (1.0, 2.0, 4.0):
        pts = np.column_stack([rng.uniform(-0.2, 0.2, 300) * z, rng.uniform(-0.2, 0.2, 300) * z,
                               np.full(300, z)])
        left, right = pinhole.project(pts)
        right[:, 0] += rng.normal(0, 0.5, 300)
        zz = triangulate_batch(left[:, 0], left[:, 1], right[:, 0], right[:, 1],
                               pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
        errs.append(np.nanmean(np.abs(zz - z)))
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] / errs[0] > 8  # ~ (4/1)^2 = 16 in theory, well above linear


def test_gd_steps_do_not_change_answer(pinhole):
    args = (1500, 900, 1300, 900, pinhole.gamma_l, pinhole.gamma_r, pinhole.d)
    assert triangulate_pair(*args, gd_steps=0).z == triangulate_pair(*args, gd_steps=50).z
