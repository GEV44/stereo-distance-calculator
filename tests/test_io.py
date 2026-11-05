"""Calibration / point-file persistence and legacy compatibility."""

import json

import numpy as np
import pytest

from stereo_distance import (
    depth_mae,
    load_cal_pts,
    load_calibration,
    load_image_gray,
    save_cal_pts,
    save_calibration,
)


def test_calibration_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    gl = rng.uniform(0.3, 0.6, (5, 5))
    gr = rng.uniform(0.3, 0.6, (5, 5))
    d = np.array([0.27, -0.002, 0.0])
    path = tmp_path / "calib.json"
    save_calibration(gl, gr, d, path)
    gl2, gr2, d2 = load_calibration(path)
    assert np.allclose(gl, gl2) and np.allclose(gr, gr2) and np.allclose(d, d2)
    raw = json.loads(path.read_text())
    assert raw["model"] == "pure_gamma_5x5" and raw["baseline"]["dz"] == 0.0


def test_legacy_4x4_file_is_upsampled(tmp_path, rig):
    legacy = {
        "gamma_L_4x4": np.full((4, 4), 0.5).tolist(),
        "gamma_R_4x4": np.full((4, 4), 0.4).tolist(),
        "baseline": {"dx": 0.2, "dy": 0.0, "dz": 0.0},
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy))
    gl, gr, d = load_calibration(path)
    assert gl.shape == (5, 5) and gr.shape == (5, 5)
    assert gl[2, 2] == rig.g0 and gr[2, 2] == rig.g0  # re-anchored centre
    assert gl[0, 0] == pytest.approx(0.5) and gr[-1, -1] == pytest.approx(0.4)


def test_empty_calibration_file_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("   ")
    with pytest.raises(ValueError):
        load_calibration(path)


def test_cal_pts_roundtrip_and_missing_or_empty(tmp_path):
    pts = [(2480, 1086, 2171, 1130, 2.02), (1760.5, 1542, 1440, 1570, 2.17)]
    path = tmp_path / "pts.json"
    save_cal_pts(pts, path)
    back = load_cal_pts(path)
    assert back == [tuple(float(x) for x in p) for p in pts]
    assert load_cal_pts(tmp_path / "nope.json") == []
    (tmp_path / "blank.json").write_text("")
    assert load_cal_pts(tmp_path / "blank.json") == []


def test_collect_points_records_are_accepted(tmp_path):
    rows = [{"left_pt": [100, 200], "right_pt": [80, 201], "distance": 3.5,
             "left_file": "l.jpg", "right_file": "r.jpg"}]
    path = tmp_path / "training_data.json"
    path.write_text(json.dumps(rows))
    assert load_cal_pts(path) == [(100.0, 200.0, 80.0, 201.0, 3.5)]


def test_shipped_calibration_scores_its_own_points(shipped_calibration_path, shipped_cal_pts_path):
    gl, gr, d = load_calibration(shipped_calibration_path)
    pts = load_cal_pts(shipped_cal_pts_path)
    assert gl.shape == (5, 5) and d[0] == pytest.approx(0.2744, abs=1e-3)
    assert depth_mae(pts, gl, gr, d) < 0.12


def test_load_image_gray_with_pillow(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (64, 32), (255, 0, 0))
    p = tmp_path / "red.png"
    img.save(p)
    g = load_image_gray(p)
    assert g.shape == (32, 64) and g.dtype == np.float32
    assert g[0, 0] == pytest.approx(0.299 * 255, abs=0.5)
    g2 = load_image_gray(p, size=(32, 16))
    assert g2.shape == (16, 32)
