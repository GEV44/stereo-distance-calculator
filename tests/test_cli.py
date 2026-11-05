"""CLI smoke tests (in-process via ``main`` and as ``python -m stereo_distance``)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from stereo_distance import SyntheticRig, __version__, save_cal_pts
from stereo_distance.cli import main

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_module_help_and_version():
    out = subprocess.run([sys.executable, "-m", "stereo_distance", "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0
    for cmd in ("calibrate", "measure", "predict", "info", "ui"):
        assert cmd in out.stdout
    ver = subprocess.run([sys.executable, "-m", "stereo_distance", "--version"],
                         capture_output=True, text=True, cwd=REPO)
    assert ver.returncode == 0 and __version__ in ver.stdout


def test_calibrate_measure_predict_info_cycle(tmp_path, capsys):
    rig = SyntheticRig.pinhole(dx=0.25)
    pts = rig.make_cal_pts(30, z_range=(1.0, 4.0), noise_px=0.2, seed=5)
    pts_path = tmp_path / "pts.json"
    save_cal_pts([tuple(p) for p in pts], pts_path)
    calib = tmp_path / "calib.json"

    rc = main(["calibrate", "--points", str(pts_path), "--out", str(calib),
               "--epochs", "3000", "--quiet"])
    assert rc == 0 and calib.exists()
    out = capsys.readouterr().out
    assert "[SAVE]" in out and "train_MAE" in out

    # measure a fresh synthetic point at 2 m
    left, right = rig.project(np.array([[0.1, -0.05, 2.0]]))
    rc = main(["measure", *map(str, [*left[0], *right[0]]), "--calib", str(calib), "--json"])
    assert rc == 0
    z = json.loads(capsys.readouterr().out)["Z"]
    assert z == pytest.approx(2.0, rel=0.06)

    rc = main(["predict", str(left[0, 0]), str(left[0, 1]), "2.0", "--calib", str(calib)])
    assert rc == 0
    txt = capsys.readouterr().out
    assert txt.startswith("uR = ")
    u_r = float(txt.split()[2])
    assert u_r == pytest.approx(right[0, 0], abs=15)

    rc = main(["info", "--calib", str(calib), "--points", str(pts_path)])
    assert rc == 0
    txt = capsys.readouterr().out
    assert "baseline" in txt and "depth MAE" in txt


def test_measure_reports_bad_pairs(capsys):
    rc = main(["measure", "1000", "900", "1200", "900",
               "--calib", str(REPO / "stereo_calibration.json")])
    assert rc == 1
    assert "bad disparity" in capsys.readouterr().out


def test_measure_shipped_calibration_human_output(capsys):
    rc = main(["measure", "2480", "1086", "2171", "1130",
               "--calib", str(REPO / "stereo_calibration.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("Z = ")
    assert float(out.split()[2]) == pytest.approx(2.02, abs=0.3)  # labelled 2.02 m


def test_calibrate_needs_points(tmp_path, capsys):
    p = tmp_path / "one.json"
    p.write_text("[[100,100,50,100,2.0]]")
    assert main(["calibrate", "--points", str(p), "--out", str(tmp_path / "c.json")]) == 2
    assert "need at least 2" in capsys.readouterr().err


def test_legacy_script_wrapper_imports_without_pygame():
    sys.path.insert(0, str(REPO))
    try:
        import run_math
    finally:
        sys.path.pop(0)
    assert run_math.IMG_W == 2592 and run_math.G0_INIT == pytest.approx(0.45)
    gl, gr, d = run_math.defaults()
    assert gl.shape == (5, 5) and d[0] == 0.10
    z, *_ = run_math.triangulate(2480, 1086, 2171, 1130, gl, gr, d)
    assert z > 0
