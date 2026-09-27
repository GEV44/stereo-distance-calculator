import os

import numpy as np

from stereo_gamma.cli import main
from stereo_gamma.model import StereoModel


def test_train_and_evaluate(tmp_path, root, capsys):
    out = tmp_path / "c.json"
    main(["train", "--points", os.path.join(root, "cal_pts.json"), "--out", str(out),
          "--epochs", "300", "--quiet"])
    m = StereoModel.load(out)
    assert m.is_trained and m.meta["n_points"] == 27
    main(["evaluate", "--calib", str(out), "--points", os.path.join(root, "cal_pts.json")])
    assert "MAPE" in capsys.readouterr().out


def test_shipped_calibration_quality(root, real_points):
    """Regression guard for the calibration shipped in the repository."""
    m = StereoModel.load(os.path.join(root, "stereo_calibration.json"))
    Z = m.triangulate(*real_points[:, :4].T)[0]
    assert np.isfinite(Z).all()
    assert np.mean(np.abs(Z / real_points[:, 4] - 1)) < 0.035
    assert m.meta["rel_model_error_source"] == "loocv"
    assert m.meta["loocv"]["mape_pct"] < 6.0


def test_evaluate_cv_prints_out_of_sample_table(tmp_path, real_points, capsys):
    from stereo_gamma.data import save_points

    p = tmp_path / "pts.json"
    save_points(real_points[:12], p)
    main(["evaluate", "--cv", "--points", str(p)])
    out = capsys.readouterr().out
    assert "out-of-sample" in out and "within 2σ" in out and out.count("%") > 12
