"""
Audit of the distance measurement — run before trusting a calibration.

    python scripts/audit.py [--calib stereo_calibration.json] [--points cal_pts.json]

Checks
  A  projection → triangulation round trip with the given calibration
  B  sensitivity of depths to the assumed nominal focal length f₀
  C  recovery of a known rig (distortion-free, f₀ correct): baseline and angles
  D  agreement of the two ensemble members on the calibration points
  E  depth range covered by the calibration (outside it measurements extrapolate)
  F  depth vs straight-line distance at the image corner
  H  lateral scale (X, Y, lengths): calibrated from known lengths or assumed from f₀
  G  dependency scan: no library vision/calibration algorithms, no pretrained models
Exit status 1 if a hard check (A, C, G) fails.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from stereo_gamma import StereoModel, load_points, train  # noqa: E402
from stereo_gamma.config import Sensor, TrainConfig  # noqa: E402
from stereo_gamma.evaluation import metrics  # noqa: E402
from stereo_gamma.synthetic import Camera, SyntheticRig  # noqa: E402

FORBIDDEN = re.compile(r"\b(scipy|sklearn|torch|tensorflow|onnx|cv2\.(stereo|calibrate|triangulate|undistort|"
                       r"findChessboard|solvePnP|StereoBM|StereoSGBM|matchTemplate|findFundamental|findEssential))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(ROOT, "stereo_calibration.json"))
    ap.add_argument("--points", default=os.path.join(ROOT, "cal_pts.json"))
    a = ap.parse_args()
    m, pts = StereoModel.load(a.calib), load_points(a.points)
    s, ok = m.sensor, True

    rng = np.random.default_rng(0)
    errs = []
    for _ in range(2000):
        u, v, Z = rng.uniform(100, s.width - 100), rng.uniform(100, s.height - 100), rng.uniform(0.3, 30)
        uR, vR, valid = m.project_left_point_to_right(u, v, Z)
        if valid[0] and 0 <= uR[0] < s.width and 0 <= vR[0] < s.height:
            errs.append(m.triangulate_gamma([u], [v], uR, vR)[0][0] / Z - 1)
    e = float(np.max(np.abs(errs)))
    ok &= e < 1e-6
    print(f"A  round trip: {len(errs)} points, max relative error {e:.1e}  {'OK' if e < 1e-6 else 'FAIL'}")

    base = None
    for f0 in (2500.0, s.f_init, 3300.0):
        mm, _ = train(pts, TrainConfig(), sensor=Sensor(s.width, s.height, s.grid_rows, s.grid_cols, f0), log=None)
        z = mm.triangulate(*pts[:, :4].T)[0]
        base = z if base is None else base
        print(f"B  f0 = {f0:6.0f} px: in-sample MAPE {metrics(z, pts[:, 4])['mape_pct']:.2f} %, "
              f"max depth change vs f0 = 2500: {100 * np.nanmax(np.abs(z / base - 1)):.2f} %")

    rig = SyntheticRig(Camera(s.f_init, s.cx, s.cy), Camera(s.f_init, s.cx, s.cy), np.array([0.25, 0.003, 0.0]),
                       np.radians([0.3, 1.2, -0.8]), s)
    tr, _ = rig.sample(150, np.random.default_rng(1), click_sigma=0.0)
    mm, _ = train(tr, TrainConfig(ensemble=False), sensor=s, log=None)
    good = (np.allclose(mm.baseline[:2], [0.25, 0.003], atol=2e-5)
            and np.allclose(np.degrees(mm.rotation), [0.3, 1.2, -0.8], atol=2e-3))
    ok &= good
    print(f"C  known rig recovered: d = {np.round(mm.baseline[:2], 5)}, ω = {np.round(np.degrees(mm.rotation), 4)}° "
          f"{'OK' if good else 'FAIL'}")

    if m.companion is not None:
        q = pts[:, :4].T
        d = np.abs(m.triangulate_gamma(*q)[0] / m.companion_model().triangulate(*q) - 1) * 100
        print(f"D  ensemble members differ on the calibration points: median {np.median(d):.2f} %, max {d.max():.2f} %")
    else:
        print("D  no Brown–Conrady companion in this calibration (fewer than 25 points)")

    lo, hi = m.meta.get("depth_range_m", [pts[:, 4].min(), pts[:, 4].max()])
    print(f"E  calibrated depths {lo:.2f}–{hi:.2f} m; the app flags depths outside {0.8 * lo:.2f}–{1.25 * hi:.2f} m")

    r = float(np.linalg.norm(m.left_rays([0.0], [0.0])[0]))
    print(f"F  straight-line / depth at the image corner = {r:.3f} (reported separately in the app)")

    k = m.lateral_scale
    n = len(m.meta.get("length_refs", []))
    print(f"H  lateral scale k = {k:.4f} from {n} known length(s)" if n else
          f"H  lateral scale NOT calibrated: X, Y and lengths assume f0 = {s.f_init:.0f} px (depth unaffected) — "
          "measure a known length and press L in the app")

    hits = []
    for folder in ("stereo_gamma", "scripts"):
        for f in sorted(os.listdir(os.path.join(ROOT, folder))):
            if f.endswith(".py") and f != "audit.py":
                for i, line in enumerate(open(os.path.join(ROOT, folder, f), encoding="utf-8"), 1):
                    if FORBIDDEN.search(line) and not line.lstrip().startswith("#"):
                        hits.append(f"{folder}/{f}:{i}: {line.strip()}")
    ok &= not hits
    print(f"G  library vision / ML algorithms used: {'none — OK' if not hits else 'FAIL'}")
    for h in hits:
        print("     " + h)
    print("AUDIT PASSED" if ok else "AUDIT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
