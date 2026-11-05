"""Leave-one-out evaluation of the Γ-grid stereo model.

For each calibration point: train on the other N-1 points with the same
settings as run_math.train(), triangulate the held-out point from its
labelled left/right pixels, and record the relative distance error.

This measures the geometric model (lens grids + baseline + triangulation)
on unseen points. It does not include matching error, because the held-out
point uses its hand-labelled right-image pixel rather than the ZNCC match.

Usage:  python eval_loocv.py [cal_pts.json]
"""

import contextlib
import io
import json
import sys

import numpy as np

import run_math as rm


def main(path="cal_pts.json"):
    pts = json.load(open(path))
    rel, rows = [], []
    for i, p in enumerate(pts):
        train = pts[:i] + pts[i + 1:]
        with contextlib.redirect_stdout(io.StringIO()):
            GL, GR, d = rm.train(train, epochs=9000, verbose=10**9)
        z = rm.triangulate(*p[:4], GL, GR, d)[0]
        e = abs(z - p[4]) / p[4]
        rel.append(e)
        rows.append((p[4], z, e))
        print(f"{i:2d}  true={p[4]:.2f} m  pred={z:.3f} m  error={100 * e:.1f}%", flush=True)
    rel = np.array(rel)
    print(f"\nN={len(pts)}  median error {100 * np.median(rel):.1f}%  "
          f"mean error {100 * rel.mean():.1f}%  "
          f"within 10%: {np.sum(rel < 0.10)}/{len(rel)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
