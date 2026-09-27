"""
Command-line interface.

    python -m stereo_gamma measure  LEFT RIGHT      interactive measuring / calibration
    python -m stereo_gamma train    [--cv]          fit a calibration from points
    python -m stereo_gamma evaluate                 accuracy report + outlier check
    python -m stereo_gamma demo                     synthetic scene, no camera needed
    python -m stereo_gamma collect                  label points in left/ right/ folders
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from . import __version__
from .config import TrainConfig


def _print_metrics(title, m):
    print(f"{title}: MAE {m['mae_m']:.3f} m | RMSE {m['rmse_m']:.3f} m | MAPE {m['mape_pct']:.2f}% | "
          f"median APE {m['median_ape_pct']:.2f}% | δ<1.05 {m['delta_1.05_pct']:.0f}% | "
          f"worst {m['max_ape_pct']:.1f}%")


def cmd_train(a):
    from .data import load_points
    from .evaluation import loocv, metrics
    from .training import train

    pts = load_points(a.points)
    if len(pts) < 3:
        sys.exit(f"{a.points}: need at least 3 points, found {len(pts)}")
    cfg = TrainConfig(epochs=a.epochs, log_every=0 if a.quiet else max(1, a.epochs // 10))
    from .config import Sensor

    sensor = Sensor() if a.focal_px is None else Sensor(f_init=a.focal_px)
    print(f"[train] {len(pts)} points from {a.points}  (f0 = {sensor.f_init:.0f} px)")
    model, _ = train(pts, cfg, sensor)
    if os.path.exists(a.out):  # keep known-length references from the previous calibration
        try:
            from .model import StereoModel

            refs = StereoModel.load(a.out).meta.get("length_refs", [])
            if refs:
                print(f"[train] lateral scale k = {model.fit_lateral_scale(refs):.4f} from {len(refs)} known length(s)")
        except (ValueError, KeyError):
            pass
    d, w = model.baseline, np.degrees(model.rotation)
    print(f"[train] dx={d[0]:.4f}  dy={d[1]:+.4f}  pitch={w[0]:+.3f}°  yaw={w[1]:+.3f}°  roll={w[2]:+.3f}°")
    _print_metrics("[train] in-sample", metrics(model.triangulate(*pts[:, :4].T)[0], pts[:, 4]))
    if a.cv:
        print(f"[train] nested leave-one-out cross-validation ({len(pts)} full trainings, ~2 min) …")
        z_cv = loocv(pts, cfg, sensor)
        m = metrics(z_cv, pts[:, 4])
        _print_metrics("[train] LOOCV    ", m)
        model.meta["loocv"] = m
        model.meta["rel_model_error"] = math.sqrt(float(np.nanmean((z_cv / pts[:, 4] - 1) ** 2)))
        model.meta["rel_model_error_source"] = "loocv"
    model.save(a.out)
    print(f"[train] saved {a.out}")


def cmd_evaluate(a):
    from .data import load_points
    from .evaluation import metrics
    from .model import StereoModel

    model, pts = StereoModel.load(a.calib), load_points(a.points)
    Z = model.triangulate(*pts[:, :4].T)[0]
    rel = Z / pts[:, 4] - 1
    _print_metrics(f"{a.calib} on {a.points} (in-sample)", metrics(Z, pts[:, 4]))
    mad = 1.4826 * np.nanmedian(np.abs(rel - np.nanmedian(rel)))
    print(f"\n  #   uL    vL    uR    vR    Z_true  Z_pred   error   {'flag' if mad > 0 else ''}")
    for i, (p, z, r) in enumerate(zip(pts, Z, rel)):
        flag = "  ← outlier (>3σ robust)" if abs(r) > 3 * mad else ""
        print(f"{i:3d} {p[0]:5.0f} {p[1]:5.0f} {p[2]:5.0f} {p[3]:5.0f}  {p[4]:6.2f}  {z:6.2f}  {100 * r:+6.1f}%{flag}")


def cmd_measure(a):
    from .app import run

    run(a.left, a.right, a.calib, a.points)


def cmd_demo(a):
    from .data import save_points
    from .synthetic import SyntheticRig, render, scene_points
    from .training import train

    os.makedirs(a.dir, exist_ok=True)
    lp, rp = os.path.join(a.dir, "demo_left.png"), os.path.join(a.dir, "demo_right.png")
    pp, cp = os.path.join(a.dir, "demo_cal_pts.json"), os.path.join(a.dir, "demo_calibration.json")
    rig = SyntheticRig()
    if not (os.path.exists(lp) and os.path.exists(rp)):
        import matplotlib.pyplot as plt

        print("[demo] ray-tracing the synthetic stereo pair (≈15 s) …")
        for which, path in (("left", lp), ("right", rp)):
            plt.imsave(path, render(rig, which)[0], cmap="gray", vmin=0, vmax=255)
    pts = scene_points(rig, 40, np.random.default_rng(0))
    save_points(pts, pp)
    model, _ = train(pts, TrainConfig())
    model.save(cp)
    print("[demo] ground truth: box face 2.0 m, panel 3.5 m, back wall 6.0 m, floor slopes 0.5–6 m")
    print(f"[demo] files in {a.dir}/")
    if not a.no_gui:
        from .app import run

        run(lp, rp, cp, pp)


def cmd_collect(a):
    from .collect import main

    main(a.left_dir, a.right_dir, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="stereo_gamma", description="Γ-grid stereo distance measurement")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("measure", help="interactive measuring / calibration UI")
    s.add_argument("left")
    s.add_argument("right")
    s.add_argument("--calib", default="stereo_calibration.json")
    s.add_argument("--points", default="cal_pts.json")
    s.set_defaults(func=cmd_measure)

    s = sub.add_parser("train", help="fit a calibration from labelled points")
    s.add_argument("--points", default="cal_pts.json")
    s.add_argument("--out", default="stereo_calibration.json")
    s.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    s.add_argument("--cv", action="store_true", help="also run leave-one-out cross-validation")
    s.add_argument("--focal-px", type=float, default=None,
                   help="true focal length in pixels (focal mm / pixel pitch mm) if known — sets the lateral "
                        "scale (X, Y, lengths); depth does not depend on it")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="accuracy of a calibration on labelled points")
    s.add_argument("--calib", default="stereo_calibration.json")
    s.add_argument("--points", default="cal_pts.json")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("demo", help="synthetic scene demo (no camera required)")
    s.add_argument("--dir", default="demo_output")
    s.add_argument("--no-gui", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("collect", help="label corresponding points in image folders")
    s.add_argument("--left-dir", default="left")
    s.add_argument("--right-dir", default="right")
    s.add_argument("--out", default="training_data.json")
    s.set_defaults(func=cmd_collect)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
