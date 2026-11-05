"""Command-line interface: ``python -m stereo_distance <command>``.

Commands:
    calibrate  fit the gamma grids + baseline to a cal_pts JSON file
    measure    depth of one matched pixel pair from a calibration file
    predict    right-image pixel of a left pixel at a given depth
    info       summarise a calibration file (and its fit on a point file)
    ui         launch the interactive Pygame measurement tool (needs ``[ui]`` extra)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .calibration import TrainConfig, depth_mae, train
from .config import DEFAULT_RIG, RigConfig
from .io import CAL_PTS_PATH, CALIB_PATH, load_cal_pts, load_calibration, save_calibration
from .lens import predict_right
from .triangulation import triangulate_pair


def _add_rig_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--img-w", type=int, default=DEFAULT_RIG.img_w, help="sensor width (px)")
    p.add_argument("--img-h", type=int, default=DEFAULT_RIG.img_h, help="sensor height (px)")
    p.add_argument("--f-init", type=float, default=DEFAULT_RIG.f_init, help="nominal focal (px)")


def _rig_from(args: argparse.Namespace) -> RigConfig:
    return RigConfig(img_w=args.img_w, img_h=args.img_h, f_init=args.f_init)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stereo_distance",
        description="Pure-NumPy stereo distance: 5x5 gamma lens grid, analytic-gradient "
        "calibration, OLS triangulation.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="train grids + baseline from labelled point pairs")
    c.add_argument("--points", default=CAL_PTS_PATH, help="JSON of [uL,vL,uR,vR,Z] rows")
    c.add_argument("--out", default=CALIB_PATH, help="calibration JSON to write")
    c.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    c.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    c.add_argument("--huber", type=float, default=None, help="Huber delta on e_x, e_y (off)")
    c.add_argument("--quiet", action="store_true")
    _add_rig_args(c)

    m = sub.add_parser("measure", help="depth (m) of one matched pixel pair")
    m.add_argument("uL", type=float)
    m.add_argument("vL", type=float)
    m.add_argument("uR", type=float)
    m.add_argument("vR", type=float)
    m.add_argument("--calib", default=CALIB_PATH)
    m.add_argument("--json", action="store_true", help="machine-readable output")
    _add_rig_args(m)

    r = sub.add_parser("predict", help="right pixel of a left pixel at depth Z")
    r.add_argument("uL", type=float)
    r.add_argument("vL", type=float)
    r.add_argument("Z", type=float, help="depth in metres")
    r.add_argument("--calib", default=CALIB_PATH)
    _add_rig_args(r)

    i = sub.add_parser("info", help="summarise a calibration file")
    i.add_argument("--calib", default=CALIB_PATH)
    i.add_argument("--points", default=None, help="optional cal_pts JSON to score")
    _add_rig_args(i)

    u = sub.add_parser("ui", help="interactive Pygame tool (pip install stereo-distance[ui])")
    u.add_argument("left", help="left image path")
    u.add_argument("right", help="right image path")
    u.add_argument("--calib", default=CALIB_PATH)
    u.add_argument("--points", default=CAL_PTS_PATH)
    return p


def cmd_calibrate(args: argparse.Namespace) -> int:
    rig = _rig_from(args)
    pts = load_cal_pts(args.points)
    if len(pts) < 2:
        print(f"error: {args.points} has {len(pts)} points; need at least 2", file=sys.stderr)
        return 2
    cfg = TrainConfig(epochs=args.epochs, optimizer=args.optimizer, huber_delta=args.huber)
    res = train(pts, cfg, rig, log=None if args.quiet else print)
    save_calibration(res.gamma_l, res.gamma_r, res.d, args.out)
    mae = depth_mae(pts, res.gamma_l, res.gamma_r, res.d, rig)
    print(
        f"[SAVE] {args.out}  dx={res.d[0]:.4f} m  dy={res.d[1]:.4f} m  "
        f"best_loss={res.best_loss:.6f}  train_MAE={mae:.3f} m  (N={len(pts)})"
    )
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    rig = _rig_from(args)
    gamma_l, gamma_r, d = load_calibration(args.calib, rig)
    t = triangulate_pair(args.uL, args.vL, args.uR, args.vR, gamma_l, gamma_r, d, rig)
    if args.json:
        print(json.dumps({"Z": None if math.isnan(t.z) else t.z, "Z_right": None
                          if math.isnan(t.z_right) else t.z_right, "error": t.error}))
    elif t.ok:
        print(f"Z = {t.z:.4f} m  (Z_right = {t.z_right:.4f} m, residual = {t.residual:.2e})")
    else:
        print(f"triangulate: {t.error}")
    return 0 if t.ok else 1


def cmd_predict(args: argparse.Namespace) -> int:
    rig = _rig_from(args)
    gamma_l, gamma_r, d = load_calibration(args.calib, rig)
    pr = predict_right(args.uL, args.vL, args.Z, gamma_l, gamma_r, d, rig)
    if pr is None:
        print("point is behind the right camera")
        return 1
    print(f"uR = {pr[0]:.2f}  vR = {pr[1]:.2f}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    rig = _rig_from(args)
    gamma_l, gamma_r, d = load_calibration(args.calib, rig)
    print(f"calibration: {args.calib}")
    print(f"  grid       : {gamma_l.shape[0]}x{gamma_l.shape[1]} per camera, centre g0={rig.g0:.4f}")
    print(f"  baseline   : dx={d[0]:.4f} m  dy={d[1]:+.4f} m")
    for name, g in (("GammaL", gamma_l), ("GammaR", gamma_r)):
        f_eq = rig.cx / g
        print(
            f"  {name}     : min={g.min():.4f} max={g.max():.4f}  "
            f"equivalent focal {f_eq.min():.0f}-{f_eq.max():.0f} px"
        )
    if args.points:
        pts = load_cal_pts(args.points)
        mae = depth_mae(pts, gamma_l, gamma_r, d, rig)
        print(f"  fit        : N={len(pts)} points, depth MAE={mae:.3f} m")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui import run
    except ImportError as e:  # pragma: no cover - needs pygame
        print(f"error: the UI needs pygame ({e}); pip install 'stereo-distance[ui]'", file=sys.stderr)
        return 2
    run(args.left, args.right, calib_path=args.calib, cal_pts_path=args.points)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    handler = {
        "calibrate": cmd_calibrate,
        "measure": cmd_measure,
        "predict": cmd_predict,
        "info": cmd_info,
        "ui": cmd_ui,
    }[args.cmd]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
