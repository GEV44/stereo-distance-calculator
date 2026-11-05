"""Persistence: calibration JSON, calibration-point cache, image loading."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .calibration import CalPoint
from .config import DEFAULT_RIG, RigConfig
from .lens import resize_grid

CALIB_PATH = "stereo_calibration.json"
CAL_PTS_PATH = "cal_pts.json"

PathLike = str | os.PathLike


def save_calibration(
    gamma_l: np.ndarray, gamma_r: np.ndarray, d: np.ndarray, path: PathLike = CALIB_PATH
) -> None:
    """Write the grids and baseline as JSON.

    The grid keys are ``gamma_L_4x4`` / ``gamma_R_4x4`` for compatibility with
    files written by the first release; the arrays themselves are 5x5.
    """
    out = {
        "model": "pure_gamma_5x5",
        "coords": "raw pixels",
        "gamma_L_4x4": np.asarray(gamma_l).tolist(),
        "gamma_R_4x4": np.asarray(gamma_r).tolist(),
        "baseline": {"dx": float(d[0]), "dy": float(d[1]), "dz": 0.0},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)


def load_calibration(
    path: PathLike = CALIB_PATH, rig: RigConfig = DEFAULT_RIG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a calibration file and return ``(GammaL, GammaR, d)``.

    Grids of a different size (legacy 4x4) are bilinearly resampled to the
    rig's grid and the centre node re-anchored at ``g0``.
    """
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    p = json.loads(text)
    gamma_l = np.array(p["gamma_L_4x4"], dtype=np.float64)
    gamma_r = np.array(p["gamma_R_4x4"], dtype=np.float64)
    if gamma_l.shape != rig.grid_shape:
        gamma_l = resize_grid(gamma_l, rig.grid_shape)
        gamma_r = resize_grid(gamma_r, rig.grid_shape)
        gamma_l[rig.centre] = rig.g0
        gamma_r[rig.centre] = rig.g0
    d = np.array([p["baseline"]["dx"], p["baseline"]["dy"], 0.0], dtype=np.float64)
    return gamma_l, gamma_r, d


def save_cal_pts(cal_pts: Sequence[CalPoint], path: PathLike = CAL_PTS_PATH) -> None:
    """Write the calibration points as a JSON list of ``[uL, vL, uR, vR, Z]`` rows."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([list(map(float, s)) for s in cal_pts], f, indent=2)


def load_cal_pts(path: PathLike = CAL_PTS_PATH) -> list[CalPoint]:
    """Read calibration points; a missing or empty file gives ``[]``.

    Accepts both the flat ``[uL, vL, uR, vR, Z]`` rows of ``cal_pts.json`` and
    the ``{"left_pt": [..], "right_pt": [..], "distance": z}`` records that
    ``collect_points.py`` writes.
    """
    if not os.path.exists(path):
        return []
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    rows = json.loads(text)
    out: list[CalPoint] = []
    for r in rows:
        if isinstance(r, dict):
            lp, rp = r["left_pt"], r["right_pt"]
            out.append((float(lp[0]), float(lp[1]), float(rp[0]), float(rp[1]), float(r["distance"])))
        else:
            out.append(tuple(float(x) for x in r))  # type: ignore[arg-type]
    return out


def load_image_gray(path: PathLike, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as a ``float32`` grayscale array with Pillow, optionally resized to ``(w, h)``."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    from .matching import to_gray

    return to_gray(np.asarray(img))
