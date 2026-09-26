"""
Calibration-point I/O.

Two on-disk formats are accepted and normalised to an ``(N, 5)`` float array
``[uL, vL, uR, vR, Z]``:

* compact tuples (``cal_pts.json`` written by the measuring app)::

      [[uL, vL, uR, vR, Z], ...]

* labelled records (``training_data.json`` written by ``collect_points.py``)::

      [{"left_pt": [uL, vL], "right_pt": [uR, vR], "distance": Z,
        "left_file": "...", "right_file": "..."}, ...]
"""

from __future__ import annotations

import json
import os

import numpy as np


def parse_points(raw) -> np.ndarray:
    rows = []
    for i, e in enumerate(raw):
        if isinstance(e, dict):
            row = [*e["left_pt"], *e["right_pt"], e["distance"]]
        else:
            row = list(e)
        if len(row) != 5:
            raise ValueError(f"point {i}: expected 5 values (uL, vL, uR, vR, Z), got {row!r}")
        rows.append([float(x) for x in row])
    pts = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    bad = ~np.isfinite(pts).all(axis=1) | (pts[:, 4] <= 0)
    if bad.any():
        raise ValueError(f"invalid points (non-finite or Z ≤ 0) at rows {np.flatnonzero(bad).tolist()}")
    return pts


def load_points(path: str | os.PathLike) -> np.ndarray:
    """Load calibration points; a missing or empty file yields an empty array."""
    if not os.path.exists(path):
        return np.zeros((0, 5))
    with open(path, encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return np.zeros((0, 5))
    return parse_points(json.loads(content))


def save_points(pts, path: str | os.PathLike) -> None:
    """Write the compact tuple format (integers stay integers for readability)."""
    out = []
    for row in np.asarray(pts, dtype=np.float64).reshape(-1, 5):
        out.append([int(x) if float(x).is_integer() and i < 4 else float(x) for i, x in enumerate(row)])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
