"""Shared fixtures: synthetic rigs and the shipped calibration files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stereo_distance import DEFAULT_RIG, SyntheticRig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rig():
    return DEFAULT_RIG


@pytest.fixture(scope="session")
def pinhole():
    """Ideal pinhole pair, f = 2880 px, 25 cm baseline."""
    return SyntheticRig.pinhole(dx=0.25)


@pytest.fixture(scope="session")
def distorted():
    """Smoothly distorted pair (about 8 % gamma variation) with a small vertical offset."""
    return SyntheticRig.distorted(amplitude=0.08, dx=0.25, dy=0.004, seed=1)


@pytest.fixture(scope="session")
def scene_points():
    """Fixed cloud of 3-D points in front of the left camera, 0.8-8 m."""
    rng = np.random.default_rng(42)
    z = 1.0 / rng.uniform(1 / 8.0, 1 / 0.8, 60)
    x = rng.uniform(-0.3, 0.3, 60) * z
    y = rng.uniform(-0.25, 0.25, 60) * z
    return np.stack([x, y, z], axis=1)


@pytest.fixture(scope="session")
def shipped_calibration_path():
    return REPO / "stereo_calibration.json"


@pytest.fixture(scope="session")
def shipped_cal_pts_path():
    return REPO / "cal_pts.json"
