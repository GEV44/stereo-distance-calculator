import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def real_points():
    from stereo_gamma.data import load_points

    return load_points(os.path.join(ROOT, "cal_pts.json"))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def small_scene():
    """Quarter-resolution ray-traced stereo pair with a model trained on it."""
    from stereo_gamma.synthetic import SyntheticRig, render, scene_points
    from stereo_gamma.training import train

    rig = SyntheticRig().scaled(0.25)
    left, depth = render(rig, "left")
    right, _ = render(rig, "right")
    pts = scene_points(rig, 50, np.random.default_rng(0), click_sigma=0.3)
    model, _ = train(pts, sensor=rig.sensor, log=None)
    return rig, left, right, depth, model
