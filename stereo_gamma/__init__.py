"""
stereo_gamma — metric distance from a calibrated stereo pair with a
per-camera Γ-grid lens model, analytic gradients and pure NumPy.

    from stereo_gamma import StereoModel, train, load_points
    model, _ = train(load_points("cal_pts.json"))
    Z = model.depth(uL, vL, uR, vR)
"""

from .config import DEFAULT_SENSOR, Sensor, TrainConfig
from .data import load_points, save_points
from .model import StereoModel
from .training import train

__version__ = "2.1.0"
__all__ = ["DEFAULT_SENSOR", "Sensor", "StereoModel", "TrainConfig", "load_points", "save_points", "train"]
