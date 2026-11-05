"""stereo_distance - pure-NumPy stereo depth from a hand-calibrated camera pair.

Public API (also re-exported by the legacy ``run_math.py`` script)::

    from stereo_distance import (
        RigConfig, DEFAULT_RIG, default_params,
        bilinear, build_ray, laplacian_grad, predict_right,
        triangulate, triangulate_pair, triangulate_batch,
        forward_grads, train, TrainConfig, CalibrationResult,
        ncc_match, ssd_subpixel, lk_refine, clahe, to_gray,
        save_calibration, load_calibration, save_cal_pts, load_cal_pts,
    )
"""

from .calibration import (
    CalibrationResult,
    CalPoint,
    TrainConfig,
    depth_mae,
    forward_grads,
    forward_grads_batch,
    loss_only,
    train,
    warm_start_baseline,
)
from .config import DEFAULT_RIG, RigConfig
from .io import (
    CAL_PTS_PATH,
    CALIB_PATH,
    load_cal_pts,
    load_calibration,
    load_image_gray,
    save_cal_pts,
    save_calibration,
)
from .lens import (
    bilinear,
    bilinear_batch,
    build_ray,
    build_rays_batch,
    default_params,
    laplacian_grad,
    predict_right,
    ray_to_pixel,
    resize_grid,
)
from .matching import clahe, lk_refine, ncc_match, ssd_subpixel, to_gray, zncc
from .synthetic import SyntheticRig, make_texture, shift_image
from .triangulation import Triangulation, triangulate, triangulate_batch, triangulate_pair

__version__ = "1.0.0"

__all__ = [
    "CAL_PTS_PATH",
    "CALIB_PATH",
    "DEFAULT_RIG",
    "CalPoint",
    "CalibrationResult",
    "RigConfig",
    "SyntheticRig",
    "TrainConfig",
    "Triangulation",
    "__version__",
    "bilinear",
    "bilinear_batch",
    "build_ray",
    "build_rays_batch",
    "clahe",
    "default_params",
    "depth_mae",
    "forward_grads",
    "forward_grads_batch",
    "laplacian_grad",
    "lk_refine",
    "load_cal_pts",
    "load_calibration",
    "load_image_gray",
    "loss_only",
    "make_texture",
    "ncc_match",
    "predict_right",
    "ray_to_pixel",
    "resize_grid",
    "save_cal_pts",
    "save_calibration",
    "shift_image",
    "ssd_subpixel",
    "to_gray",
    "train",
    "triangulate",
    "triangulate_batch",
    "triangulate_pair",
    "warm_start_baseline",
    "zncc",
]
