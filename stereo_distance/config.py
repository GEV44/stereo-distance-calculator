"""Rig geometry and model hyper-parameters.

Every numeric constant of the original ``run_math.py`` lives here as a field of
:class:`RigConfig`, so the same code can serve a different sensor by passing a
different config instead of editing module globals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RigConfig:
    """Camera / grid / optimisation constants for one stereo rig.

    Attributes:
        img_w, img_h: sensor resolution in pixels (both cameras).
        grid_rows, grid_cols: size of the per-camera gamma grid.
        f_init: nominal focal length in pixels; fixes the absolute scale of
            the frozen centre node ``g0 = cx / f_init``.
        dx_min: physical floor for the horizontal baseline (metres).
        g_min, g_max: clamp range for gamma nodes.
        smooth_w: weight of the Laplacian smoothness gradient.
        min_cal_pts: calibration points required before auto-training.
    """

    img_w: int = 2592
    img_h: int = 1944
    grid_rows: int = 5
    grid_cols: int = 5
    f_init: float = 2880.0
    dx_min: float = 0.02
    g_min: float = 0.05
    g_max: float = 5.0
    smooth_w: float = 1e-2
    min_cal_pts: int = 15

    @property
    def cx(self) -> float:
        """Optical centre x (image centre)."""
        return self.img_w / 2.0

    @property
    def cy(self) -> float:
        """Optical centre y (image centre)."""
        return self.img_h / 2.0

    @property
    def cell_w(self) -> float:
        """Grid cell width in pixels."""
        return self.img_w / (self.grid_cols - 1)

    @property
    def cell_h(self) -> float:
        """Grid cell height in pixels."""
        return self.img_h / (self.grid_rows - 1)

    @property
    def g0(self) -> float:
        """Nominal gamma: ``cx / f_init`` (about 0.45 for the default rig)."""
        return self.cx / self.f_init

    @property
    def centre(self) -> tuple[int, int]:
        """Index of the frozen centre node."""
        return self.grid_rows // 2, self.grid_cols // 2

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.grid_rows, self.grid_cols


DEFAULT_RIG = RigConfig()
"""The 2592x1944 rig the project was built for."""
