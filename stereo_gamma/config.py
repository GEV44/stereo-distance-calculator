"""
Sensor geometry and training hyper-parameters.

Everything that describes *the rig* lives in :class:`Sensor`; everything that
describes *how we fit it* lives in :class:`TrainConfig`.  Both are plain
dataclasses so they serialise to JSON and can be overridden from the CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# ─── SENSOR ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Sensor:
    """
    Image geometry of one camera of the rig (both cameras share it).

    Pixel coordinates are mapped to normalised coordinates

        ũ = (u − CX) / CX ,   ṽ = (v − CY) / CY       ∈ [−1, 1]

    and the Γ grid has ``grid_rows × grid_cols`` nodes spanning the full image.
    """

    width: int = 2592
    height: int = 1944
    grid_rows: int = 5
    grid_cols: int = 5
    f_init: float = 2880.0  # nominal focal length (px) — fixes the scale of Γ (accuracy is insensitive to it)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def cell_w(self) -> float:
        return self.width / (self.grid_cols - 1)

    @property
    def cell_h(self) -> float:
        return self.height / (self.grid_rows - 1)

    @property
    def gamma0(self) -> np.ndarray:
        """Nominal (pinhole) Γ = [CX/f, CY/f]  →  ray = [(u−CX)/f, (v−CY)/f, 1]."""
        return np.array([self.cx / self.f_init, self.cy / self.f_init])

    @property
    def center_node(self) -> tuple[int, int]:
        return self.grid_rows // 2, self.grid_cols // 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Sensor:
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})


DEFAULT_SENSOR = Sensor()


# ─── PHYSICAL BOUNDS ──────────────────────────────────────────────────────────
DX_MIN = 0.02  # baseline floor (m) — prevents division by zero in the loss
G_MIN, G_MAX = 0.05, 5.0  # Γ clamp — no negative / infinite focal lengths
MIN_CAL_PTS = 15  # points required before the app trains automatically


# ─── TRAINING ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """
    Hyper-parameters of the calibration objective

        L(θ) = 1/N Σ [ ρδ(e_x) + λ_y ρδ(e_y) ]
               + λ_a · ½‖Γ − Γ₀‖²                       (anchor prior)
               + λ_s · ½ Σ_edges (Γ_i − Γ_j)²            (membrane smoothness)

    minimised by Levenberg–Marquardt (default, IRLS for the Huber loss) or by
    Adam; both reach the same optimum (see tests).  The ``learn_*`` switches
    exist for ablation studies.
    """

    solver: str = "lm"  # "lm" | "adam"
    lm_max_iter: int = 200
    epochs: int = 6000  # Adam only: reaches the LM optimum to ~1e-13 relative (tests)
    lr_gamma: float = 2e-3
    lr_baseline: float = 2e-3
    lr_rotation: float = 1e-3
    lr_final_frac: float = 0.05  # cosine decay of every learning rate to this fraction
    huber_delta: float = 0.05  # 5 % relative disparity error
    weight_y: float = 0.5  # λ_y — weight of the epipolar (vertical) term
    anchor: float = 3e-3  # λ_a — pull towards the pinhole prior Γ₀
    smooth: float = 0.3  # λ_s — used as-is when auto_smooth is False
    auto_smooth: bool = True  # choose λ_s by inner k-fold CV on the training points
    smooth_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    cv_folds: int = 5
    cv_repeats: int = 3  # repeated k-fold: averages out the luck of one shuffle
    noise_model: str = "fgls"  # "fgls": estimated click/label noise → ML weights | "relative"
    radial_prior: bool = True  # shrink Γ towards a learned radial field Γ₀(1 + a₁r² + a₂r⁴), not a constant
    learn_gamma: bool = True
    learn_rotation: bool = True
    warm_start: bool = True  # closed-form linear initialisation of d and ω
    ensemble: bool = True  # average with a Brown–Conrady companion fitted to the same points …
    ensemble_min_points: int = 25  # … once there is enough data for its 18 parameters (synthetic study)
    postcorrection: bool = False  # quadratic Z-correction (PDF §12) — off: see README ablation
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    log_every: int = 0  # 0 = silent

    def to_dict(self) -> dict:
        d = asdict(self)
        d["adam_betas"] = list(self.adam_betas)
        d["smooth_grid"] = list(self.smooth_grid)
        return d
