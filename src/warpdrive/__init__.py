# ==========================================================
# warpdrive - numerical study of warp bubble spacetimes
#
# Author: Lorenzo Monti
# ==========================================================

from .constants import C_LIGHT, D_PROXIMA, G, LY, M_SUN, YEAR
from .diagnostics import (
    MissionProfile,
    energy_scaling_table,
    format_profile,
    profile_mission,
    relativistic_rocket,
)
from .metrics import AlcubierreMetric, WarpMetric
from .shapes import tanh_top_hat, tanh_top_hat_derivative, wall_thickness
from .tracers import integrate_tracers, make_tracer_grid

__version__ = "0.1.0"

__all__ = [
    "AlcubierreMetric",
    "WarpMetric",
    "MissionProfile",
    "profile_mission",
    "format_profile",
    "energy_scaling_table",
    "relativistic_rocket",
    "integrate_tracers",
    "make_tracer_grid",
    "tanh_top_hat",
    "tanh_top_hat_derivative",
    "wall_thickness",
    "C_LIGHT",
    "D_PROXIMA",
    "G",
    "LY",
    "M_SUN",
    "YEAR",
    "__version__",
]
