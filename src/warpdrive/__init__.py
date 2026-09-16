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
from .metrics import AlcubierreMetric, EnergyBudget, WarpMetric
from .shapes import (
    broeck_volume_profile,
    broeck_volume_profile_derivative,
    broeck_volume_profile_second_derivative,
    tanh_top_hat,
    tanh_top_hat_derivative,
    tanh_top_hat_derivative_from_wall,
    wall_thickness,
)
from .tracers import integrate_tracers, make_tracer_grid

__version__ = "0.1.0"

__all__ = [
    "AlcubierreMetric",
    "WarpMetric",
    "EnergyBudget",
    "MissionProfile",
    "profile_mission",
    "format_profile",
    "energy_scaling_table",
    "relativistic_rocket",
    "integrate_tracers",
    "make_tracer_grid",
    "tanh_top_hat",
    "tanh_top_hat_derivative",
    "tanh_top_hat_derivative_from_wall",
    "wall_thickness",
    "broeck_volume_profile",
    "broeck_volume_profile_derivative",
    "broeck_volume_profile_second_derivative",
    "C_LIGHT",
    "D_PROXIMA",
    "G",
    "LY",
    "M_SUN",
    "YEAR",
    "__version__",
]
