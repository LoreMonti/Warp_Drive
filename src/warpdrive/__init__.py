# ==========================================================
# warpdrive - numerical study of warp bubble spacetimes
#
# Author: Lorenzo Monti
# ==========================================================

from .constants import C_LIGHT, D_PROXIMA, G, LY, M_SUN, YEAR
from .diagnostics import (
    MissionProfile,
    NeckScaling,
    energy_scaling_table,
    format_profile,
    neck_scaling,
    pocket_energy_floor,
    profile_mission,
    relativistic_rocket,
)
from .geodesics import (
    RayBundle,
    RayField,
    SkyMap,
    horizon_surface_gravity,
    sky_map,
    throat_radius,
    trace_rays,
    trace_rays_3d,
    unlensed_brightness,
    visible_cone,
)
from .metrics import AlcubierreMetric, BroeckMetric, EnergyBudget, WarpMetric
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
    "BroeckMetric",
    "WarpMetric",
    "EnergyBudget",
    "MissionProfile",
    "profile_mission",
    "format_profile",
    "energy_scaling_table",
    "NeckScaling",
    "neck_scaling",
    "pocket_energy_floor",
    "relativistic_rocket",
    "integrate_tracers",
    "RayBundle",
    "trace_rays",
    "horizon_surface_gravity",
    "SkyMap",
    "sky_map",
    "unlensed_brightness",
    "RayField",
    "trace_rays_3d",
    "throat_radius",
    "visible_cone",
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
