# ==========================================================
# Mission diagnostics: travel times, energy budget, causal structure
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math
from dataclasses import dataclass

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .constants import (
    C_LIGHT,
    D_PROXIMA,
    G_EARTH,
    L_PLANCK,
    LY,
    M_JUP,
    M_SUN,
    YEAR,
)
from .metrics.base import EnergyBudget
from .shapes import wall_thickness


@dataclass
class MissionProfile:
    """Everything worth reporting about one bubble configuration."""

    metric_name: str
    radius: float
    thickness: float
    speed: float

    distance: float
    coordinate_time: float
    proper_time: float
    time_rate: float

    rocket_proper_time: float
    rocket_coordinate_time: float

    energy: EnergyBudget

    horizon: float | None


def relativistic_rocket(distance, accel=G_EARTH):
    """
    Reference case with no new physics: a rocket that accelerates at
    `accel` for half the trip and decelerates for the other half.

    Returns (proper time [s], coordinate time [s]).
    """

    half = 0.5 * distance
    reduced = accel * half / C_LIGHT ** 2 + 1.0

    tau = 2.0 * (C_LIGHT / accel) * math.acosh(reduced)
    t = 2.0 * (C_LIGHT / accel) * math.sqrt(reduced ** 2 - 1.0)
    return tau, t


def profile_mission(metric, distance=D_PROXIMA, analytic_energy=True):
    """
    Assemble the full diagnostic profile of a bubble on a given trip.

    The proper time is evaluated from the line element via
    `metric.proper_time_rate`, not assumed: for this whole family of
    metrics it comes out exactly equal to coordinate time, at any v_s.
    """

    coordinate_time = distance / metric.speed
    rate = metric.proper_time_rate()

    energy = metric.energy_budget_analytic() if analytic_energy else None
    if energy is None:
        energy = metric.energy_budget()

    tau_rocket, t_rocket = relativistic_rocket(distance)

    return MissionProfile(
        metric_name=metric.name,
        radius=metric.radius,
        thickness=wall_thickness(metric.sigma),
        speed=metric.speed,
        distance=distance,
        coordinate_time=coordinate_time,
        proper_time=coordinate_time * rate,
        time_rate=rate,
        rocket_proper_time=tau_rocket,
        rocket_coordinate_time=t_rocket,
        energy=energy,
        horizon=metric.horizon_offset(),
    )


def format_profile(profile):
    """Render a MissionProfile as a fixed-width report."""

    p = profile
    lines = []
    add = lines.append

    add("=" * 68)
    add(f"  {p.metric_name.upper()} WARP DRIVE - MISSION PROFILE")
    add("=" * 68)
    add(f"  bubble radius      R      = {p.radius:>12.1f} m")
    add(f"  wall thickness     1/sig  = {p.thickness:>12.2f} m"
        f"   ({p.thickness / L_PLANCK:.2e} Planck lengths)")
    add(f"  apparent speed     v_s    = {p.speed / C_LIGHT:>12.1f} c")
    add("-" * 68)
    add(f"  target at {p.distance / LY:.4f} ly")
    add(f"  coordinate time                 t   = "
        f"{p.coordinate_time / YEAR:>10.4f} yr")
    add(f"  crew proper time                tau = "
        f"{p.proper_time / YEAR:>10.4f} yr   (dtau/dt = {p.time_rate:.6f})")
    add(f"  same trip, 1g relativistic rocket:  tau = "
        f"{p.rocket_proper_time / YEAR:.3f} yr, "
        f"t = {p.rocket_coordinate_time / YEAR:.3f} yr")
    add("-" * 68)
    add("  ENERGY BUDGET")
    add(f"  negative (exotic)      E-  = {p.energy.negative:>12.4e} J")
    add(f"  positive               E+  = {p.energy.positive:>12.4e} J")
    add(f"  net                    E   = {p.energy.net:>12.4e} J")
    add(f"  exotic mass            M-  = {p.energy.negative_mass:>12.4e} kg")
    add(f"                             = "
        f"{p.energy.negative_mass / M_SUN:>12.4e} solar masses")
    add(f"                             = "
        f"{p.energy.negative_mass / M_JUP:>12.4e} Jupiter masses")
    add("-" * 68)
    add("  CAUSAL STRUCTURE")
    if p.horizon is None:
        add("  subluminal bubble: no horizon, the crew can steer the wall.")
    else:
        add(f"  future horizon at x_s = {p.horizon:>10.3f} m ahead of the ship")
        add("  -> the crew cannot signal the front wall: the bubble must be")
        add("     fully pre-programmed, it cannot be steered or stopped from")
        add("     the inside once it is superluminal.")
    add("=" * 68)

    return "\n".join(lines)


def energy_scaling_table(metric_factory, speeds, radii):
    """
    Tabulate the exotic mass, the mass equivalent of the negative part
    of the energy budget, over a grid of speeds and bubble radii.

    `metric_factory(speed, radius)` returns a configured metric, so the
    same table can be produced for any member of the family.

    Returns (table [len(speeds) x len(radii)] in kg, formatted string).
    """

    table = np.empty((len(speeds), len(radii)))
    for i, speed in enumerate(speeds):
        for j, radius in enumerate(radii):
            metric = metric_factory(speed, radius)
            budget = metric.energy_budget_analytic()
            if budget is None:
                budget = metric.energy_budget()
            table[i, j] = budget.negative_mass

    lines = []
    add = lines.append
    add("  EXOTIC MASS SCALING   (solar masses)")
    add(f"  {'v_s/c':>8} | " + " | ".join(f"R={r:>5.0f} m" for r in radii))
    add("  " + "-" * (11 + 13 * len(radii)))
    for i, speed in enumerate(speeds):
        row = " | ".join(f"{table[i, j] / M_SUN:>10.2e}"
                         for j in range(len(radii)))
        add(f"  {speed / C_LIGHT:>8.1f} | " + row)

    return table, "\n".join(lines)
