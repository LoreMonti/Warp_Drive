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
from .metrics.alcubierre import AlcubierreMetric
from .metrics.base import EnergyBudget
from .metrics.broeck import BroeckMetric
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


@dataclass
class NeckScaling:
    """
    Energy budget of Van Den Broeck bubbles with a fixed pocket, as a
    function of the radius of the neck. All energies in joules.
    """

    #: Radius R of the shift wall, the neck [m].
    neck_radius: np.ndarray

    #: Negative energy of the shift wall.
    wall: np.ndarray

    #: Negative and positive energy of the transition region of B.
    transition_negative: np.ndarray
    transition_positive: np.ndarray

    #: Negative energy of an Alcubierre bubble as large as the pocket,
    #: with the same wall: what the neck is there to avoid.
    alcubierre: float

    @property
    def total_negative(self):
        return self.wall + self.transition_negative


def neck_scaling(neck_radii, pocket_radius=100.0, wall=1.0e2 * L_PLANCK,
                 speed=C_LIGHT, order=80):
    """
    Scan the neck radius R at a fixed proper pocket radius.

    Each bubble keeps the proportions of the 1999 paper, R~ = D~ = R/3,
    and inflates the pocket with alpha = pocket_radius / R~ - 1, so that
    (1 + alpha) R~ = pocket_radius throughout. The shift wall has
    thickness `wall`, 1e2 Planck lengths by default.

    The wall energy scales as R^2, while the transition region depends on
    alpha R~ alone and stays put: shrinking the neck lowers the exotic
    energy only until the transition region dominates.

    Returns a NeckScaling.
    """

    radii = np.asarray(neck_radii, dtype=float)
    if np.any(radii > 3.0 * pocket_radius):
        raise ValueError("the neck must not be larger than the pocket: "
                         "need R <= 3 pocket_radius so that alpha >= 0")

    wall_part = np.empty_like(radii)
    negative = np.empty_like(radii)
    positive = np.empty_like(radii)

    for i, radius in enumerate(radii):
        inner = radius / 3.0
        metric = BroeckMetric(
            speed=speed, radius=radius, sigma=1.0 / wall,
            inner_radius=inner, thickness=inner,
            alpha=pocket_radius / inner - 1.0, order=order,
        )
        transition = metric.transition_energy_budget()
        total = metric.energy_budget_analytic()

        wall_part[i] = total.negative - transition.negative
        negative[i] = transition.negative
        positive[i] = transition.positive

    reference = AlcubierreMetric(speed=speed, radius=pocket_radius,
                                 sigma=1.0 / wall)

    return NeckScaling(
        neck_radius=radii,
        wall=wall_part,
        transition_negative=negative,
        transition_positive=positive,
        alcubierre=reference.energy_budget_analytic().negative,
    )
