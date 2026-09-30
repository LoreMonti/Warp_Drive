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
    G,
    G_EARTH,
    HBAR,
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

    #: Offset of the horizon ahead of the ship [m]; None when there is
    #: none, or when `horizon_resolved` is False.
    horizon: float | None

    #: False when the horizon lies in a wall too thin to be located in
    #: double precision.
    horizon_resolved: bool = True

    #: Proper radius of an inflated pocket, for metrics that have one [m].
    pocket_radius: float | None = None


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

    try:
        horizon, resolved = metric.horizon_offset(), True
    except ValueError:
        horizon, resolved = None, False

    pocket = getattr(metric, "pocket_proper_radius", None)

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
        horizon=horizon,
        horizon_resolved=resolved,
        pocket_radius=pocket() if pocket is not None else None,
    )


def _length(value, decimals, width=12):
    """Fixed-point metres at human scale, scientific notation below."""

    if value >= 0.01:
        return f"{value:>{width}.{decimals}f}"
    return f"{value:>{width}.3e}"


def format_profile(profile):
    """Render a MissionProfile as a fixed-width report."""

    p = profile
    lines = []
    add = lines.append

    add("=" * 68)
    add(f"  {p.metric_name.upper()} WARP DRIVE - MISSION PROFILE")
    add("=" * 68)
    add(f"  bubble radius      R      = {_length(p.radius, 1)} m")
    add(f"  wall thickness     1/sig  = {_length(p.thickness, 2)} m"
        f"   ({p.thickness / L_PLANCK:.2e} Planck lengths)")
    add(f"  apparent speed     v_s    = {p.speed / C_LIGHT:>12.1f} c")
    if p.pocket_radius is not None:
        add(f"  pocket, proper radius     = {p.pocket_radius:>12.1f} m")
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
    if not p.horizon_resolved:
        add("  superluminal bubble: the horizon lies inside a wall too thin")
        add("  to locate in double precision, within a few wall thicknesses")
        add("  of R. The crew still cannot signal the front wall.")
    elif p.horizon is None:
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


def pocket_energy_floor(pocket_radius, outer_radius):
    """
    Smallest net E_tot of any Van Den Broeck transition region with proper
    pocket radius P and outer radius b, over every profile of B and every
    inner radius a [J].

    At fixed a the Dirichlet bound of `BroeckMetric.pocket_energy_bound`
    reads, with (1 + alpha) a = P,

        (2 c^4 / G) (sqrt(P) - sqrt(a))^2 b / (b - a),

    and its minimum over 0 < a < b sits at a* = b^2 / P, where it equals

        E_floor = (2 c^4 / G) (P - b).

    The floor grows with how much larger the pocket is inside than
    outside, P - b, the same excess that forces a throat and a violation
    of the null energy condition. For P <= b there is no floor.

    Returns (E_floor [J], a* [m]), or (0.0, None) when P <= b.
    """

    if pocket_radius <= outer_radius:
        return 0.0, None
    floor = 2.0 * C_LIGHT ** 4 / G * (pocket_radius - outer_radius)
    return floor, outer_radius ** 2 / pocket_radius


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


@dataclass
class QuantumInequalityCheck:
    """
    Ford-Roman quantum inequality applied to the transition region of a
    Van Den Broeck pocket. Densities in kg m^-3.
    """

    #: Most negative density seen by the static observers.
    peak_density: float

    #: Coordinate radius r_s of that peak [m].
    peak_radius: float

    #: Smallest curvature radius of the region [m].
    curvature_radius: float

    #: Lower bound on the density for sampling time beta r_c / c.
    limit: float

    #: Radial speed v / c of the most restrictive observer.
    worst_speed: float

    #: |density| / |limit| for that observer: above 1 the inequality is
    #: violated.
    margin: float

    @property
    def violated(self):
        return self.margin > 1.0


def quantum_inequality_check(metric, beta=0.1, n_points=400000,
                             n_speeds=1000):
    """
    Van Den Broeck's quantum-inequality check of the transition region,
    extended to radially moving observers.

    Ford and Roman's inequality for a massless scalar field, trusted in
    curved spacetime for sampling times tau_0 much shorter than the
    smallest curvature radius, tau_0 = beta r_c / c, bounds the density
    from below by

        rho >= -3 hbar c / (32 pi^2 (c tau_0)^4) / c^2   [kg m^-3].

    An observer moving radially at speed v sees eps' = gamma^2 (eps +
    v^2 p_r), and since curvature components are boosted by up to
    gamma^2 its r_c, and with it tau_0, shrinks by gamma: the limit grows
    as gamma^4. The margin of that observer is therefore

        |eps + v^2 p_r| (1 - v^2) / |limit at rest|,

    maximised here over v and over the region. Writing eps + v^2 p_r =
    (1 - v^2) eps + v^2 (eps + p_r), the margin grows away from v = 0 only
    if the most negative null contraction eps + p_r is more than twice
    the most negative density. It is not for the paper's profile, nor for
    any polynomial profile tried, so the observer at rest, the only one
    the 1999 paper checks, is already the most restrictive.

    Returns a QuantumInequalityCheck.
    """

    r = metric._transition_grid(n_points)
    r_c, _ = metric.curvature_radius(n_points)
    return _quantum_inequality(r, metric._conformal_density(r),
                               metric.pocket_pressures(r)[0], r_c, beta,
                               n_speeds)


def _quantum_inequality(position, eps, radial, r_c, beta, n_speeds):
    # eps and radial in J m^-3 on a grid of positions [m]
    k = int(np.argmin(eps))
    limit = -3.0 * HBAR * C_LIGHT / (32.0 * math.pi ** 2
                                     * (beta * r_c) ** 4) / C_LIGHT ** 2

    speeds = np.arange(n_speeds) / n_speeds
    worst = [-np.min(eps + v ** 2 * radial) * (1.0 - v ** 2)
             for v in speeds]
    j = int(np.argmax(worst))

    return QuantumInequalityCheck(
        peak_density=float(eps[k] / C_LIGHT ** 2),
        peak_radius=float(position[k]),
        curvature_radius=float(r_c),
        limit=float(limit),
        worst_speed=float(speeds[j]),
        margin=float(worst[j] / C_LIGHT ** 2 / abs(limit)),
    )


def quantum_inequality_threshold(beta=0.1):
    """
    Largest curvature radius at which a pocket whose density is set by its
    curvature, eps ~ -c^4 / (4 pi G r_c^2) as at the corners of
    `ConePocket`, can satisfy the quantum inequality [m]:

        ratio = (8 pi / 3) beta^4 (r_c / l_P)^2 <= 1
        =>  r_c <= sqrt(3 / 8 pi) l_P / beta^2,

    35 Planck lengths for beta = 0.1. The density grows as 1/r_c^2 and the
    limit as 1/r_c^4, so the inequality caps the curvature radius instead
    of bounding it from below: the check passes only for geometries
    curved on a few tens of Planck lengths. The same reasoning bounds the
    shift wall of Pfenning and Ford to ~1e2 Planck lengths.
    """

    return math.sqrt(3.0 / (8.0 * math.pi)) * L_PLANCK / beta ** 2


@dataclass
class ConePocket:
    """
    The cheapest static pocket with curvature radius at least r_min, in
    the areal form ds^2 = dl^2 + A(l)^2 dOmega^2.

    For any static spherical pocket with A = P and A' = 1 on the inside
    and A = b and A' = 1 on the outside, integrating the density
    eps = (c^4/8 pi G)(1 - A'^2 - 2 A A'')/A^2 over 4 pi A^2 dl by parts
    gives

        E_tot = (c^4 / 2G) \\int (1 + A'^2) dl + (c^4 / G)(P - b)
              >= (2 c^4 / G)(P - A_min) >= (2 c^4 / G)(P - b),

    using 1 + A'^2 >= 2|A'|: the floor of `pocket_energy_floor` for
    every spherical pocket, not only the conformally flat ones. It is
    approached by a cone, A' = -1 from A = P down to A = b, where the
    density vanishes; all the energy sits at its two corners.

    Here the corners are rounded with |A''| constant, set so that the
    radial curvature |A''|/A reaches 1/r_min^2 exactly at the smallest A
    of each corner; A' runs linearly from 1 to -1 over L_1 = 2 / k_1 and
    back over L_2 = 2 / k_2. Each ramp adds (2 c^4 / 3G) L_i, so

        E_tot = (2 c^4 / G)(P - b) + (2 c^4 / 3G)(L_1 + L_2).

    The inner corner carries positive density, the outer one the negative
    peak, close to -c^4 / (4 pi G r_min^2).

    Parameters
    ----------
    pocket_radius  : proper radius P of the pocket [m]
    outer_radius   : areal radius b outside the transition [m]
    min_curvature  : smallest curvature radius r_min allowed [m]
    """

    pocket_radius: float
    outer_radius: float
    min_curvature: float

    def __post_init__(self):
        P, b, r = self.pocket_radius, self.outer_radius, self.min_curvature
        if not P > b > 0.0:
            raise ValueError("need P > b > 0: a pocket larger inside")
        if not 0.0 < r < b / 2.0:
            raise ValueError("need 0 < r_min < b / 2 to round the corners")

    @property
    def inner_rate(self):
        """k_1 = |A''| on the inner corner, where A >= P [m^-1]."""

        return self.pocket_radius / self.min_curvature ** 2

    @property
    def outer_rate(self):
        """
        k_2 on the outer corner, where A dips to b - L_2/4 = b - 1/(2 k_2):
        the root of k_2^2 r^2 - b k_2 + 1/2 = 0 closest to b / r^2.
        """

        b, r = self.outer_radius, self.min_curvature
        return (b + math.sqrt(b ** 2 - 2.0 * r ** 2)) / (2.0 * r ** 2)

    def ramp_lengths(self):
        """Proper lengths L_1, L_2 of the two rounded corners [m]."""

        return 2.0 / self.inner_rate, 2.0 / self.outer_rate

    def energy(self):
        """Net E_tot of the transition, in closed form [J]."""

        first, second = self.ramp_lengths()
        return 2.0 * C_LIGHT ** 4 / G * (
            self.pocket_radius - self.outer_radius
            + (first + second) / 3.0)

    def _outer_corner(self, n_points):
        # A and A' across the outer corner, s the distance into it
        k, b = self.outer_rate, self.outer_radius
        s = (np.arange(n_points) + 0.5) / n_points * (2.0 / k)
        return s, b - s + 0.5 * k * s ** 2, -1.0 + k * s

    def riemann_components(self, n_points=10001):
        """
        Largest |K_1| = |A''|/A and |K_2| = |1 - A'^2|/A^2 over both
        corners; zero along the cone and outside. [m^-2]
        """

        _, A, slope = self._outer_corner(n_points)
        radial = max(self.inner_rate / self.pocket_radius,
                     float(np.max(self.outer_rate / A)))
        tangential = max(1.0 / self.pocket_radius ** 2,
                         float(np.max((1.0 - slope ** 2) / A ** 2)))
        return radial, tangential

    def curvature_radius(self, n_points=10001):
        """1 / sqrt(max |R|): r_min up to the sampling of the corner [m]."""

        return 1.0 / math.sqrt(max(self.riemann_components(n_points)))

    def densities(self, n_points=10001):
        """
        eps and p_r across the outer corner, the only place where the
        density is negative [J m^-3]:

            eps = (c^4/8 pi G)(1 - A'^2 - 2 A A'')/A^2,
            p_r = (c^4/8 pi G)(A'^2 - 1)/A^2.
        """

        s, A, slope = self._outer_corner(n_points)
        scale = C_LIGHT ** 4 / (8.0 * math.pi * G)
        eps = scale * (1.0 - slope ** 2 - 2.0 * A * self.outer_rate) / A ** 2
        radial = scale * (slope ** 2 - 1.0) / A ** 2
        return s, eps, radial

    def quantum_inequality(self, beta=0.1, n_points=10001, n_speeds=1000):
        """The check of `quantum_inequality_check`, on the outer corner."""

        s, eps, radial = self.densities(n_points)
        return _quantum_inequality(s, eps, radial,
                                   self.curvature_radius(n_points), beta,
                                   n_speeds)


def curvature_radius_by_order(alpha, orders, n_points=200000):
    """
    D~ / r_c of Van Den Broeck's polynomial profile for each order n, at
    fixed alpha. The ratio depends on alpha and n alone, not on the
    lengths, so it is computed with R~ = D~ = 1 m.

    The 1999 paper says n = 80 gives the largest r_c. That holds for
    alpha = 1e34, the configuration of its first four arXiv versions,
    whose optimum n = 84 is only 0.1% smoother; for the alpha = 1e17 of
    the published eq. (7) the best order is n = 44.

    Returns an array of D~ / r_c, smaller meaning smoother.
    """

    ratios = []
    for order in orders:
        metric = BroeckMetric(speed=C_LIGHT, radius=3.0, sigma=1.0e3,
                              inner_radius=1.0, thickness=1.0,
                              alpha=alpha, order=int(order))
        ratios.append(1.0 / metric.curvature_radius(n_points)[0])
    return np.array(ratios)
