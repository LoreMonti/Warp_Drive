# ==========================================================
# Abstract interface for warp bubble metrics
#
# Every metric in this package is written in the 3+1 (ADM) form
#
#     ds^2 = -c^2 dt^2 + B(r_s)^2 [ (dx - beta(r_s) dt)^2 + dy^2 + dz^2 ]
#
# so that a concrete metric is fully specified by two radial profiles:
#
#     beta(r_s)  the drag velocity, which carries the coordinates along
#     B(r_s)     the spatial conformal factor, which inflates volume
#
# beta is the velocity of the Eulerian observers, not the ADM shift: the
# standard ADM form has (dx^i + beta^i_ADM dt), so beta^x_ADM = -beta.
#
# Alcubierre (1994) has B = 1 and beta = v_s f(r_s).
# Van Den Broeck (1999) keeps that shift and adds a non-trivial B.
#
# All coordinates passed to these methods are measured *from the centre
# of the bubble*; the driver is responsible for the rigid translation
# x -> x - x_s(t).
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
from abc import ABC, abstractmethod
from dataclasses import dataclass

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT


#: Radial extent of a tanh wall, in units of 1/sigma. sech^2 has decayed
#: to ~e^-60 there, far below anything that affects the budget.
WALL_HALF_WIDTH = 30.0

#: Narrowest region the generic quadrature accepts, relative to its outer
#: radius. Below this the grid spacing approaches the rounding of r
#: itself and the trapezoid rule returns noise rather than an error.
MIN_RELATIVE_WIDTH = 1.0e-6


@dataclass(frozen=True)
class EnergyBudget:
    """
    Total energy measured by the Eulerian observers, split by sign.

    The two parts are kept apart because they are different physical
    statements: the negative part is the exotic matter the geometry
    demands, the positive part is ordinary matter. Summing them into a
    single number would report, for Van Den Broeck's bubble, a positive
    total that hides several solar masses of negative energy.
    """

    #: Integral of the negative part of the density [J], <= 0.
    negative: float

    #: Integral of the positive part of the density [J], >= 0.
    positive: float

    @property
    def net(self):
        """negative + positive [J]."""

        return self.negative + self.positive

    @property
    def negative_mass(self):
        """Mass equivalent of the negative part, E_- / c^2 [kg]."""

        return self.negative / C_LIGHT ** 2

    @property
    def positive_mass(self):
        """Mass equivalent of the positive part, E_+ / c^2 [kg]."""

        return self.positive / C_LIGHT ** 2

    @property
    def net_mass(self):
        """Mass equivalent of the net energy [kg]."""

        return self.net / C_LIGHT ** 2


def merge_regions(regions):
    """
    Sort radial intervals and merge the ones that overlap or touch, so
    that no part of space is integrated twice.
    """

    merged = []
    for inner, outer in sorted(regions):
        if merged and inner <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], outer))
        else:
            merged.append((inner, outer))
    return merged


class WarpMetric(ABC):
    """
    Base class for axisymmetric warp bubble metrics.

    Subclasses must provide the two radial profiles and the two derived
    quantities that cannot be obtained generically: the expansion scalar
    and the energy density, both of which follow from a metric-specific
    computation of the Einstein tensor.

    Everything else - proper time, the energy budget, the location of
    the horizon - is derived here from the interface, so it is written
    once and stays correct for every metric.
    """

    #: Bubble speed [m s^-1]; a coordinate velocity, not a local one.
    speed: float

    #: Bubble radius [m].
    radius: float

    #: Inverse wall thickness [m^-1].
    sigma: float

    name: str = "warp metric"

    # --- Metric functions ---
    @staticmethod
    def lapse():
        """
        ADM lapse. Constant in this family of metrics: coordinate time
        and the proper time of the Eulerian observers tick together.
        """

        return C_LIGHT

    @abstractmethod
    def shift(self, x, y, z):
        """
        Drag velocity beta(r_s) along the axis of motion, equal to minus
        the ADM shift beta^x_ADM. [m s^-1]
        """

    @abstractmethod
    def conformal_factor(self, x, y, z):
        """Spatial conformal factor B(r_s), dimensionless."""

    @abstractmethod
    def expansion(self, x, y, z):
        """Expansion of the normal volume elements, theta. [s^-1]"""

    @abstractmethod
    def energy_density(self, x, y, z):
        """Energy density seen by the Eulerian observers. [J m^-3]"""

    # --- Derived quantities ---
    def is_superluminal(self):
        """True when the bubble outruns light in the exterior region."""

        return self.speed > C_LIGHT

    def proper_time_rate(self, dx_dt=None, x=0.0, y=0.0, z=0.0):
        """
        dtau/dt for an observer at (x, y, z) moving with coordinate
        velocity dx_dt along the axis.

        Evaluated from the line element rather than assumed. The default
        is the ship itself: sitting at the centre of the bubble and
        comoving with it, dx_dt = v_s.

        Inside the bubble beta = v_s and B = 1, so the two terms cancel
        exactly and dtau/dt = 1 for any v_s, however large.
        """

        if dx_dt is None:
            dx_dt = self.speed

        beta = float(self.shift(x, y, z))
        conformal = float(self.conformal_factor(x, y, z))
        ds2 = -C_LIGHT ** 2 + (conformal * (dx_dt - beta)) ** 2

        if ds2 >= 0.0:
            raise ValueError(
                "worldline is not timelike at this point: "
                f"ds^2 = {ds2:.3e} >= 0"
            )
        return np.sqrt(-ds2) / C_LIGHT

    def energy_regions(self):
        """
        Radial intervals (inner, outer) [m] outside which the energy
        density vanishes.

        The default is the wall of the shift, R +- 30/sigma. A metric
        with more structure, such as the transition region of B, adds its
        own intervals; overlaps are merged before integrating.
        """

        half = WALL_HALF_WIDTH / self.sigma
        return [(max(0.0, self.radius - half), self.radius + half)]

    def energy_budget(self, n_radial=4000, n_polar=800):
        """
        Energy budget by direct quadrature of the density,

            E_-+ = \\int min/max(eps, 0) sqrt(gamma) d^3x,
            sqrt(gamma) = B^3,

        on a spherical grid laid over each region of `energy_regions`
        separately, so a thin wall gets its own n_radial points instead
        of a share of a grid spanning the whole bubble. Axisymmetry about
        the x axis makes the azimuthal integral a factor of 2 pi.

        This works for any metric implementing the interface. Metrics
        with a closed form override `energy_budget_analytic` and are
        checked against this routine in the test suite.

        Raises ValueError for a region too thin, relative to its radius,
        to be resolved in double precision; such walls need a closed
        form written in the distance from the wall.

        Returns an EnergyBudget.
        """

        polar = np.linspace(0.0, np.pi, n_polar)
        negative = positive = 0.0

        for inner, outer in merge_regions(self.energy_regions()):
            if outer - inner < MIN_RELATIVE_WIDTH * outer:
                raise ValueError(
                    f"region [{inner:.3e}, {outer:.3e}] m is too thin to "
                    "resolve in r; use a closed form in the wall offset"
                )

            r = np.linspace(inner, outer, n_radial)
            R_GRID, THETA = np.meshgrid(r, polar, indexing="ij")

            x = R_GRID * np.cos(THETA)
            rho = R_GRID * np.sin(THETA)

            eps = self.energy_density(x, rho, 0.0)
            conformal = self.conformal_factor(x, rho, 0.0)
            integrand = eps * conformal ** 3 * R_GRID ** 2 * np.sin(THETA)

            def integrate(values):
                return 2.0 * np.pi * np.trapezoid(
                    np.trapezoid(values, polar, axis=1), r
                )

            negative += integrate(np.minimum(integrand, 0.0))
            positive += integrate(np.maximum(integrand, 0.0))

        return EnergyBudget(negative=negative, positive=positive)

    def energy_budget_analytic(self):
        """
        Closed-form energy budget, when the metric admits one.

        Returns an EnergyBudget, or None.
        """

        return None

    def horizon_offset(self, tol=1.0e-9, max_iterations=200):
        """
        Distance ahead of the ship at which a future horizon forms.

        A photon travelling forward along the axis obeys, from ds^2 = 0,

            dx/dt = beta + c/B,

        so relative to the bubble it advances at beta + c/B - v_s. Where
        that vanishes the crew can no longer send a signal forward: the
        front wall of their own bubble is causally disconnected, and the
        drive cannot be steered, slowed or switched off from the inside.

        The bisection stops once the bracket is `tol` times the smaller of
        the radius and the wall thickness, so it locates the horizon
        within the wall at any scale.

        Returns the offset in metres, or None for a subluminal bubble
        (which has no horizon). Raises ValueError when the wall is too
        thin relative to the radius to be resolved in double precision.
        """

        def photon_speed(offset):
            beta = float(self.shift(offset, 0.0, 0.0))
            conformal = float(self.conformal_factor(offset, 0.0, 0.0))
            return beta + C_LIGHT / conformal - self.speed

        half = WALL_HALF_WIDTH / self.sigma
        lo, hi = 0.0, self.radius + half
        if photon_speed(lo) <= 0.0 or photon_speed(hi) >= 0.0:
            return None

        if half < MIN_RELATIVE_WIDTH * hi:
            raise ValueError(
                f"wall of thickness {1.0 / self.sigma:.3e} m is too thin to "
                f"resolve around a radius of {self.radius:.3e} m"
            )

        # Not monotonic in general: with a conformal factor, c/B is tiny in
        # the pocket and grows through the transition region. It is
        # positive everywhere inside the shift wall and negative outside
        # it, though, so the bracket holds a single sign change.
        resolution = tol * min(self.radius, 1.0 / self.sigma)
        for _ in range(max_iterations):
            if hi - lo <= resolution:
                break
            mid = 0.5 * (lo + hi)
            if photon_speed(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def __repr__(self):
        return (
            f"{type(self).__name__}(speed={self.speed / C_LIGHT:g}c, "
            f"radius={self.radius:g} m, sigma={self.sigma:g} 1/m)"
        )
