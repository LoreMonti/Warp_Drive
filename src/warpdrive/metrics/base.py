# ==========================================================
# Abstract interface for warp bubble metrics
#
# Every metric in this package is written in the 3+1 (ADM) form
#
#     ds^2 = -c^2 dt^2 + B(r_s)^2 [ (dx - beta(r_s) dt)^2 + dy^2 + dz^2 ]
#
# so that a concrete metric is fully specified by two radial profiles:
#
#     beta(r_s)  the shift vector, which drags the coordinates along
#     B(r_s)     the spatial conformal factor, which inflates volume
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

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT


class WarpMetric(ABC):
    """
    Base class for axisymmetric warp bubble metrics.

    Subclasses must provide the two radial profiles and the two derived
    quantities that cannot be obtained generically: the expansion scalar
    and the energy density, both of which follow from a metric-specific
    computation of the Einstein tensor.

    Everything else - proper time, the total exotic energy budget, the
    location of the horizon - is derived here from the interface, so it
    is written once and stays correct for every metric.
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
        """Shift vector component beta^x(r_s). [m s^-1]"""

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

    def total_exotic_energy(self, r_max=None, n_radial=4000, n_polar=800):
        """
        Total energy of the bubble,

            E = \\int eps sqrt(gamma) d^3x,   sqrt(gamma) = B^3,

        integrated on a spherical grid centred on the bubble. Axisymmetry
        about the x axis makes the azimuthal integral a factor of 2 pi.

        This generic quadrature works for any metric implementing the
        interface. Metrics with a closed form should override
        `exotic_energy_analytic` and are checked against this routine in
        the test suite.

        Returns (E [J], M_equivalent [kg]).
        """

        if r_max is None:
            r_max = self.radius + 30.0 / self.sigma

        r = np.linspace(0.0, r_max, n_radial)
        polar = np.linspace(0.0, np.pi, n_polar)
        R_GRID, THETA = np.meshgrid(r, polar, indexing="ij")

        x = R_GRID * np.cos(THETA)
        rho = R_GRID * np.sin(THETA)

        eps = self.energy_density(x, rho, 0.0)
        conformal = self.conformal_factor(x, rho, 0.0)
        integrand = eps * conformal ** 3 * R_GRID ** 2 * np.sin(THETA)

        energy = 2.0 * np.pi * np.trapezoid(
            np.trapezoid(integrand, polar, axis=1), r
        )
        return energy, energy / C_LIGHT ** 2

    def exotic_energy_analytic(self):
        """
        Closed-form energy budget, when the metric admits one.

        Returns (E [J], M_equivalent [kg]) or None.
        """

        return None

    def horizon_offset(self, tol=1.0e-9):
        """
        Distance ahead of the ship at which a future horizon forms.

        A photon travelling forward along the axis obeys, from ds^2 = 0,

            dx/dt = beta + c/B,

        so relative to the bubble it advances at beta + c/B - v_s. Where
        that vanishes the crew can no longer send a signal forward: the
        front wall of their own bubble is causally disconnected, and the
        drive cannot be steered, slowed or switched off from the inside.

        Returns the offset in metres, or None for a subluminal bubble
        (which has no horizon).
        """

        def photon_speed(offset):
            beta = float(self.shift(offset, 0.0, 0.0))
            conformal = float(self.conformal_factor(offset, 0.0, 0.0))
            return beta + C_LIGHT / conformal - self.speed

        lo, hi = 0.0, self.radius + 30.0 / self.sigma
        if photon_speed(lo) <= 0.0 or photon_speed(hi) >= 0.0:
            return None

        # photon_speed decreases monotonically from the centre outwards
        while hi - lo > tol * max(1.0, self.radius):
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
