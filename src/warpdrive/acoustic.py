# ==========================================================
# Phonons in a two-dimensional condensate at rest
#
# Low-energy phonons of a Bose-Einstein condensate with density n(r) and
# speed of sound c_s(r) see, in 2+1 dimensions, the acoustic metric
#
#     ds^2 = (n / c_s)^2 [ -c_s^2 dt^2 + dr^2 + r^2 dphi^2 ].
#
# Rays ignore the conformal factor and follow the geodesics of the
# optical metric dsigma^2 = (dr^2 + r^2 dphi^2) / c_s^2, whose length is
# the travel time. Waves do not: in 2+1 a scalar is not conformally
# invariant, and (n / c_s) enters the wave equation.
#
# This is the tool section 5 needs for an acoustic pocket. It is first
# checked on a published experiment: Viermann et al., Nature 611, 260
# (2022), where a harmonically trapped 2D condensate approximates a
# space of negative curvature.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .integrators import integrate_adaptive


class ThomasFermiDisc:
    """
    A two-dimensional condensate in a harmonic trap, in the Thomas-Fermi
    limit: n = n_0 (1 - r^2/R^2) and, since c_s^2 = g n / m,

        c_s(r) = c_0 sqrt(1 - r^2 / R^2),   r < R.

    Its optical metric has Gaussian curvature

        K(r) = c_s^2 lap(ln c_s) = -2 c_0^2 / (R^2 - r^2)   [s^-2],

    negative everywhere, -2 c_0^2 / R^2 at the centre and diverging at
    the edge: a space of constant negative curvature only near the
    centre, which is the sense in which Viermann et al. call it
    hyperbolic.

    Parameters
    ----------
    sound_speed : c_0, speed of sound at the centre [m s^-1]
    radius      : Thomas-Fermi radius R [m]
    density     : n_0, areal density at the centre [m^-2]
    """

    def __init__(self, sound_speed, radius, density=1.0):
        if sound_speed <= 0.0 or radius <= 0.0:
            raise ValueError("need a positive speed of sound and radius")
        self.sound_speed = float(sound_speed)
        self.radius = float(radius)
        self.density = float(density)

    @classmethod
    def from_viermann_2022(cls):
        """
        The potassium-39 condensate of Viermann et al. (2022): speed of
        sound 1.2 um/ms at the centre, Thomas-Fermi radius 25 um, central
        density 1.3e9 cm^-2.
        """

        return cls(sound_speed=1.2e-3, radius=25.0e-6, density=1.3e13)

    # --- Profiles ---
    def speed(self, r):
        """c_s(r) [m s^-1], zero at and beyond the edge."""

        u = 1.0 - (np.asarray(r, dtype=float) / self.radius) ** 2
        return self.sound_speed * np.sqrt(np.clip(u, 0.0, None))

    def speed_derivative(self, r):
        """dc_s/dr [s^-1], for r < R."""

        r = np.asarray(r, dtype=float)
        return -self.sound_speed ** 2 * r / (self.radius ** 2
                                             * self.speed(r))

    def number_density(self, r):
        """n(r) = n_0 (1 - r^2/R^2) [m^-2]."""

        u = 1.0 - (np.asarray(r, dtype=float) / self.radius) ** 2
        return self.density * np.clip(u, 0.0, None)

    def conformal_factor(self, r):
        """
        n / c_s, the factor whose square multiplies the 2+1 acoustic
        metric. Rays do not see it; waves do. For a Thomas-Fermi disc it
        is (n_0 / c_0) sqrt(1 - r^2/R^2). [s m^-3]
        """

        return self.number_density(r) / self.speed(r)

    def gaussian_curvature(self, r):
        """K(r) = -2 c_0^2 / (R^2 - r^2) of the optical metric [s^-2]."""

        r = np.asarray(r, dtype=float)
        return -2.0 * self.sound_speed ** 2 / (self.radius ** 2 - r ** 2)

    # --- Rays ---
    def radial_travel_time(self, start, end):
        """
        Time for a ray to run radially from r = start to r = end, the
        length of the optical metric along the radius,

            t = (R / c_0) [arcsin(start / R) - arcsin(end / R)]   [s].
        """

        R = self.radius
        return R / self.sound_speed * (math.asin(start / R)
                                       - math.asin(end / R))

    def trace_rays(self, positions, directions, duration, rtol=1.0e-10):
        """
        Phonon rays from the eikonal Hamiltonian w = c_s(r) |p|, in lab
        time t:

            dx/dt = c_s p / |p|,   dp/dt = -|p| grad c_s.

        w and the angular momentum x p_y - y p_x are conserved, so every
        ray is a geodesic of the optical metric.

        positions  : (n, 2) starting points [m]
        directions : (n, 2) starting directions, any length
        duration   : integration time [s]
        Returns the final (n, 2) positions and (n, 2) momenta, with |p| = 1
        initially.
        """

        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        momenta = directions / np.linalg.norm(directions, axis=1)[:, None]
        state0 = np.vstack([positions.T, momenta.T])

        def rhs(t, state, members):
            x, y, px, py = state
            r = np.hypot(x, y)
            safe = np.where(r > 0.0, r, 1.0)
            c = self.speed(r)
            dc = np.where(r > 0.0, self.speed_derivative(r), 0.0)
            norm = np.hypot(px, py)
            return np.array([c * px / norm, c * py / norm,
                             -norm * dc * x / safe, -norm * dc * y / safe])

        _, state, _ = integrate_adaptive(rhs, state0, duration, rtol=rtol,
                                         atol=1.0e-14 * self.radius,
                                         first_step=duration * 1.0e-4)
        return state[:2].T, state[2:].T
