# ==========================================================
# The Alcubierre (1994) warp drive metric
#
#     ds^2 = -c^2 dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2
#
# Reference: M. Alcubierre, Class. Quantum Grav. 11, L73 (1994)
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math
from dataclasses import dataclass

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT, G
from ..shapes import tanh_top_hat, tanh_top_hat_derivative
from .base import WarpMetric


@dataclass
class AlcubierreMetric(WarpMetric):
    """
    The original warp drive.

    Space is exactly flat both inside and outside the bubble; all of the
    curvature is confined to a wall of thickness ~ 1/sigma at r_s = R.
    The ship sits at rest in the flat interior while the wall contracts
    space ahead of it and expands it behind, so the superluminal speed
    v_s is a coordinate velocity and no local light cone is crossed.

    Parameters
    ----------
    speed  : bubble velocity dx_s/dt [m s^-1]
    radius : bubble radius R [m]
    sigma  : inverse wall thickness [m^-1]
    """

    speed: float
    radius: float = 100.0
    sigma: float = 0.10

    name: str = "Alcubierre"

    # --- Radial profiles ---
    def _radius_from(self, x, y, z):
        return np.sqrt(
            np.asarray(x, dtype=float) ** 2
            + np.asarray(y, dtype=float) ** 2
            + np.asarray(z, dtype=float) ** 2
        )

    def shape(self, r_s):
        """Shape function f(r_s): 1 inside the bubble, 0 outside."""

        return tanh_top_hat(r_s, self.radius, self.sigma)

    def shape_derivative(self, r_s):
        """df/dr_s, non-zero only inside the wall."""

        return tanh_top_hat_derivative(r_s, self.radius, self.sigma)

    # --- WarpMetric interface ---
    def shift(self, x, y, z):
        return self.speed * self.shape(self._radius_from(x, y, z))

    def conformal_factor(self, x, y, z):
        return np.ones_like(np.asarray(x, dtype=float))

    def expansion(self, x, y, z):
        """
        Expansion of the normal volume elements,

            theta = v_s (x / r_s) df/dr_s     [s^-1]

        Negative ahead of the ship (space contracting) and positive
        behind it (space expanding). This is the entire mechanism of the
        drive.

        Checked against the derivation in `symbolic.py`.
        """

        x = np.asarray(x, dtype=float)
        r_s = self._radius_from(x, y, z)
        safe = np.where(r_s > 0.0, r_s, 1.0)
        theta = self.speed * (x / safe) * self.shape_derivative(r_s)
        return np.where(r_s > 0.0, theta, 0.0)

    def energy_density(self, x, y, z):
        """
        Energy density measured by the Eulerian observers,

            eps = -(c^4 / 8 pi G) (v_s^2 / c^2)
                  (rho^2 / 4 r_s^2) (df/dr_s)^2      [J m^-3]

        with rho^2 = y^2 + z^2.

        It is negative wherever it is non-zero, so the drive violates the
        weak energy condition, and it vanishes on the axis: the exotic
        matter forms a torus around the direction of motion. That torus
        is the physical reason concept ships are drawn with rings.

        This is the fast path. The authority for the expression, the
        prefactor included, is the derivation in `symbolic.py`, which
        `tests/test_symbolic.py` checks it against.
        """

        x = np.asarray(x, dtype=float)
        rho2 = np.asarray(y, dtype=float) ** 2 + np.asarray(z, dtype=float) ** 2
        r_s2 = x ** 2 + rho2
        safe = np.where(r_s2 > 0.0, r_s2, 1.0)

        prefactor = C_LIGHT ** 2 * self.speed ** 2 / (32.0 * math.pi * G)
        eps = (
            -prefactor
            * (rho2 / safe)
            * self.shape_derivative(np.sqrt(r_s2)) ** 2
        )
        return np.where(r_s2 > 0.0, eps, 0.0)

    # --- Closed form energy budget ---
    def exotic_energy_analytic(self, n_points=200000):
        """
        The angular part of the volume integral is analytic,

            \\int (rho^2 / r^2) dOmega = 8 pi / 3,

        which collapses the budget to a single radial quadrature,

            E = -(c^2 v_s^2 / 12 G) \\int_0^inf (df/dr)^2 r^2 dr.

        Scaling: E ~ v_s^2 R^2 sigma.

        Returns (E [J], M_equivalent [kg]).
        """

        r_max = self.radius + 25.0 / self.sigma
        r = np.linspace(0.0, r_max, n_points)
        integral = np.trapezoid(self.shape_derivative(r) ** 2 * r ** 2, r)

        energy = -(C_LIGHT ** 2 * self.speed ** 2 / (12.0 * G)) * integral
        return energy, energy / C_LIGHT ** 2
