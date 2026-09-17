# ==========================================================
# The Van Den Broeck (1999) two-scale warp bubble
#
#     ds^2 = -c^2 dt^2 + B(r_s)^2 [ (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2 ]
#
# Alcubierre's shift is kept unchanged; the conformal factor B inflates
# the spatial volume inside a small neck, so a pocket with a large proper
# radius hides behind a wall of microscopic area.
#
# Only the separated configuration is implemented: the transition region
# of B lies entirely inside the flat interior of f. The coupling terms of
# the derivation, all proportional to B' (1 - f), then vanish, and the
# energy density is the sum of a static term from B and the Alcubierre
# term from f.
#
# Reference: C. Van Den Broeck, Class. Quantum Grav. 16, 3973 (1999)
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math
from dataclasses import dataclass

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT, G, L_PLANCK
from ..shapes import (
    broeck_volume_profile,
    broeck_volume_profile_derivative,
    broeck_volume_profile_second_derivative,
)
from .alcubierre import AlcubierreMetric
from .base import WALL_HALF_WIDTH, EnergyBudget


@dataclass
class BroeckMetric(AlcubierreMetric):
    """
    Alcubierre's shift wrapped around a volume-inflating pocket.

    Four regions, from the centre outwards: the pocket (B = 1 + alpha,
    f = 1), the transition of B, the flat Alcubierre interior (B = 1,
    f = 1), and the shift wall at r_s = R. Space is flat except in the
    transition and the wall.

    The defaults are macroscopic, so that every generic tool in the
    package - figures, quadrature, horizon - can resolve them. They carry
    no physical meaning; `from_paper` builds the configuration of the
    1999 paper.

    Parameters
    ----------
    speed        : bubble velocity dx_s/dt [m s^-1]
    radius       : radius R of the shift wall [m]
    sigma        : inverse thickness of the shift wall [m^-1]
    inner_radius : radius R~ of the pocket [m]
    thickness    : thickness D~ of the transition region of B [m]
    alpha        : B = 1 + alpha in the pocket
    order        : degree n of the polynomial profile
    """

    radius: float = 100.0
    sigma: float = 1.0
    inner_radius: float = 10.0
    thickness: float = 10.0
    alpha: float = 10.0
    order: int = 80

    name: str = "Van Den Broeck"

    def __post_init__(self):
        # evaluating the profile once validates alpha, order and lengths
        self.conformal_profile(0.0)

        outer = self.inner_radius + self.thickness
        wall = self.radius - WALL_HALF_WIDTH / self.sigma
        if outer > wall:
            raise ValueError(
                f"transition region of B ends at {outer:.3e} m but the shift "
                f"wall starts at {wall:.3e} m; overlapping regions are not "
                "implemented"
            )

    @classmethod
    def from_paper(cls, speed=C_LIGHT):
        """
        The configuration of the 1999 paper (eq. 7): alpha = 1e17,
        R~ = D~ = 1e-15 m, R = 3e-15 m, n = 80, and a shift wall of
        1e2 Planck lengths, the thickest the quantum inequalities allow
        for v_s ~ c (eq. 6).
        """

        return cls(
            speed=speed,
            radius=3.0e-15,
            sigma=1.0 / (1.0e2 * L_PLANCK),
            inner_radius=1.0e-15,
            thickness=1.0e-15,
            alpha=1.0e17,
            order=80,
        )

    # --- Radial profiles ---
    def _profile_parameters(self):
        return dict(inner_radius=self.inner_radius, thickness=self.thickness,
                    alpha=self.alpha, order=self.order)

    def conformal_profile(self, r_s):
        """B(r_s): 1 + alpha in the pocket, 1 outside the transition."""

        return broeck_volume_profile(r_s, **self._profile_parameters())

    def pocket_proper_radius(self):
        """
        Proper radius of the pocket, \\int_0^R~ B dr = (1 + alpha) R~ [m].

        For the paper's parameters this is 100 m, behind a shift wall of
        coordinate radius 3e-15 m.
        """

        return (1.0 + self.alpha) * self.inner_radius

    def _conformal_density(self, r_s):
        """
        The static part of the energy density, from the curvature of the
        spatial metric gamma_ij = B^2 delta_ij,

            eps_B = (c^4 / 8 pi G)
                    [B'^2 / B^4 - 2 B'' / B^3 - 4 B' / (r_s B^3)]  [J m^-3]

        independent of v_s and of direction. It changes sign inside the
        transition region, but its volume integral is strictly positive.
        Matches eq. (11) of the 1999 paper.
        """

        parameters = self._profile_parameters()
        r_s = np.asarray(r_s, dtype=float)
        B = broeck_volume_profile(r_s, **parameters)
        first = broeck_volume_profile_derivative(r_s, **parameters)
        second = broeck_volume_profile_second_derivative(r_s, **parameters)

        safe = np.where(r_s > 0.0, r_s, 1.0)
        bracket = (first ** 2 / B ** 4 - 2.0 * second / B ** 3
                   - 4.0 * first / (safe * B ** 3))
        eps = C_LIGHT ** 4 / (8.0 * math.pi * G) * bracket
        return np.where(r_s > 0.0, eps, 0.0)

    # --- WarpMetric interface ---
    def conformal_factor(self, x, y, z):
        return self.conformal_profile(self._radius_from(x, y, z))

    def conformal_radial_derivative(self, r_s):
        return broeck_volume_profile_derivative(r_s,
                                                **self._profile_parameters())

    def energy_density(self, x, y, z):
        """
        Energy density measured by the Eulerian observers, from
        `symbolic.py` with B left abstract. In the separated configuration
        it is the static term of the transition region plus the Alcubierre
        term of the shift wall,

            eps = eps_B(r_s)
                  - (c^4 / 8 pi G) (v_s^2 / c^2)
                    (rho^2 / 4 r_s^2) (df/dr_s)^2      [J m^-3]

        Unlike Alcubierre's, it is not sign-definite: the transition region
        carries both signs.

        The expansion is inherited unchanged: its B-dependent term is
        -3 v_s (x_s / r_s) (B'/B)(1 - f), which vanishes wherever B' does
        or f = 1.
        """

        return (self._conformal_density(self._radius_from(x, y, z))
                + super().energy_density(x, y, z))

    # --- Energy budget ---
    def energy_regions(self):
        return super().energy_regions() + [
            (self.inner_radius, self.inner_radius + self.thickness)
        ]

    def transition_energy_budget(self, n_points=200000):
        """
        Energy budget of the transition region of B alone. The density is
        isotropic there, so the angular integral is 4 pi and each sign
        reduces to

            E_B,-+ = 4 pi \\int min/max(eps_B, 0) B^3 r^2 dr,

        evaluated with the midpoint rule.

        Returns an EnergyBudget.
        """

        # Midpoint rule: d^2B/dr^2 jumps at r = R~, so the endpoint value a
        # trapezoid rule would use is not the limit from inside the region,
        # and the error would fall only as 1/n_points.
        step = self.thickness / n_points
        r = self.inner_radius + (np.arange(n_points) + 0.5) * step
        integrand = (4.0 * math.pi * self._conformal_density(r)
                     * self.conformal_profile(r) ** 3 * r ** 2)

        return EnergyBudget(
            negative=float(np.minimum(integrand, 0.0).sum() * step),
            positive=float(np.maximum(integrand, 0.0).sum() * step),
        )

    def energy_budget_analytic(self, n_points=200000):
        """
        Shift wall, integrated over the wall offset as for Alcubierre,
        plus the transition region of B. The wall contributes only to the
        negative part.

        Returns an EnergyBudget.
        """

        wall = super().energy_budget_analytic(n_points)
        transition = self.transition_energy_budget(n_points)
        return EnergyBudget(negative=wall.negative + transition.negative,
                            positive=wall.positive + transition.positive)
