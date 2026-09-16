# ==========================================================
# Tests for the Van Den Broeck metric
#
# The comparison with the symbolic derivation lives in test_symbolic.py;
# these pin the energy budget against a closed form, against the
# published numbers, and against the Alcubierre limit.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest

from warpdrive import AlcubierreMetric, BroeckMetric, neck_scaling
from warpdrive.constants import C_LIGHT, G, L_PLANCK
from warpdrive.shapes import broeck_volume_profile_derivative


@pytest.fixture(scope="module")
def metric():
    return BroeckMetric(speed=10.0 * C_LIGHT)


@pytest.fixture(scope="module")
def paper():
    return BroeckMetric.from_paper()


def test_net_transition_energy_matches_the_closed_form(paper):
    """
    Writing psi = sqrt(B), the static density is -(c^4/8 pi G) 8 psi^-5
    lap(psi) / 2 and sqrt(gamma) = psi^6, so integrating by parts gives

        E_B = (c^4 / 2 G) \\int (B'^2 / B) r^2 dr > 0,

    a positive net budget however the density is distributed in sign.
    """

    budget = paper.transition_energy_budget()

    n = 200000
    step = paper.thickness / n
    r = paper.inner_radius + (np.arange(n) + 0.5) * step
    first = broeck_volume_profile_derivative(
        r, paper.inner_radius, paper.thickness, paper.alpha, paper.order)
    closed = (C_LIGHT ** 4 / (2.0 * G)
              * np.sum(first ** 2 / paper.conformal_profile(r) * r ** 2)
              * step)

    assert budget.net > 0.0
    assert budget.net == pytest.approx(closed, rel=1e-5)


def test_published_transition_energies(paper):
    """
    Eqs. (15)-(16) of the 1999 paper, given to two significant figures:
    E_II,- = -1.4e30 kg and E_II,+ = 4.9e30 kg.
    """

    budget = paper.transition_energy_budget()

    assert budget.negative_mass == pytest.approx(-1.4e30, rel=0.05)
    assert budget.positive_mass == pytest.approx(4.9e30, rel=0.05)


def test_published_sign_change(paper):
    """The density turns positive at w = 0.981, w = (R~ + D~ - r) / D~."""

    def density_at(w):
        r = paper.inner_radius + paper.thickness * (1.0 - w)
        return float(paper.energy_density(r, 0.0, 0.0))

    assert density_at(0.975) < 0.0
    assert density_at(0.987) > 0.0


def test_published_pocket_size(paper):
    """A 200 m pocket diameter behind a 3e-15 m shift wall."""

    assert 2.0 * paper.pocket_proper_radius() == pytest.approx(200.0)


def test_paper_budget_is_finite_and_the_wall_is_refused(paper):
    """
    The Planck-thin shift wall is out of reach of any grid in r: the
    closed form handles it, the generic routines must refuse it.
    """

    budget = paper.energy_budget_analytic()
    transition = paper.transition_energy_budget()

    assert np.isfinite(budget.negative) and np.isfinite(budget.positive)
    assert budget.negative < transition.negative
    assert budget.positive == transition.positive
    with pytest.raises(ValueError):
        paper.energy_budget()


def test_generic_quadrature_matches_the_closed_form(metric):
    """Independent paths through sqrt(gamma) = B^3, for both signs."""

    analytic = metric.energy_budget_analytic()
    numeric = metric.energy_budget()

    assert numeric.negative == pytest.approx(analytic.negative, rel=1e-4)
    assert numeric.positive == pytest.approx(analytic.positive, rel=1e-3)


def test_density_is_not_sign_definite(metric):
    """The Alcubierre invariant eps <= 0 must not carry over."""

    r = np.linspace(metric.inner_radius, metric.inner_radius
                    + metric.thickness, 20001)
    eps = metric.energy_density(r, 0.0, 0.0)

    assert eps.min() < 0.0
    assert eps.max() > 0.0


def test_no_inflation_reduces_to_alcubierre():
    broeck = BroeckMetric(speed=10.0 * C_LIGHT, alpha=0.0)
    alcubierre = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=broeck.radius,
                                  sigma=broeck.sigma)

    x, y = np.meshgrid(np.linspace(-150.0, 150.0, 61),
                       np.linspace(0.0, 150.0, 31))
    assert np.array_equal(broeck.energy_density(x, y, 0.0),
                          alcubierre.energy_density(x, y, 0.0))
    assert np.all(broeck.conformal_factor(x, y, 0.0) == 1.0)

    budget = broeck.energy_budget_analytic()
    reference = alcubierre.energy_budget_analytic()
    assert budget.negative == reference.negative
    assert budget.positive == 0.0


def test_interface_quantities_see_only_the_shift_wall(metric):
    """
    The ship sits in the pocket but comoves with it, so B drops out of
    dtau/dt; the horizon lies in the shift wall, where B = 1, so it is
    Alcubierre's.
    """

    alcubierre = AlcubierreMetric(speed=metric.speed, radius=metric.radius,
                                  sigma=metric.sigma)

    assert metric.proper_time_rate() == pytest.approx(1.0)
    assert metric.horizon_offset() == pytest.approx(
        alcubierre.horizon_offset(), rel=1e-9
    )


def test_overlapping_regions_are_rejected():
    with pytest.raises(ValueError):
        BroeckMetric(speed=C_LIGHT, radius=100.0, sigma=0.1)


def test_invalid_profile_parameters_are_rejected():
    with pytest.raises(ValueError):
        BroeckMetric(speed=C_LIGHT, alpha=-2.0)


# --- Neck radius scan ---
NECKS = np.logspace(math.log10(3.0e-15), 1.0, 8)


@pytest.fixture(scope="module")
def scan():
    return neck_scaling(NECKS)


def test_neck_scan_starts_from_the_paper(scan, paper):
    """R = 3e-15 m with a 100 m pocket is exactly the 1999 configuration."""

    transition = paper.transition_energy_budget()
    total = paper.energy_budget_analytic()

    assert scan.transition_negative[0] == pytest.approx(transition.negative,
                                                        rel=1e-6)
    assert scan.transition_positive[0] == pytest.approx(transition.positive,
                                                        rel=1e-6)
    assert scan.wall[0] == pytest.approx(total.negative - transition.negative,
                                         rel=1e-9)


def test_wall_energy_scales_with_the_square_of_the_neck(scan):
    slopes = np.diff(np.log(-scan.wall)) / np.diff(np.log(scan.neck_radius))
    assert np.allclose(slopes, 2.0, atol=1e-6)


def test_transition_energy_does_not_depend_on_the_neck(scan):
    """
    With R~ = D~ and (1 + alpha) R~ fixed, the transition region only sees
    the pocket, as long as the pocket is much larger than the neck.
    """

    small = scan.neck_radius < 1.0e-2
    for part in (scan.transition_negative, scan.transition_positive):
        values = part[small]
        assert np.ptp(values) < 0.01 * np.abs(values).mean()


def test_alcubierre_reference_is_a_pocket_sized_bubble(scan):
    reference = AlcubierreMetric(speed=C_LIGHT, radius=100.0,
                                 sigma=1.0 / (1.0e2 * L_PLANCK))

    assert scan.alcubierre == reference.energy_budget_analytic().negative
    assert np.all(scan.total_negative > scan.alcubierre)


def test_neck_larger_than_the_pocket_is_rejected():
    with pytest.raises(ValueError):
        neck_scaling([400.0])
