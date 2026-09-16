# ==========================================================
# Tests for the energy budget and causal structure
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest

from warpdrive import (
    AlcubierreMetric,
    EnergyBudget,
    format_profile,
    profile_mission,
    relativistic_rocket,
)
from warpdrive.constants import C_LIGHT, D_PROXIMA, G, L_PLANCK, LY
from warpdrive.metrics.base import merge_regions


def thin_wall_energy(metric):
    """
    Thin-wall limit of the Alcubierre budget, sigma R >> 1. With
    int sech^4 x dx = 4/3 and int x^2 sech^4 x dx = (pi^2 - 6)/9,

        E = -(c^2 v_s^2 / 12 G) [R^2 sigma / 3 + (pi^2 - 6) / (36 sigma)]
            / tanh^2(sigma R),

    up to terms of order exp(-2 sigma R). Derived independently of the
    quadrature it is compared with.
    """

    R, sigma = metric.radius, metric.sigma
    bracket = R ** 2 * sigma / 3.0 + (math.pi ** 2 - 6.0) / (36.0 * sigma)
    return (-(C_LIGHT ** 2 * metric.speed ** 2 / (12.0 * G)) * bracket
            / math.tanh(sigma * R) ** 2)


def test_analytic_budget_matches_the_generic_quadrature():
    """
    The closed form uses the analytic angular integral 8 pi / 3; the
    generic routine integrates the density on a spherical grid. They are
    independent code paths and must agree, which is the check that keeps
    the base class honest once other metrics are added.
    """

    metric = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0, sigma=0.1)

    analytic = metric.energy_budget_analytic()
    numeric = metric.energy_budget()

    assert numeric.negative == pytest.approx(analytic.negative, rel=1.0e-3)
    assert numeric.positive == 0.0
    assert analytic.positive == 0.0


@pytest.mark.parametrize("radius, sigma", [(100.0, 1.0), (1000.0, 0.1)])
def test_closed_form_matches_the_thin_wall_limit(radius, sigma):
    metric = AlcubierreMetric(speed=C_LIGHT, radius=radius, sigma=sigma)

    assert metric.energy_budget_analytic().negative == pytest.approx(
        thin_wall_energy(metric), rel=1.0e-9
    )


def test_planck_thin_wall_is_integrated_in_the_wall_offset():
    """
    Van Den Broeck's outer wall: R = 3e-15 m, 1e2 Planck lengths thick.
    R / (1/sigma) ~ 2e18 is beyond double precision, so r cannot resolve
    the wall. The closed form integrates over the offset and must still
    reach the thin-wall limit; the generic quadrature must refuse rather
    than return noise.
    """

    radius = 3.0e-15
    sigma = 1.0 / (1.0e2 * L_PLANCK)
    assert radius + 30.0 / sigma == radius

    metric = AlcubierreMetric(speed=C_LIGHT, radius=radius, sigma=sigma)
    budget = metric.energy_budget_analytic()

    assert np.isfinite(budget.negative)
    assert budget.negative == pytest.approx(thin_wall_energy(metric),
                                            rel=1.0e-6)
    with pytest.raises(ValueError):
        metric.energy_budget()


def test_overlapping_regions_are_not_integrated_twice():
    class Duplicated(AlcubierreMetric):
        def energy_regions(self):
            inner, outer = super().energy_regions()[0]
            return [(inner, outer), (0.5 * (inner + outer), outer)]

    single = AlcubierreMetric(speed=C_LIGHT).energy_budget()
    doubled = Duplicated(speed=C_LIGHT).energy_budget()

    assert doubled.negative == pytest.approx(single.negative, rel=1.0e-12)


def test_merge_regions():
    assert merge_regions([(5.0, 6.0), (0.0, 2.0), (1.0, 3.0), (3.0, 4.0)]) \
        == [(0.0, 4.0), (5.0, 6.0)]


def test_energy_budget_arithmetic():
    budget = EnergyBudget(negative=-3.0 * C_LIGHT ** 2,
                          positive=5.0 * C_LIGHT ** 2)

    assert budget.net == pytest.approx(2.0 * C_LIGHT ** 2)
    assert budget.negative_mass == pytest.approx(-3.0)
    assert budget.positive_mass == pytest.approx(5.0)
    assert budget.net_mass == pytest.approx(2.0)


def test_energy_scales_quadratically_with_speed():
    """E ~ v_s^2 R^2 sigma; the speed dependence is exact."""

    slow = AlcubierreMetric(speed=C_LIGHT).energy_budget_analytic().negative
    fast = AlcubierreMetric(
        speed=3.0 * C_LIGHT).energy_budget_analytic().negative

    assert fast == pytest.approx(9.0 * slow, rel=1.0e-9)


def test_energy_scales_with_the_square_of_the_radius():
    """Thin-wall limit: the budget follows the surface, not the volume."""

    small = AlcubierreMetric(speed=C_LIGHT, radius=100.0, sigma=0.1)
    large = AlcubierreMetric(speed=C_LIGHT, radius=1000.0, sigma=0.1)

    ratio = (large.energy_budget_analytic().negative
             / small.energy_budget_analytic().negative)
    assert ratio == pytest.approx(100.0, rel=0.02)


def test_energy_is_negative():
    budget = AlcubierreMetric(speed=C_LIGHT).energy_budget_analytic()

    assert budget.negative < 0.0
    assert budget.positive == 0.0
    assert budget.negative_mass == pytest.approx(
        budget.negative / C_LIGHT ** 2)


def test_horizon_sits_where_the_shape_function_says_it_should():
    """
    The horizon condition reduces to f = 1 - c/v_s for the Alcubierre
    metric. The bisection in the base class knows nothing about that,
    so this ties the generic solver to the analytic result.
    """

    metric = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0, sigma=0.1)
    offset = metric.horizon_offset()

    assert offset is not None
    assert 0.0 < offset < metric.radius
    assert float(metric.shape(offset)) == pytest.approx(
        1.0 - C_LIGHT / metric.speed, abs=1e-7
    )


def test_subluminal_bubbles_have_no_horizon():
    """Below c the crew can still signal the front wall and steer."""

    assert AlcubierreMetric(speed=0.5 * C_LIGHT).horizon_offset() is None


def test_relativistic_rocket_respects_time_dilation():
    """The reference case: the crew ages less than the coordinate clock."""

    tau, t = relativistic_rocket(D_PROXIMA)
    assert 0.0 < tau < t


def test_mission_profile_is_self_consistent():
    metric = AlcubierreMetric(speed=10.0 * C_LIGHT)
    profile = profile_mission(metric, distance=4.2465 * LY)

    assert profile.coordinate_time == pytest.approx(
        profile.distance / metric.speed
    )
    assert profile.proper_time == pytest.approx(profile.coordinate_time)
    assert profile.time_rate == pytest.approx(1.0)
    assert profile.energy.negative < 0.0
    assert profile.energy.positive == 0.0
    assert "E-" in format_profile(profile)
    assert profile.horizon is not None

    # the warp bubble beats the rocket in coordinate time, which is the
    # only comparison that means anything to whoever stayed behind
    assert profile.coordinate_time < profile.rocket_coordinate_time
