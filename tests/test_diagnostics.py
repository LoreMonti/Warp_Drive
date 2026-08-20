# ==========================================================
# Tests for the energy budget and causal structure
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive import AlcubierreMetric, profile_mission, relativistic_rocket
from warpdrive.constants import C_LIGHT, D_PROXIMA, LY


def test_analytic_budget_matches_the_generic_quadrature():
    """
    The closed form uses the analytic angular integral 8 pi / 3; the
    generic routine integrates the density on a spherical grid. They are
    independent code paths and must agree, which is the check that keeps
    the base class honest once other metrics are added.
    """

    metric = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0, sigma=0.1)

    analytic, _ = metric.exotic_energy_analytic()
    numeric, _ = metric.total_exotic_energy()

    assert numeric == pytest.approx(analytic, rel=1.0e-3)


def test_energy_scales_quadratically_with_speed():
    """E ~ v_s^2 R^2 sigma; the speed dependence is exact."""

    slow = AlcubierreMetric(speed=C_LIGHT).exotic_energy_analytic()[0]
    fast = AlcubierreMetric(speed=3.0 * C_LIGHT).exotic_energy_analytic()[0]

    assert fast == pytest.approx(9.0 * slow, rel=1.0e-9)


def test_energy_scales_with_the_square_of_the_radius():
    """Thin-wall limit: the budget follows the surface, not the volume."""

    small = AlcubierreMetric(speed=C_LIGHT, radius=100.0, sigma=0.1)
    large = AlcubierreMetric(speed=C_LIGHT, radius=1000.0, sigma=0.1)

    ratio = (large.exotic_energy_analytic()[0]
             / small.exotic_energy_analytic()[0])
    assert ratio == pytest.approx(100.0, rel=0.02)


def test_energy_is_negative():
    energy, mass = AlcubierreMetric(speed=C_LIGHT).exotic_energy_analytic()
    assert energy < 0.0
    assert mass == pytest.approx(energy / C_LIGHT ** 2)


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
    assert profile.energy < 0.0
    assert profile.horizon is not None

    # the warp bubble beats the rocket in coordinate time, which is the
    # only comparison that means anything to whoever stayed behind
    assert profile.coordinate_time < profile.rocket_coordinate_time
