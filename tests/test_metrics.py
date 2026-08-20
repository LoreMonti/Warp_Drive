# ==========================================================
# Tests for the metric interface and the Alcubierre metric
#
# Errors in general relativity rarely crash: a wrong sign or a missing
# factor of c yields plausible numbers. These tests pin the invariants
# that would catch that.
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive import AlcubierreMetric
from warpdrive.constants import C_LIGHT


@pytest.fixture
def metric():
    return AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0, sigma=0.1)


def test_shift_matches_the_bubble_at_the_centre(metric):
    """At r_s = 0 the coordinates are dragged at exactly v_s."""

    assert float(metric.shift(0.0, 0.0, 0.0)) == pytest.approx(metric.speed)
    assert float(metric.shift(5.0 * metric.radius, 0.0, 0.0)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_alcubierre_space_is_conformally_trivial(metric):
    """B = 1 distinguishes Alcubierre from the Van Den Broeck variant."""

    assert np.allclose(metric.conformal_factor(np.linspace(0, 300, 10),
                                               0.0, 0.0), 1.0)


def test_expansion_is_antisymmetric(metric):
    """
    theta ~ x/r_s, so contraction ahead and expansion behind are mirror
    images: whatever is destroyed in front is created behind.
    """

    x = np.linspace(1.0, 3.0 * metric.radius, 200)
    ahead = metric.expansion(x, 20.0, 0.0)
    behind = metric.expansion(-x, 20.0, 0.0)

    assert np.allclose(ahead, -behind)


def test_expansion_signs(metric):
    """Space contracts ahead of the ship and expands behind it."""

    assert float(metric.expansion(metric.radius, 0.0, 0.0)) < 0.0
    assert float(metric.expansion(-metric.radius, 0.0, 0.0)) > 0.0
    assert float(metric.expansion(0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_energy_density_violates_the_weak_energy_condition(metric):
    """The density is negative wherever it is non-zero. That is the catch."""

    grid = np.linspace(-3.0 * metric.radius, 3.0 * metric.radius, 120)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    eps = metric.energy_density(X, Y, Z)

    assert np.all(eps <= 0.0)
    assert eps.min() < 0.0


def test_energy_density_vanishes_on_the_axis(metric):
    """
    eps ~ rho^2, so the exotic matter forms a torus rather than a ball:
    the physical reason concept ships are drawn with rings.
    """

    x = np.linspace(-2.0 * metric.radius, 2.0 * metric.radius, 100)
    on_axis = metric.energy_density(x, 0.0, 0.0)
    off_axis = metric.energy_density(0.0, metric.radius, 0.0)

    assert np.allclose(on_axis, 0.0)
    assert float(off_axis) < 0.0


def test_no_time_dilation_at_any_speed():
    """
    The headline claim, checked from the line element rather than
    assumed: inside the bubble beta = v_s, so dtau/dt = 1 exactly.
    """

    for factor in (0.1, 1.0, 10.0, 1000.0):
        metric = AlcubierreMetric(speed=factor * C_LIGHT)
        assert metric.proper_time_rate() == pytest.approx(1.0, abs=1e-12)


def test_superluminal_motion_outside_the_bubble_is_rejected(metric):
    """
    The distinction the whole drive rests on. Moving at 2c through the
    flat exterior, where beta = 0, gives a spacelike worldline and must
    raise; moving at 10c while riding the bubble, where beta = v_s
    cancels it, is perfectly timelike.
    """

    outside = 5.0 * metric.radius
    assert float(metric.shift(outside, 0.0, 0.0)) == pytest.approx(0.0,
                                                                   abs=1e-6)

    with pytest.raises(ValueError):
        metric.proper_time_rate(dx_dt=2.0 * C_LIGHT, x=outside)

    assert metric.proper_time_rate() == pytest.approx(1.0)


def test_superluminal_flag():
    assert AlcubierreMetric(speed=2.0 * C_LIGHT).is_superluminal()
    assert not AlcubierreMetric(speed=0.5 * C_LIGHT).is_superluminal()
