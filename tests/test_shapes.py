# ==========================================================
# Tests for the shape functions
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive.shapes import (
    broeck_volume_profile,
    broeck_volume_profile_derivative,
    broeck_volume_profile_second_derivative,
    sech2,
    tanh_top_hat,
    tanh_top_hat_derivative,
    tanh_top_hat_derivative_from_wall,
    wall_thickness,
)


RADIUS = 100.0
SIGMA = 0.1

# Van Den Broeck profile, in units where the transition region is O(1)
INNER = 1.0
THICKNESS = 1.0
ALPHA = 10.0
ORDER = 80


def test_sech2_does_not_overflow():
    """cosh overflows past ~710; the shape function must not."""

    values = sech2([0.0, 1.0, 100.0, 1.0e4, -1.0e4])
    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(1.0)
    assert np.all(values[2:] == 0.0)


def test_interior_and_exterior_are_flat():
    """f = 1 inside the bubble, f = 0 far outside, f = 1/2 on the wall."""

    assert tanh_top_hat(0.0, RADIUS, SIGMA) == pytest.approx(1.0, abs=1e-8)
    assert tanh_top_hat(RADIUS, RADIUS, SIGMA) == pytest.approx(0.5, abs=1e-8)
    assert tanh_top_hat(4.0 * RADIUS, RADIUS, SIGMA) == pytest.approx(
        0.0, abs=1e-8
    )


def test_shape_is_monotonically_decreasing():
    r = np.linspace(0.0, 4.0 * RADIUS, 2000)
    assert np.all(np.diff(tanh_top_hat(r, RADIUS, SIGMA)) <= 1.0e-12)


def test_derivative_matches_finite_differences():
    """The analytic derivative is what actually enters the stress tensor."""

    r = np.linspace(1.0, 3.0 * RADIUS, 500)
    step = 1.0e-4
    numeric = (
        tanh_top_hat(r + step, RADIUS, SIGMA)
        - tanh_top_hat(r - step, RADIUS, SIGMA)
    ) / (2.0 * step)
    analytic = tanh_top_hat_derivative(r, RADIUS, SIGMA)

    assert np.allclose(numeric, analytic, atol=1.0e-9)


def test_derivative_from_the_wall_matches_the_derivative_in_r():
    """Same function, different variable: must agree wherever r resolves."""

    r = np.linspace(0.0, 4.0 * RADIUS, 2000)
    assert np.allclose(
        tanh_top_hat_derivative_from_wall(r - RADIUS, RADIUS, SIGMA),
        tanh_top_hat_derivative(r, RADIUS, SIGMA),
        rtol=1e-12,
        atol=1e-15,
    )


def test_derivative_is_confined_to_the_wall():
    """All the curvature lives within a few wall thicknesses of r = R."""

    derivative = tanh_top_hat_derivative(
        np.array([0.0, RADIUS, RADIUS + 30.0 / SIGMA]), RADIUS, SIGMA
    )
    assert derivative[0] == pytest.approx(0.0, abs=1e-12)
    assert derivative[1] < -0.4 * SIGMA
    assert derivative[2] == pytest.approx(0.0, abs=1e-12)


def test_wall_thickness():
    assert wall_thickness(SIGMA) == pytest.approx(1.0 / SIGMA)


# --- Van Den Broeck volume profile ---
def _broeck(function, r, **overrides):
    parameters = dict(inner_radius=INNER, thickness=THICKNESS, alpha=ALPHA,
                      order=ORDER)
    parameters.update(overrides)
    return function(r, **parameters)


def test_pocket_and_exterior_are_flat():
    """B = 1 + alpha in the pocket, B = 1 outside, both edges exact."""

    r = np.array([0.0, 0.5 * INNER, INNER, INNER + THICKNESS, 10.0])
    values = _broeck(broeck_volume_profile, r)

    assert values[:3] == pytest.approx(1.0 + ALPHA, abs=1e-12)
    assert values[3:] == pytest.approx(1.0, abs=1e-12)


def test_volume_profile_is_monotonically_decreasing():
    r = np.linspace(0.0, INNER + 2.0 * THICKNESS, 4000)
    assert np.all(np.diff(_broeck(broeck_volume_profile, r)) <= 1.0e-12)


def test_volume_profile_derivatives_match_finite_differences():
    """Both derivatives enter the energy density of the B region."""

    r = np.linspace(INNER + 0.01, INNER + THICKNESS - 0.01, 400)
    step = 1.0e-5

    value = lambda q: _broeck(broeck_volume_profile, q)
    first = lambda q: _broeck(broeck_volume_profile_derivative, q)

    numeric_first = (value(r + step) - value(r - step)) / (2.0 * step)
    numeric_second = (first(r + step) - first(r - step)) / (2.0 * step)
    second = _broeck(broeck_volume_profile_second_derivative, r)

    # the n = 80 polynomial varies fast, so the O(step^2) truncation error
    # is judged against the peak of each derivative
    assert np.allclose(numeric_first, first(r), rtol=1e-5,
                       atol=1e-5 * np.abs(first(r)).max())
    assert np.allclose(numeric_second, second, rtol=1e-5,
                       atol=1e-5 * np.abs(second).max())


def test_volume_profile_derivatives_vanish_outside_the_transition():
    r = np.array([0.0, 0.5 * INNER, INNER + THICKNESS, 10.0])

    assert np.all(_broeck(broeck_volume_profile_derivative, r) == 0.0)
    assert np.all(_broeck(broeck_volume_profile_second_derivative, r) == 0.0)


def test_volume_profile_edges():
    """
    dB/dr is continuous at both edges and d^2B/dr^2 at the outer one; at
    the inner edge d^2B/dr^2 jumps to -alpha n (n-1) / D~^2, as the
    docstring states. Pinned so that a change of profile cannot silently
    alter the continuity class.
    """

    eps = 1.0e-9
    inner_side = INNER + eps
    outer_side = INNER + THICKNESS - eps

    # at the inner edge dB/dr vanishes linearly, with slope equal to the
    # jump of the second derivative
    slope = ALPHA * ORDER * (ORDER - 1) / THICKNESS ** 2
    assert abs(_broeck(broeck_volume_profile_derivative, inner_side)) <= \
        1.001 * slope * eps
    assert _broeck(broeck_volume_profile_derivative, outer_side) == \
        pytest.approx(0.0, abs=1e-12)
    assert _broeck(broeck_volume_profile_second_derivative, outer_side) == \
        pytest.approx(0.0, abs=1e-12)

    jump = -slope
    assert _broeck(broeck_volume_profile_second_derivative, inner_side) == \
        pytest.approx(jump, rel=1e-5)


def test_volume_profile_scales_with_the_transition_thickness():
    """
    B depends on r only through w, so shrinking every length by a factor k
    leaves B unchanged and multiplies its n-th derivative by k^n.
    """

    k = 1.0e15
    r = np.linspace(INNER + 0.05, INNER + THICKNESS - 0.05, 50)
    small = dict(inner_radius=INNER / k, thickness=THICKNESS / k)

    assert np.allclose(_broeck(broeck_volume_profile, r / k, **small),
                       _broeck(broeck_volume_profile, r), rtol=1e-12)
    assert np.allclose(
        _broeck(broeck_volume_profile_derivative, r / k, **small),
        k * _broeck(broeck_volume_profile_derivative, r), rtol=1e-10)
    assert np.allclose(
        _broeck(broeck_volume_profile_second_derivative, r / k, **small),
        k ** 2 * _broeck(broeck_volume_profile_second_derivative, r),
        rtol=1e-10)


def test_volume_profile_at_the_published_parameters_is_finite():
    """alpha = 1e17 over a femtometre: nothing may overflow."""

    parameters = dict(inner_radius=1.0e-15, thickness=1.0e-15, alpha=1.0e17)
    r = np.linspace(0.0, 3.0e-15, 10001)

    for function in (broeck_volume_profile,
                     broeck_volume_profile_derivative,
                     broeck_volume_profile_second_derivative):
        assert np.all(np.isfinite(_broeck(function, r, **parameters)))


@pytest.mark.parametrize(
    "overrides",
    [dict(order=2), dict(order=80.5), dict(alpha=-1.0),
     dict(inner_radius=0.0), dict(thickness=-1.0)],
)
def test_volume_profile_rejects_invalid_parameters(overrides):
    with pytest.raises(ValueError):
        _broeck(broeck_volume_profile, 1.5, **overrides)
