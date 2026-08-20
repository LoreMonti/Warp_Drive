# ==========================================================
# Tests for the shape functions
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive.shapes import (
    sech2,
    tanh_top_hat,
    tanh_top_hat_derivative,
    wall_thickness,
)


RADIUS = 100.0
SIGMA = 0.1


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
