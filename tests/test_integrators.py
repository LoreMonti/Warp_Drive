# ==========================================================
# Tests for the ODE integrators and the tracer congruence
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive import AlcubierreMetric, integrate_tracers, make_tracer_grid
from warpdrive.constants import C_LIGHT
from warpdrive.integrators import integrate


def test_rk4_is_fourth_order():
    """Halving the step must cut the error by roughly a factor of 16."""

    def rhs(t, y):
        return -y

    errors = []
    for n in (11, 21):
        t_grid = np.linspace(0.0, 1.0, n)
        history = integrate(rhs, np.array([1.0]), t_grid)
        errors.append(abs(history[-1, 0] - np.exp(-1.0)))

    assert errors[1] < errors[0] / 10.0


def test_integrate_handles_an_ensemble():
    """The same routine must advance a whole lattice of particles."""

    def rhs(t, y):
        return np.full_like(y, 2.0)

    history = integrate(rhs, np.zeros(5), np.linspace(0.0, 1.0, 11))
    assert history.shape == (11, 5)
    assert np.allclose(history[-1], 2.0)


@pytest.fixture
def metric():
    return AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0, sigma=0.1)


def test_distant_particles_are_undisturbed(metric):
    """Outside the wall the shift vanishes and nothing happens."""

    t_grid = np.linspace(0.0, 8.0 * metric.radius / metric.speed, 60)
    x0 = np.array([0.0])
    y0 = np.array([8.0 * metric.radius])

    traj = integrate_tracers(metric, x0, y0, t_grid, -4.0 * metric.radius)
    assert traj[-1, 0] == pytest.approx(x0[0], abs=1e-6)


def test_on_axis_particles_are_captured(metric):
    """
    Inside the bubble f = 1 forces dx/dt = v_s, so a particle on the axis
    is swallowed and carried along instead of being released. This is the
    bulldozer problem, and it must show up in the integration.
    """

    t_grid = np.linspace(0.0, 8.0 * metric.radius / metric.speed, 400)
    x_start = -4.0 * metric.radius

    traj = integrate_tracers(metric, np.array([0.0]), np.array([0.0]),
                             t_grid, x_start)
    final_centre = x_start + metric.speed * t_grid[-1]

    # the particle ends up inside the bubble, not back where it started
    assert abs(traj[-1, 0] - final_centre) < metric.radius
    assert traj[-1, 0] > 3.0 * metric.radius


def test_tracer_grid_is_symmetric(metric):
    x0, y0 = make_tracer_grid(metric, rows=(0.0, 1.0), n_columns=5)

    assert len(x0) == len(y0)
    assert np.sum(y0 == 0.0) == 5
    assert np.sum(y0 > 0.0) == np.sum(y0 < 0.0)
