# ==========================================================
# Tests for the ODE integrators and the tracer congruence
#
# Author: Lorenzo Monti
# ==========================================================

import numpy as np
import pytest

from warpdrive import AlcubierreMetric, integrate_tracers, make_tracer_grid
from warpdrive.constants import C_LIGHT
from warpdrive.integrators import (
    OUT_OF_STEPS,
    REACHED_END,
    STOPPED,
    integrate,
    integrate_adaptive,
)


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


def test_adaptive_integrator_is_accurate_per_member():
    """
    Harmonic oscillators of different frequencies, integrated together:
    each member picks its own steps and must land on cos(omega t).
    """

    omega = np.array([0.5, 1.0, 7.0, 40.0])

    def rhs(t, y, members):
        return np.array([y[1], -omega[members] ** 2 * y[0]])

    t, state, status = integrate_adaptive(
        rhs, np.array([np.ones(4), np.zeros(4)]), 10.0, rtol=1e-11,
        atol=1e-13, max_steps=200000)

    assert np.all(status == REACHED_END)
    assert np.allclose(t, 10.0)
    assert np.allclose(state[0], np.cos(10.0 * omega), atol=1e-7)


def test_adaptive_integrator_runs_backwards():
    """Members finish at different steps and keep their own rate."""

    rates = np.array([0.3, 2.0])

    def rhs(t, y, members):
        return -rates[members] * y

    t, state, status = integrate_adaptive(rhs, np.array([[1.0, 1.0]]), -2.0)

    assert np.all(status == REACHED_END)
    assert np.allclose(t, -2.0)
    assert np.allclose(state[0], np.exp(2.0 * rates), rtol=1e-7)


def test_adaptive_integrator_freezes_stopped_members():
    """y' = v with different speeds, each member stopping once y > 5."""

    speed = np.array([1.0, 2.0, 0.1])

    def rhs(t, y, members):
        return np.array([speed[members]])

    t, state, status = integrate_adaptive(
        rhs, np.zeros((1, 3)), 10.0, max_step=0.05,
        stop=lambda t, y, members: y[0] > 5.0)

    assert list(status) == [STOPPED, STOPPED, REACHED_END]
    assert np.all(state[0, :2] > 5.0)
    assert np.all(state[0, :2] < 5.0 + 0.05 * speed[:2])
    assert np.allclose(t[:2], state[0, :2] / speed[:2])
    assert state[0, 2] == pytest.approx(1.0)


def test_adaptive_integrator_reports_exhausted_steps():
    t, state, status = integrate_adaptive(
        lambda t, y, members: np.ones_like(y), np.zeros((1, 2)), 1.0,
        max_step=0.01, max_steps=10)

    assert np.all(status == OUT_OF_STEPS)
    assert np.all(t < 1.0)


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
