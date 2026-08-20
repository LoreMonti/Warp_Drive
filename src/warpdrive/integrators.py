# ==========================================================
# Explicit ODE integrators
#
# Hand-rolled rather than pulled from scipy: the package depends only on
# numpy and matplotlib, and the right-hand sides here are cheap enough
# that a fixed-step RK4 vectorised over particles is the fastest thing
# available.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import numpy as np


def rk4_step(rhs, t, state, step_size):
    """
    Advance `state` by one classical fourth-order Runge-Kutta step.

    `rhs(t, state)` must return an array shaped like `state`, so the same
    routine integrates a single trajectory or a whole ensemble at once.
    """

    h = step_size
    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * h, state + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, state + 0.5 * h * k2)
    k4 = rhs(t + h, state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(rhs, state0, t_grid):
    """
    Integrate an initial value problem on a prescribed time grid.

    Returns an array of shape (len(t_grid),) + state0.shape holding the
    state at every requested time, the first row being `state0`.
    """

    state = np.asarray(state0, dtype=float).copy()
    history = np.empty((len(t_grid),) + state.shape, dtype=float)
    history[0] = state

    for i in range(1, len(t_grid)):
        step = t_grid[i] - t_grid[i - 1]
        state = rk4_step(rhs, t_grid[i - 1], state, step)
        history[i] = state

    return history
