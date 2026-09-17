# ==========================================================
# Explicit ODE integrators
#
# Hand-rolled rather than pulled from scipy: the package depends only on
# numpy and matplotlib, and the right-hand sides here are cheap enough
# that stepping a whole ensemble at once is the fastest thing available.
#
#   rk4_step, integrate   fixed step, for the tracers
#   integrate_adaptive    Dormand-Prince 5(4), one step size per member,
#                         for null rays crossing thin walls
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


# --- Dormand-Prince 5(4) tableau ---
_DP_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_DP_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0,
     -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
     -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
     11.0 / 84.0),
)
_DP_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
          -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_DP_B4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
          -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)

#: Status codes returned by `integrate_adaptive`, one per member.
REACHED_END, STOPPED, OUT_OF_STEPS = 0, 1, 2


def integrate_adaptive(rhs, state0, t_end, t0=0.0, rtol=1.0e-9,
                       atol=1.0e-12, first_step=None, max_step=np.inf,
                       max_steps=100000, stop=None):
    """
    Integrate an ensemble with the embedded Dormand-Prince 5(4) pair,
    each member on its own adaptive step.

    `state0` has shape (n_variables, n_members). `rhs(t, state, members)`
    receives only the columns of the members still running: `t` holds
    one time per column and `members` their indices in the ensemble, for
    right-hand sides with per-member parameters. Integration runs
    from `t0` towards `t_end`, forwards or backwards.

    A step is accepted when the RMS over variables of the error estimate,
    scaled by atol + rtol |y|, is at most one. `max_step` bounds |h|: an
    error estimate only sees the right-hand side where it is sampled, so
    a step longer than a thin feature can jump over it unnoticed.

    `stop(t, state, members)`, if given, is called after every accepted
    step on the running columns and returns True for members to freeze
    there.

    Returns (t, state, status): the final time and state of every member,
    and REACHED_END, STOPPED or OUT_OF_STEPS for each.
    """

    state = np.array(state0, dtype=float)
    n_members = state.shape[1]
    direction = 1.0 if t_end >= t0 else -1.0

    t = np.full(n_members, float(t0))
    status = np.full(n_members, OUT_OF_STEPS)
    running = np.ones(n_members, dtype=bool)
    if first_step is None:
        first_step = min(abs(t_end - t0) * 1.0e-3, max_step)
    h = np.full(n_members, direction * min(first_step, max_step))

    for _ in range(max_steps):
        if not running.any():
            break

        idx = np.flatnonzero(running)
        y, ti = state[:, idx], t[idx]
        remaining = t_end - ti
        hi = direction * np.minimum(np.abs(h[idx]), np.abs(remaining))

        k = []
        for c_i, a_i in zip(_DP_C, _DP_A):
            stage = y.copy()
            for a, kj in zip(a_i, k):
                if a != 0.0:
                    stage += hi * a * kj
            k.append(rhs(ti + c_i * hi, stage, idx))

        y5 = y + hi * sum(b * kj for b, kj in zip(_DP_B5, k) if b != 0.0)
        y4 = y + hi * sum(b * kj for b, kj in zip(_DP_B4, k) if b != 0.0)

        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error = np.sqrt(np.mean(((y5 - y4) / scale) ** 2, axis=0))
        accepted = error <= 1.0

        with np.errstate(divide="ignore"):
            factor = np.where(error > 0.0,
                              0.9 * error ** -0.2, 5.0)
        factor = np.clip(factor, 0.2, 5.0)
        h[idx] = direction * np.minimum(np.abs(hi) * factor, max_step)

        done_idx = idx[accepted]
        t[done_idx] = ti[accepted] + hi[accepted]
        state[:, done_idx] = y5[:, accepted]

        finished = np.abs(t_end - t[done_idx]) <= 1.0e-12 * max(
            1.0, abs(t_end))
        status[done_idx[finished]] = REACHED_END
        running[done_idx[finished]] = False

        if stop is not None:
            live = done_idx[~finished]
            if live.size:
                halt = np.asarray(stop(t[live], state[:, live], live),
                                  dtype=bool)
                status[live[halt]] = STOPPED
                running[live[halt]] = False

    return t, state, status
