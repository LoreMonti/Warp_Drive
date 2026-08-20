# ==========================================================
# Eulerian congruence: test particles carried by a passing bubble
#
# The Eulerian observers are the ones at rest with respect to the
# spatial slices; their coordinate velocity is the shift vector itself,
#
#     dx/dt = beta^x,   dy/dt = dz/dt = 0.
#
# Integrating them shows two behaviours, both physical:
#
#   - particles with an impact parameter larger than R are dragged
#     forward by the passing wall and then released, slightly displaced;
#   - particles close to the axis are swallowed, because f = 1 inside
#     the bubble forces dx/dt = v_s exactly, and are carried along for
#     the whole trip.
#
# The second one is the "bulldozer problem": a real bubble sweeps up the
# interstellar medium and dumps it, hugely blueshifted, at the
# destination (McMonigal, Lewis & O'Byrne, PRD 85, 064024 (2012)).
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .integrators import integrate


def bubble_centre(t, x_start, speed):
    """Position of the bubble centre at coordinate time t. [m]"""

    return x_start + speed * np.asarray(t, dtype=float)


def integrate_tracers(metric, x0, y0, t_grid, x_start):
    """
    Follow an ensemble of Eulerian observers past a moving bubble.

    Parameters
    ----------
    metric  : a WarpMetric instance
    x0, y0  : initial positions [m], arrays of equal length
    t_grid  : coordinate times to sample [s]
    x_start : position of the bubble centre at t = 0 [m]

    Returns an array of shape (len(t_grid), len(x0)) with the x
    coordinate of every tracer; y is constant by construction.
    """

    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    def rhs(t, x):
        centre = bubble_centre(t, x_start, metric.speed)
        return metric.shift(x - centre, y0, 0.0)

    return integrate(rhs, x0, np.asarray(t_grid, dtype=float))


def make_tracer_grid(metric, rows, n_columns=21, span=3.7):
    """
    Build a symmetric lattice of tracers around the axis.

    `rows` are impact parameters in units of the bubble radius, `span`
    is the half-width of the lattice in the same units.
    """

    rows = np.asarray(rows, dtype=float) * metric.radius
    columns = np.linspace(-span, span, n_columns) * metric.radius

    x0, y0 = [], []
    for row in rows:
        for column in columns:
            for sign in ((1,) if row == 0.0 else (1, -1)):
                x0.append(column)
                y0.append(sign * row)

    return np.array(x0), np.array(y0)
