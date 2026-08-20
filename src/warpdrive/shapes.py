# ==========================================================
# Shape functions for warp bubble metrics
#
# A shape function interpolates between the flat interior of the bubble
# and the flat exterior. Every warp metric in this package is built out
# of one or more of them.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math

# --- Third-party imports ---
import numpy as np


def sech2(x):
    """
    Numerically safe hyperbolic secant squared, sech^2(x) = 1 / cosh^2(x).

    cosh overflows for |x| > ~710, while sech^2 is already
    indistinguishable from zero well before that; the large-argument
    branch is therefore short-circuited to exactly zero.
    """

    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    mask = x < 40.0
    out[mask] = 1.0 / np.cosh(x[mask]) ** 2
    return out


def tanh_top_hat(r, radius, sigma):
    """
    Alcubierre's top-hat shape function,

        f(r) = [tanh(sigma (r + R)) - tanh(sigma (r - R))]
               / [2 tanh(sigma R)]

    with f(0) = 1, f(R) = 1/2 and f -> 0 for r >> R. The transition
    happens over a wall of thickness ~ 1/sigma.
    """

    r = np.asarray(r, dtype=float)
    return (np.tanh(sigma * (r + radius)) - np.tanh(sigma * (r - radius))) / (
        2.0 * math.tanh(sigma * radius)
    )


def tanh_top_hat_derivative(r, radius, sigma):
    """
    Radial derivative df/dr of `tanh_top_hat`.

    This is where all of the curvature lives: it is non-zero only inside
    the bubble wall, and it is the only part of the shape function that
    appears in the stress-energy tensor.
    """

    r = np.asarray(r, dtype=float)
    return (
        sigma
        * (sech2(sigma * (r + radius)) - sech2(sigma * (r - radius)))
        / (2.0 * math.tanh(sigma * radius))
    )


def wall_thickness(sigma):
    """Characteristic thickness of the bubble wall, ~ 1/sigma. [m]"""

    return 1.0 / sigma
