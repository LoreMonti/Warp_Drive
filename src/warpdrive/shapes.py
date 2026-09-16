# ==========================================================
# Shape functions for warp bubble metrics
#
# A shape function interpolates between the flat interior of the bubble
# and the flat exterior. Every warp metric in this package is built out
# of one or more of them:
#
#   tanh_top_hat            Alcubierre's f(r), which carries the shift
#   broeck_volume_profile   Van Den Broeck's B(r), which inflates volume
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


def _broeck_coordinate(r, inner_radius, thickness, order, alpha):
    """
    Validate the parameters of the Van Den Broeck profile and map r onto
    the transition coordinate

        w = (R~ + D~ - r) / D~,

    which runs from w = 0 on the outer edge of the transition region to
    w = 1 on its inner edge. Returns (w, mask of the open region 0 < w < 1).
    """

    if int(order) != order or order < 3:
        raise ValueError(f"order must be an integer >= 3, got {order}")
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1 so that B > 0, got {alpha}")
    if inner_radius <= 0.0 or thickness <= 0.0:
        raise ValueError("inner_radius and thickness must be positive")

    r = np.asarray(r, dtype=float)
    w = (inner_radius + thickness - r) / thickness
    return w, (w > 0.0) & (w < 1.0)


def broeck_volume_profile(r, inner_radius, thickness, alpha, order=80):
    """
    Van Den Broeck's conformal factor B(r),

        B = 1 + alpha [n w^(n-1) - (n-1) w^n],   w = (R~ + D~ - r) / D~,

    clamped to B = 1 + alpha for r <= R~ (the pocket) and B = 1 for
    r >= R~ + D~ (the ordinary Alcubierre region). Space inside the
    pocket is inflated by the factor 1 + alpha, so a coordinate radius R~
    encloses a proper radius (1 + alpha) R~.

    The polynomial is the one of the 1999 paper (eqs. 12-13). A tanh or a
    sine step with alpha ~ 1e17 packs a huge d^2B/dr^2 into a tiny region
    and drives the curvature radius below the Planck length; this
    polynomial instead has its first n - 2 derivatives vanishing at the
    outer edge.

    Reference: C. Van Den Broeck, Class. Quantum Grav. 16, 3973 (1999).
    """

    w, _ = _broeck_coordinate(r, inner_radius, thickness, order, alpha)
    w = np.clip(w, 0.0, 1.0)
    n = order
    return 1.0 + alpha * (n * w ** (n - 1) - (n - 1) * w ** n)


def broeck_volume_profile_derivative(r, inner_radius, thickness, alpha,
                                     order=80):
    """
    Radial derivative dB/dr of `broeck_volume_profile`,

        dB/dr = -(alpha / D~) n (n-1) w^(n-2) (1 - w),

    zero outside the transition region and continuous at both of its
    edges.
    """

    w, inside = _broeck_coordinate(r, inner_radius, thickness, order, alpha)
    n = order
    wc = np.where(inside, w, 0.0)
    value = -(alpha / thickness) * n * (n - 1) * wc ** (n - 2) * (1.0 - wc)
    return np.where(inside, value, 0.0)


def broeck_volume_profile_second_derivative(r, inner_radius, thickness,
                                            alpha, order=80):
    """
    Second radial derivative d^2B/dr^2 of `broeck_volume_profile`,

        d^2B/dr^2 = (alpha / D~^2) n (n-1) w^(n-3) [(n-2) - (n-1) w].

    It vanishes smoothly at the outer edge (w = 0) but not at the inner
    one: approaching r = R~ from outside it tends to
    -alpha n (n-1) / D~^2, and inside the pocket it is exactly zero. B is
    therefore C^1 there, not C^2, and the energy density jumps by a finite
    amount across r = R~. The jump carries no surface layer, so volume
    integrals of the energy are unaffected.
    """

    w, inside = _broeck_coordinate(r, inner_radius, thickness, order, alpha)
    n = order
    wc = np.where(inside, w, 0.0)
    value = (
        (alpha / thickness ** 2)
        * n * (n - 1)
        * wc ** (n - 3)
        * ((n - 2) - (n - 1) * wc)
    )
    return np.where(inside, value, 0.0)


def wall_thickness(sigma):
    """Characteristic thickness of the bubble wall, ~ 1/sigma. [m]"""

    return 1.0 / sigma
