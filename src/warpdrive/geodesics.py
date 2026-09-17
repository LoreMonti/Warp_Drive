# ==========================================================
# Null geodesics: what the crew sees from the centre of the bubble
#
# For a constant v_s the metric is static in the rest frame of the
# bubble, xi = x - v_s t:
#
#     ds^2 = -c^2 dt^2 + B^2 [ (dxi - b dt)^2 + dy^2 + dz^2 ],
#     b = beta - v_s = -v_s (1 - f),
#
# so a photon obeys Hamilton's equations with a conserved Hamiltonian,
#
#     H = (c/B) |p| + b p_xi,
#     dx/dt = dH/dp,   dp/dt = -dH/dx.
#
# Rays are integrated backwards in time from the ship at the centre. With
# w = c t every quantity below is a length, H is dimensionless and the
# initial momentum is normalised so that H = 1.
#
# An Eulerian observer measures a photon energy proportional to |p|/B.
# At the ship b = 0, far away B = 1 and b = -v_s, so conservation of H
# gives the frequency ratio in closed form,
#
#     E_ship / E_far = 1 - (v_s/c) n_xi,
#
# with n the far-field direction of propagation. Photons with
# n_xi > c/v_s never reach the ship: a superluminal bubble outruns the
# light chasing it, and a ray traced backwards towards the rear stalls
# at the horizon, where its momentum grows exponentially.
#
# For a ship at the centre the sky is axisymmetric about the direction of
# travel, so it is fully described by rays in one meridional plane, as a
# function of the angle between the line of sight and the motion.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
from dataclasses import dataclass

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .constants import C_LIGHT
from .integrators import STOPPED, integrate_adaptive
from .metrics.base import MIN_RELATIVE_WIDTH, WALL_HALF_WIDTH

#: Ray outcomes.
ESCAPED, HORIZON, UNFINISHED = 0, 1, 2


@dataclass
class RayBundle:
    """
    Rays traced backwards from the centre of the bubble, one per line of
    sight in the meridional plane.
    """

    #: Angle between the line of sight and the direction of travel [rad].
    look_angle: np.ndarray

    #: ESCAPED, HORIZON or UNFINISHED for each ray.
    status: np.ndarray

    #: Angle between the direction of the source, far from the bubble,
    #: and the direction of travel [rad]; NaN unless the ray escaped.
    source_angle: np.ndarray

    #: E_ship / E_far, the blueshift of the source; 0 for a ray lost at
    #: the horizon, NaN for an unfinished one.
    frequency_ratio: np.ndarray

    #: Far-field direction of propagation along the motion, n_xi.
    direction_xi: np.ndarray

    #: Final (xi, rho) of each ray [m].
    end_position: np.ndarray

    #: |H - 1| at the end of each ray relative to |p|/B, the size of the
    #: terms that cancel in H: the integration error.
    hamiltonian_drift: np.ndarray

    #: Coordinate time elapsed along each ray, as w = c |t| [m].
    elapsed: np.ndarray

    @property
    def escaped(self):
        return self.status == ESCAPED


def _escape_radius(metric):
    return metric.radius + WALL_HALF_WIDTH / metric.sigma


def ray_hamiltonian(metric, state):
    """H = |p|/B + (b/c) p_xi for states of shape (4, n), w = c t units."""

    xi, rho, p_xi, p_rho = state
    r = np.hypot(xi, rho)
    conformal = np.asarray(metric.conformal_factor(r, 0.0, 0.0), dtype=float)
    drag = (metric.shift(r, 0.0, 0.0) - metric.speed) / C_LIGHT
    return np.hypot(p_xi, p_rho) / conformal + drag * p_xi


def ray_rhs(metric):
    """
    Hamilton's equations for a ray in the meridional plane, for
    `integrate_adaptive`: state (xi, rho, p_xi, p_rho), time w = c t.
    Tracing backwards is done by integrating towards negative w, not by
    changing signs here.
    """

    def rhs(w, state, members):
        xi, rho, p_xi, p_rho = state
        r = np.hypot(xi, rho)
        safe = np.where(r > 0.0, r, 1.0)
        p = np.hypot(p_xi, p_rho)

        conformal = np.asarray(metric.conformal_factor(r, 0.0, 0.0),
                               dtype=float)
        drag = (metric.shift(r, 0.0, 0.0) - metric.speed) / C_LIGHT
        drag_slope = metric.shift_radial_derivative(r) / C_LIGHT / safe
        conformal_slope = metric.conformal_radial_derivative(r) / safe

        # dH/dp and dH/dx, gradients of radial profiles being g'(r) x_i / r
        dxi = p_xi / (conformal * p) + drag
        drho = p_rho / (conformal * p)
        common = -p * conformal_slope / conformal ** 2 + p_xi * drag_slope
        dp_xi = -common * xi
        dp_rho = -common * rho

        return np.array([dxi, drho, dp_xi, dp_rho])

    return rhs


def trace_rays(metric, look_angles, rtol=1.0e-10, atol=1.0e-12,
               min_ratio=1.0e-12, max_steps=200000):
    """
    Trace one backward null ray per line of sight from the centre of the
    bubble, in the meridional plane.

    A ray ESCAPES once it leaves the shift wall, beyond which the metric
    is uniform and the momentum no longer changes; its far-field
    direction and frequency ratio are then final. A ray whose frequency
    ratio falls below `min_ratio` is stopped as lost at the HORIZON: its
    momentum grows exponentially there and the source it comes from is
    invisible. Anything else is UNFINISHED when the step budget runs out.

    Raises ValueError for a wall too thin to resolve in double precision.

    Returns a RayBundle.
    """

    half = WALL_HALF_WIDTH / metric.sigma
    escape = _escape_radius(metric)
    if half < MIN_RELATIVE_WIDTH * escape:
        raise ValueError(
            f"wall of thickness {1.0 / metric.sigma:.3e} m is too thin to "
            f"trace rays through around a radius of {metric.radius:.3e} m"
        )

    angles = np.asarray(look_angles, dtype=float)
    n = angles.size
    centre = float(metric.conformal_factor(0.0, 0.0, 0.0))

    # the photon arriving along the line of sight propagates against it;
    # |p| = B at the centre, where b = 0, makes H = 1
    state0 = np.array([np.zeros(n), np.zeros(n),
                       -centre * np.cos(angles), -centre * np.sin(angles)])

    def stop(w, state, members):
        far = np.hypot(state[0], state[1]) > escape
        lost = np.hypot(state[2], state[3]) * min_ratio > 1.0
        return far | lost

    # a step may not cross more than half a wall thickness: the photon
    # moves at most (1 + v_s/c) per unit of w
    max_step = 0.5 / (metric.sigma * (1.0 + metric.speed / C_LIGHT))
    horizon_time = -max_steps * max_step

    w, state, integration = integrate_adaptive(
        ray_rhs(metric), state0, horizon_time, rtol=rtol, atol=atol,
        first_step=0.01 * max_step, max_step=max_step, max_steps=max_steps,
        stop=stop)

    p = np.hypot(state[2], state[3])
    conformal = np.asarray(
        metric.conformal_factor(np.hypot(state[0], state[1]), 0.0, 0.0),
        dtype=float)
    far = np.hypot(state[0], state[1]) > escape
    status = np.full(n, UNFINISHED)
    status[(integration == STOPPED) & far] = ESCAPED
    status[(integration == STOPPED) & ~far] = HORIZON

    direction_xi = state[2] / p
    escaped = status == ESCAPED
    ratio = np.where(escaped, 1.0 / p, np.nan)
    ratio[status == HORIZON] = 0.0

    return RayBundle(
        look_angle=angles,
        status=status,
        source_angle=np.where(escaped, np.arccos(np.clip(-direction_xi,
                                                         -1.0, 1.0)),
                              np.nan),
        frequency_ratio=ratio,
        direction_xi=np.where(escaped, direction_xi, np.nan),
        end_position=state[:2].copy(),
        hamiltonian_drift=np.abs(ray_hamiltonian(metric, state) - 1.0)
        / np.maximum(1.0, p / conformal),
        elapsed=-w,
    )


def horizon_surface_gravity(metric):
    """
    Exponential rate, per unit of w = c t, at which the momentum of a ray
    held at the rear horizon grows: kappa = (v_s/c) |f'(h)| on the axis,
    with h the horizon offset. It sets how fast the light from behind
    fades and plays the role of a surface gravity. [m^-1]
    """

    offset = metric.horizon_offset()
    if offset is None:
        return None
    return abs(float(metric.shift_radial_derivative(offset))) / C_LIGHT
