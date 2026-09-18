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
# Brightness follows from I_nu / nu^3 being conserved along a ray: surface
# brightness scales as R^4, R = E_ship / E_far, and a point source is
# further magnified by the ratio of solid angles mu of the map from true
# to apparent position, so its flux scales as R^4 mu. The light received
# from an isotropic background is the surface brightness integrated over
# the apparent sky, (1/4pi) \int R^4 dOmega_look, which equals
# (1/4pi) \int R^4 mu dOmega_source: the distortion of the sky enters,
# and there is no closed form for it.
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


@dataclass
class SkyMap:
    """
    Where every source on the sky appears from the centre of the bubble,
    built from a fan of rays. Angles are measured from the direction of
    travel; the azimuth about it is unchanged.
    """

    #: Look angles of the rays that reached a visible source [rad],
    #: increasing.
    look_angle: np.ndarray

    #: True angle of the source seen along each look angle [rad].
    source_angle: np.ndarray

    #: E_ship / E_far along each look angle.
    frequency_ratio: np.ndarray

    #: Largest source angle that can be seen at all [rad]:
    #: arccos(-c / v_s) for a superluminal bubble, pi otherwise.
    visible_limit: float

    def apparent(self, source_angle):
        """
        Look angle at which a source is seen and its frequency ratio;
        NaN for sources beyond the visible limit.
        """

        source = np.asarray(source_angle, dtype=float)
        visible = source <= self.source_angle[-1]
        look = np.interp(source, self.source_angle, self.look_angle)
        ratio = np.interp(source, self.source_angle, self.frequency_ratio)
        return (np.where(visible, look, np.nan),
                np.where(visible, ratio, np.nan))

    def _magnification_samples(self):
        """
        mu on the traced rays, with its limit (d look / d source)^2 on the
        axis, where both sines vanish. The rays are evenly spaced in look
        angle, so the slope is taken as d source / d look on that grid and
        inverted: differentiating along the uneven source angles instead
        loses two orders of magnitude of accuracy.
        """

        slope = 1.0 / np.gradient(self.source_angle, self.look_angle)
        on_axis = self.source_angle < 1.0e-9
        safe = np.where(on_axis, 1.0, self.source_angle)
        mu = np.sin(self.look_angle) / np.sin(safe) * slope
        return np.where(on_axis, slope ** 2, mu)

    def magnification(self, source_angle):
        """
        Ratio of the solid angle a source covers from the ship to the one
        it covers without the bubble,

            mu = (sin theta_look / sin theta_source)
                 (d theta_look / d theta_source),

        NaN beyond the visible limit.
        """

        source = np.asarray(source_angle, dtype=float)
        mu = np.interp(source, self.source_angle,
                       self._magnification_samples())
        return np.where(source <= self.source_angle[-1], mu, np.nan)

    def flux_ratio(self, source_angle):
        """
        Flux of a point source seen from the ship over its flux without
        the bubble, R^4 mu: surface brightness scales as R^4 because
        I_nu / nu^3 is conserved along the ray. NaN beyond the visible
        limit.
        """

        _, ratio = self.apparent(source_angle)
        return ratio ** 4 * self.magnification(source_angle)

    def sky_brightness(self):
        """
        Light received from an isotropic background, over the light
        received without the bubble:

            (1/4pi) \\int R^4 dOmega_look.

        The integral runs over the apparent sky, directly on the traced
        rays. Rays dropped as redshifted below `sky_map`'s threshold
        contribute below that threshold to the fourth power.
        """

        integrand = self.frequency_ratio ** 4 * np.sin(self.look_angle)
        return 0.5 * float(np.trapezoid(integrand, self.look_angle))


def sky_map(metric, n_rays=721, min_ratio=1.0e-6, **trace_options):
    """
    Trace a fan of look angles from 0 to pi and keep the rays that bring
    in visible light, with frequency ratio above `min_ratio`.

    The map from look angle to source angle is increasing, which a check
    here enforces before it is inverted by interpolation.

    Returns a SkyMap.
    """

    rays = trace_rays(metric, np.linspace(0.0, np.pi, n_rays),
                      **trace_options)
    keep = rays.escaped & (rays.frequency_ratio > min_ratio)
    if not np.all(np.diff(rays.source_angle[keep]) > 0.0):
        raise RuntimeError("look angle to source angle is not monotonic")

    ratio = metric.speed / C_LIGHT
    limit = np.arccos(-1.0 / ratio) if ratio > 1.0 else np.pi

    return SkyMap(
        look_angle=rays.look_angle[keep],
        source_angle=rays.source_angle[keep],
        frequency_ratio=rays.frequency_ratio[keep],
        visible_limit=float(limit),
    )



def unlensed_brightness(speed_ratio):
    """
    What `SkyMap.sky_brightness` would be if the bubble shifted every
    frequency but left every star where it is (mu = 1):

        (1/2) \\int R^4 dn_xi = [(1 + u)^5 - max(0, 1 - u)^5] / (10 u),

    over the visible far-field directions -1 < n_xi < min(1, c/v_s), with
    u = v_s / c. It is not the light received, which also carries the
    distortion of the sky; the two differ by 6 % at 10c, and the gap
    measures how much the bubble lenses the background.
    """

    u = float(speed_ratio)
    if u == 0.0:
        return 1.0
    return ((1.0 + u) ** 5 - max(0.0, 1.0 - u) ** 5) / (10.0 * u)
