# ==========================================================
# Phonons in a two-dimensional condensate at rest
#
# Low-energy phonons of a Bose-Einstein condensate with density n(r) and
# speed of sound c_s(r) see, in 2+1 dimensions, the acoustic metric
#
#     ds^2 = (n / c_s)^2 [ -c_s^2 dt^2 + dr^2 + r^2 dphi^2 ].
#
# Rays ignore the conformal factor and follow the geodesics of the
# optical metric dsigma^2 = (dr^2 + r^2 dphi^2) / c_s^2, whose length is
# the travel time. Waves do not: in 2+1 a scalar is not conformally
# invariant, and (n / c_s) enters the wave equation.
#
# This is the tool section 5 needs for an acoustic pocket. It is first
# checked on a published experiment: Viermann et al., Nature 611, 260
# (2022), where a harmonically trapped 2D condensate approximates a
# space of negative curvature.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .integrators import integrate_adaptive


class ThomasFermiDisc:
    """
    A two-dimensional condensate in a harmonic trap, in the Thomas-Fermi
    limit: n = n_0 (1 - r^2/R^2) and, since c_s^2 = g n / m,

        c_s(r) = c_0 sqrt(1 - r^2 / R^2),   r < R.

    Its optical metric has Gaussian curvature

        K(r) = c_s^2 lap(ln c_s) = -2 c_0^2 / (R^2 - r^2)   [s^-2],

    negative everywhere, -2 c_0^2 / R^2 at the centre and diverging at
    the edge: a space of constant negative curvature only near the
    centre, which is the sense in which Viermann et al. call it
    hyperbolic.

    Parameters
    ----------
    sound_speed : c_0, speed of sound at the centre [m s^-1]
    radius      : Thomas-Fermi radius R [m]
    density     : n_0, areal density at the centre [m^-2]
    """

    def __init__(self, sound_speed, radius, density=1.0):
        if sound_speed <= 0.0 or radius <= 0.0:
            raise ValueError("need a positive speed of sound and radius")
        self.sound_speed = float(sound_speed)
        self.radius = float(radius)
        self.density = float(density)

    @classmethod
    def from_viermann_2022(cls):
        """
        The potassium-39 condensate of Viermann et al. (2022): speed of
        sound 1.2 um/ms at the centre, Thomas-Fermi radius 25 um, central
        density 1.3e9 cm^-2.
        """

        return cls(sound_speed=1.2e-3, radius=25.0e-6, density=1.3e13)

    # --- Profiles ---
    def speed(self, r):
        """c_s(r) [m s^-1], zero at and beyond the edge."""

        u = 1.0 - (np.asarray(r, dtype=float) / self.radius) ** 2
        return self.sound_speed * np.sqrt(np.clip(u, 0.0, None))

    def speed_derivative(self, r):
        """dc_s/dr [s^-1], for r < R."""

        r = np.asarray(r, dtype=float)
        return -self.sound_speed ** 2 * r / (self.radius ** 2
                                             * self.speed(r))

    def number_density(self, r):
        """n(r) = n_0 (1 - r^2/R^2) [m^-2]."""

        u = 1.0 - (np.asarray(r, dtype=float) / self.radius) ** 2
        return self.density * np.clip(u, 0.0, None)

    def conformal_factor(self, r):
        """
        n / c_s, the factor whose square multiplies the 2+1 acoustic
        metric. Rays do not see it; waves do. For a Thomas-Fermi disc it
        is (n_0 / c_0) sqrt(1 - r^2/R^2). [s m^-3]
        """

        return self.number_density(r) / self.speed(r)

    def gaussian_curvature(self, r):
        """K(r) = -2 c_0^2 / (R^2 - r^2) of the optical metric [s^-2]."""

        r = np.asarray(r, dtype=float)
        return -2.0 * self.sound_speed ** 2 / (self.radius ** 2 - r ** 2)

    # --- Rays ---
    def radial_travel_time(self, start, end):
        """
        Time for a ray to run radially from r = start to r = end, the
        length of the optical metric along the radius,

            t = (R / c_0) [arcsin(start / R) - arcsin(end / R)]   [s].
        """

        R = self.radius
        return R / self.sound_speed * (math.asin(start / R)
                                       - math.asin(end / R))

    def trace_rays(self, positions, directions, duration, rtol=1.0e-10):
        """
        Phonon rays from the eikonal Hamiltonian w = c_s(r) |p|, in lab
        time t:

            dx/dt = c_s p / |p|,   dp/dt = -|p| grad c_s.

        w and the angular momentum x p_y - y p_x are conserved, so every
        ray is a geodesic of the optical metric.

        positions  : (n, 2) starting points [m]
        directions : (n, 2) starting directions, any length
        duration   : integration time [s]
        Returns the final (n, 2) positions and (n, 2) momenta, with |p| = 1
        initially.
        """

        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        momenta = directions / np.linalg.norm(directions, axis=1)[:, None]
        state0 = np.vstack([positions.T, momenta.T])

        def rhs(t, state, members):
            x, y, px, py = state
            r = np.hypot(x, y)
            safe = np.where(r > 0.0, r, 1.0)
            c = self.speed(r)
            dc = np.where(r > 0.0, self.speed_derivative(r), 0.0)
            norm = np.hypot(px, py)
            return np.array([c * px / norm, c * py / norm,
                             -norm * dc * x / safe, -norm * dc * y / safe])

        _, state, _ = integrate_adaptive(rhs, state0, duration, rtol=rtol,
                                         atol=1.0e-14 * self.radius,
                                         first_step=duration * 1.0e-4)
        return state[:2].T, state[2:].T


# --- Waves: the acoustic pocket as a cavity ---
def cylindrical_riccati(mmax, x):
    """
    hat j_m(x) = sqrt(pi x / 2) J_m(x) and hat y_m(x) = sqrt(pi x / 2)
    Y_m(x), with their x-derivatives, for m = 0 .. mmax and x > 0: the
    free radial solutions u of a 2D wave in the Liouville form, with
    Wronskian hat j hat y' - hat y hat j' = 1.

    J_m by downward recurrence normalised with J_0 + 2 sum J_2k = 1
    (Miller), Y_m upward from Y_0, Y_1 (mpmath), stable for both.

    Returns (j, y, dj, dy), each of shape (mmax + 1,) + x.shape.
    """

    import mpmath

    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Bessel functions here need x > 0")
    flat = x.ravel()

    start = 2 * ((mmax + int(np.max(flat)) + 40) // 2) + 2
    J = np.zeros((mmax + 1, flat.size))
    norm = np.zeros(flat.size)
    above, current = np.zeros(flat.size), np.full(flat.size, 1.0e-300)
    for m in range(start, 0, -1):
        above, current = current, 2.0 * m / flat * current - above
        index = m - 1
        if index <= mmax:
            J[index] = current
        norm += current * (2.0 if index % 2 == 0 and index > 0 else
                           1.0 if index == 0 else 0.0)
        big = np.abs(current) > 1.0e250
        if np.any(big):
            J[:, big] *= 1.0e-250
            norm[big] *= 1.0e-250
            above[big] *= 1.0e-250
            current[big] *= 1.0e-250
    J /= norm

    Y = np.empty((max(mmax, 1) + 1, flat.size))
    Y[0] = [float(mpmath.bessely(0, v)) for v in flat]
    Y[1] = [float(mpmath.bessely(1, v)) for v in flat]
    with np.errstate(over="ignore", invalid="ignore"):
        for m in range(1, mmax):
            Y[m + 1] = 2.0 * m / flat * Y[m] - Y[m - 1]
    Y = Y[:mmax + 1]

    # Z_m' = Z_{m-1} - (m/x) Z_m, with Z_{-1} = -Z_1
    ms = np.arange(mmax + 1)[:, None]
    Jlow = np.vstack([-J[1:2] if mmax else -_j1(flat), J[:-1]])
    Ylow = np.vstack([-Y[1:2] if mmax else -_y1(flat), Y[:-1]])
    with np.errstate(over="ignore", invalid="ignore"):
        dJ = Jlow - ms / flat * J
        dY = Ylow - ms / flat * Y
        scale = np.sqrt(0.5 * np.pi * flat)
        dscale = 0.5 * scale / flat
        out = (scale * J, scale * Y, dscale * J + scale * dJ,
               dscale * Y + scale * dY)
    shape = (mmax + 1,) + x.shape
    return tuple(f.reshape(shape) for f in out)


def _j1(x):
    import mpmath
    return np.array([[float(mpmath.besselj(1, v)) for v in x]])


def _y1(x):
    import mpmath
    return np.array([[float(mpmath.bessely(1, v)) for v in x]])


class AcousticPocket:
    """
    An acoustic Van Den Broeck pocket in a 2D condensate at rest: the
    speed of sound follows the conformal factor of section 1,

        c_s(r) = c_out / B(r),   B = 1 + alpha inside r < R~, 1 beyond
        R~ + D~, Van Den Broeck's polynomial in between,

    so the optical areal radius A = r / c_s = B r / c_out is the areal
    radius of section 3 in units of time, and rays coincide with light
    rays. The density is n proportional to B^(-2 nu): nu = 1 for a
    uniform coupling (c_s^2 = g n / m, slow sound from low density),
    nu = 0 for uniform density and a coupling tuned in space.

    For phi = exp(-i w t) exp(i m varphi) R(r), with optical distance
    ds = dr / c_s and R = u / sqrt(q), q = r n / c_s = A n,

        u'' + [w^2 - m^2 / A^2 - (sqrt q)'' / sqrt q] u = 0,

    primes in s. The density enters through q: the conformal factor
    n / c_s of the 2+1 acoustic metric, which does not drop out of the
    wave equation as it would in 3+1 for a conformal field.

    Parameters
    ----------
    metric  : BroeckMetric giving R~, D~, alpha and the order n of B
    c_out   : speed of sound outside the pocket [m s^-1]
    nu      : density exponent, 1 (uniform coupling) or 0 (uniform density)
    n_steps : RK4 steps across the transition
    """

    def __init__(self, metric, c_out, nu=1.0, n_steps=4000):
        self.metric = metric
        self.c_out = float(c_out)
        self.nu = float(nu)
        self.n_steps = int(n_steps)
        self.inner = metric.inner_radius
        self.outer = metric.inner_radius + metric.thickness
        self.c_in = self.c_out / (1.0 + metric.alpha)

        h = metric.thickness / self.n_steps
        r = self.inner + 0.5 * h * np.arange(2 * self.n_steps + 1)
        # B'' jumps at R~: take the limit from the transition side
        r[0] += 1.0e-9 * h
        _, B, B1, B2 = metric._profiles(r)
        c = self.c_out / B
        a, b = B1 / B, B2 / B
        # log-derivatives of c and n, from c = c_out / B and n ~ B^(-2 nu)
        c_r, c_rr = -a, 2.0 * a ** 2 - b
        n_r = -2.0 * self.nu * a
        n_rr = (2.0 * self.nu * (2.0 * self.nu + 1.0) * a ** 2
                - 2.0 * self.nu * b)
        # lambda = ln sqrt(q) = (ln r + ln n - ln c) / 2
        lam_r = 0.5 * (1.0 / r + n_r - c_r)
        lam_rr = 0.5 * (-1.0 / r ** 2 + n_rr - n_r ** 2 - c_rr + c_r ** 2)
        self._h = h
        self._c = c
        self._areal = r / c
        self._curvature = c ** 2 * (lam_rr + lam_r ** 2) + c ** 2 * c_r * lam_r

    def sqrt_q(self, r):
        """sqrt(q) up to a constant: q = r n / c_s with n ~ B^(-2 nu)."""

        B = self.metric.conformal_profile(r)
        return np.sqrt(np.asarray(r) * B ** (1.0 - 2.0 * self.nu))

    def throat(self):
        """Optical areal radius A_min = min (B r) / c_out [s]."""

        _, areal = self.metric.throat()
        return areal / self.c_out

    def round_trip(self, ms, omegas):
        """Time to cross the pocket to the turning point and back [s]."""

        ms = np.asarray(ms, dtype=float)[:, None]
        omegas = np.asarray(omegas, dtype=float)[None, :]
        edge = self.inner / self.c_in
        turning = np.minimum(ms / omegas, edge)
        return 2.0 * np.sqrt(edge ** 2 - turning ** 2)

    def _carry(self, ms, omegas, u, du):
        lam = (np.asarray(ms, dtype=float) ** 2)[:, None]
        w2 = (np.asarray(omegas) ** 2)[None, :]
        h = self._h

        def rhs(i, u, du):
            potential = lam / self._areal[i] ** 2 + self._curvature[i]
            return du / self._c[i], (potential - w2) * u / self._c[i]

        for n in range(self.n_steps):
            i = 2 * n
            a1, b1 = rhs(i, u, du)
            a2, b2 = rhs(i + 1, u + 0.5 * h * a1, du + 0.5 * h * b1)
            a3, b3 = rhs(i + 1, u + 0.5 * h * a2, du + 0.5 * h * b2)
            a4, b4 = rhs(i + 2, u + h * a3, du + h * b3)
            u = u + h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            du = du + h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
        return u, du

    def transfer(self, ms, omegas):
        """
        Exterior coefficients (a, c) of the interior solutions hat j_m and
        hat y_m of w s, s = r / c_in, written as a hat j_m(w A) +
        c hat y_m(w A) outside. Returns (a_j, c_j, a_y, c_y).
        """

        ms = np.asarray(ms, dtype=int)
        omegas = np.asarray(omegas, dtype=float)
        mmax = int(ms.max())
        j, y, dj, dy = cylindrical_riccati(mmax, omegas * self.inner
                                           / self.c_in)
        u0 = np.concatenate([j[ms], y[ms]])
        du0 = np.concatenate([omegas * dj[ms], omegas * dy[ms]])
        with np.errstate(over="ignore", invalid="ignore"):
            u, du = self._carry(np.concatenate([ms, ms]), omegas, u0, du0)
            J, Y, dJ, dY = cylindrical_riccati(mmax, omegas * self.outer
                                               / self.c_out)
            J, Y, dJ, dY = (np.concatenate([f[ms], f[ms]])
                            for f in (J, Y, dJ, dY))
            a = u * dY - Y * du / omegas
            c = J * du / omegas - u * dJ
        n = len(ms)
        return a[:n], c[:n], a[n:], c[n:]

    def interior_weight(self, ms, omegas):
        """1 / (a_j^2 + c_j^2): intensity inside per unit incident one."""

        a, c, _, _ = self.transfer(ms, omegas)
        with np.errstate(over="ignore", invalid="ignore"):
            weight = 1.0 / (a ** 2 + c ** 2)
        return np.nan_to_num(weight, nan=0.0, posinf=0.0)

    def crossing(self, ms, omegas):
        """(I, O) for a wave entering the pocket and never coming back."""

        a_j, c_j, a_y, c_y = self.transfer(ms, omegas)
        alpha, gamma = a_j - 1j * a_y, c_j - 1j * c_y
        return 0.5 * (alpha + 1j * gamma), 0.5 * (alpha - 1j * gamma)

    def transmission(self, ms, omegas):
        """Single-pass transmission Gamma_m of the transition region."""

        incoming, _ = self.crossing(ms, omegas)
        with np.errstate(over="ignore", invalid="ignore"):
            gamma = 1.0 / np.abs(incoming) ** 2
        return np.nan_to_num(gamma, nan=0.0, posinf=0.0)

    # --- What an experiment sees ---
    def measured_spectrum(self, omegas, duration, weight=None):
        """
        Interior s-wave intensity at the centre, |H|^2 = 1 / N_0, as a
        record of finite duration T resolves it: smoothed with the
        Fejer kernel sinc^2(w T / 2), the expected periodogram of a
        stationary signal observed for a time T. The comb appears only
        once T exceeds about one round trip, and reaches the
        Fabry-Perot contrast for T much longer than it.

        omegas   : uniform grid [rad s^-1], wide enough for the kernel
        duration : record length T [s]
        weight   : 1 / N_0 on that grid, if already computed
        """

        omegas = np.asarray(omegas, dtype=float)
        if weight is None:
            weight = self.interior_weight([0], omegas)[0]
        kernel = np.sinc((omegas[None, :] - omegas[:, None]) * duration
                         / (2.0 * np.pi)) ** 2
        # NumPy 2 on macOS Accelerate flags spurious floating-point errors
        # in matmul even for finite inputs; the result is checked finite
        with np.errstate(all="ignore"):
            smoothed = kernel @ weight
        return smoothed / kernel.sum(axis=1)

    def fill_time(self, ms, omegas):
        """tau_m = t_round_trip / (-ln(1 - Gamma_m)), as in section 4 [s]."""

        gamma = np.clip(self.transmission(ms, omegas), 0.0, 1.0 - 1.0e-15)
        with np.errstate(divide="ignore"):
            return self.round_trip(ms, omegas) / -np.log1p(-gamma)

    def filled_intensity(self, rho, omegas, t, mmax=None):
        """
        Intensity at distance rho from the centre of the pocket, a time t
        after a stationary isotropic flux is switched on outside, relative
        to the flux and averaged over each resonance:

            S = sum_m eps_m J_m(w rho / c_in)^2 (1 - exp(-t / tau_m)),

        eps_0 = 1, eps_m = 2; t may be an array of times, which adds a
        leading axis. It tends to 1 everywhere. With a throat,
        points closer to the edge than A_min c_in stay darker for many
        round trips, while modes with m > w A_min tunnel in; a slow-sound
        cavity without a throat fills them all at once.
        """

        omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
        times = np.asarray(t, dtype=float)
        x = omegas * rho / self.c_in
        if mmax is None:
            mmax = int(np.max(x)) + 20
        ms = np.arange(mmax + 1)
        j, _, _, _ = cylindrical_riccati(mmax, x)
        bessel2 = j ** 2 / (0.5 * np.pi * x)
        eps = np.where(ms == 0, 1.0, 2.0)[:, None]
        # tau = 0 for modes turning outside the pocket: they fill at once
        tau = self.fill_time(ms, omegas)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            filled = -np.expm1(-times[..., None, None] / tau)
        return np.sum(eps * bessel2 * np.nan_to_num(filled, nan=0.0),
                      axis=-2)
