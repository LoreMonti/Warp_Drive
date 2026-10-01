# ==========================================================
# Scalar waves in the pocket: a cavity behind a throat
#
# Inside the shift wall, in the frame of the bubble, the metric is
# ultrastatic, ds^2 = -c^2 dt^2 + dl^2 + A(l)^2 dOmega^2. A massless
# scalar Phi = exp(-i w t) Y_lm(theta, phi) u(l) / A obeys
#
#     u'' + [k^2 - V_l(l)] u = 0,   V_l = l(l+1) / A^2 + A'' / A,
#
# with k = w / c and primes in proper distance. A''/A is the radial
# curvature of the throat; the centrifugal term peaks there too, so only
# l < k A_min crosses it classically.
#
# The pocket has no exit on the far side: it is a closed cavity, and in a
# steady state every partial wave fills it to the intensity outside,
# whatever the throat transmits. The throat only sets how long the
# filling takes.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import math

# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from .constants import C_LIGHT


def riccati_bessel(lmax, x):
    """
    Riccati-Bessel functions hat j_l(x) = x j_l(x), hat y_l(x) = x y_l(x)
    and their derivatives, for l = 0 .. lmax and x > 0.

        hat j_0 = sin x,  hat y_0 = -cos x,
        hat f_l' = hat f_{l-1} - (l / x) hat f_l.

    hat y is built by upward recurrence, which is stable for it; hat j by
    downward recurrence from far above both lmax and x (Miller's method),
    normalised to sin x, since upward recurrence loses it for l > x.

    Returns (j, y, dj, dy), each of shape (lmax + 1,) + x.shape.
    """

    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Riccati-Bessel functions need x > 0")

    y = np.empty((lmax + 1,) + x.shape)
    y[0] = -np.cos(x)
    if lmax > 0:
        y[1] = -np.cos(x) / x - np.sin(x)
    with np.errstate(over="ignore", invalid="ignore"):
        for l in range(1, lmax):
            y[l + 1] = (2 * l + 1) / x * y[l] - y[l - 1]

    j = np.zeros((lmax + 1,) + x.shape)
    start = lmax + int(np.max(x)) + 40
    above, current = np.zeros_like(x), np.full_like(x, 1.0e-300)
    for l in range(start, 0, -1):
        above, current = current, (2 * l + 1) / x * current - above
        if l - 1 <= lmax:
            j[l - 1] = current
        big = np.abs(current) > 1.0e250
        if np.any(big):
            j[:, big] *= 1.0e-250
            above = np.where(big, above * 1.0e-250, above)
            current = np.where(big, current * 1.0e-250, current)
    j *= np.sin(x) / j[0]

    with np.errstate(over="ignore", invalid="ignore"):
        lower_j = np.concatenate([np.cos(x)[None], j[:-1]])
        lower_y = np.concatenate([np.sin(x)[None], y[:-1]])
        ells = np.arange(lmax + 1).reshape((-1,) + (1,) * x.ndim)
        dj = lower_j - ells / x * j
        dy = lower_y - ells / x * y
    return j, y, dj, dy


class PocketWaves:
    """
    Partial waves of a massless scalar across the transition region of a
    `BroeckMetric`, from the edge of the flat pocket, proper radius P, to
    the flat region outside it, areal radius b.

    Inside the pocket A = l and outside A = r, so in both the radial
    solutions are Riccati-Bessel functions, of k l and of k A. Two real
    solutions, hat j(k l) and hat y(k l) at l = P, are carried across the
    transition by RK4 in the coordinate r, with du/dr = B du/dl, and
    written outside as a hat j(k A) + c hat y(k A). Everything else
    follows from that 2 x 2 transfer.

    Parameters
    ----------
    metric  : BroeckMetric with alpha > 0
    n_steps : RK4 steps across the transition region
    """

    def __init__(self, metric, n_steps=20000):
        self.metric = metric
        self.n_steps = int(n_steps)
        self.pocket_radius = metric.pocket_proper_radius()
        self.outer_radius = metric.inner_radius + metric.thickness

        # V_l = lambda / A^2 + A_ll / A at the RK4 nodes and midpoints
        h = metric.thickness / self.n_steps
        r = metric.inner_radius + 0.5 * h * np.arange(2 * self.n_steps + 1)
        # B'' jumps at r = R~: take its limit from the transition side, or
        # the first step sees the flat pocket and RK4 drops to first order
        r[0] += 1.0e-9 * h
        _, B, first, second = metric._profiles(r)
        areal = B * r
        curvature = ((second * r + 2.0 * first) / B
                     - (first * r + B) * first / B ** 2) / B
        self._h = h
        self._B = B
        self._inverse_area = 1.0 / areal ** 2
        self._curvature = curvature / areal

    # --- Transfer across the transition ---
    def _carry(self, ells, ks, u, du):
        """Carry u and du/dl from r = R~ to r = R~ + D~ (RK4 in r)."""

        lam = (ells * (ells + 1.0))[:, None]
        k2 = (ks ** 2)[None, :]
        h = self._h

        def rhs(i, u, du):
            potential = lam * self._inverse_area[i] + self._curvature[i]
            return self._B[i] * du, self._B[i] * (potential - k2) * u

        for n in range(self.n_steps):
            i = 2 * n
            a1, b1 = rhs(i, u, du)
            a2, b2 = rhs(i + 1, u + 0.5 * h * a1, du + 0.5 * h * b1)
            a3, b3 = rhs(i + 1, u + 0.5 * h * a2, du + 0.5 * h * b2)
            a4, b4 = rhs(i + 2, u + h * a3, du + h * b3)
            u = u + h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            du = du + h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
        return u, du

    def transfer(self, ells, ks):
        """
        Exterior coefficients of the two interior solutions,

            hat j_l(k l) -> a_j hat j_l(k A) + c_j hat y_l(k A),
            hat y_l(k l) -> a_y hat j_l(k A) + c_y hat y_l(k A).

        ells : array of l, ks : array of k = w / c [m^-1].
        Returns (a_j, c_j, a_y, c_y), each of shape (len(ells), len(ks)).
        """

        ells = np.asarray(ells, dtype=int)
        ks = np.asarray(ks, dtype=float)
        lmax = int(ells.max())

        j, y, dj, dy = riccati_bessel(lmax, ks * self.pocket_radius)
        start_u = np.concatenate([j[ells], y[ells]])
        start_du = np.concatenate([ks * dj[ells], ks * dy[ells]])
        both = np.concatenate([ells, ells])
        with np.errstate(over="ignore", invalid="ignore"):
            u, du = self._carry(both, ks, start_u, start_du)

            J, Y, dJ, dY = riccati_bessel(lmax, ks * self.outer_radius)
            J, Y, dJ, dY = (np.concatenate([f[ells], f[ells]])
                            for f in (J, Y, dJ, dY))
            # Wronskian hat j hat y' - hat y hat j' = 1
            a = u * dY - Y * du / ks
            c = J * du / ks - u * dJ
        n = len(ells)
        return a[:n], c[:n], a[n:], c[n:]

    # --- What the crew sees ---
    def interior_weight(self, ells, ks):
        """
        1 / N_l = 1 / (a_j^2 + c_j^2): intensity of the regular partial
        wave inside the pocket per unit incident intensity, relative to
        flat space. A Fabry-Perot comb in k whose average over a resolved
        band is 1 for every l, however opaque the throat: the peaks grow
        as 4 / Gamma_l while their width shrinks as Gamma_l.
        """

        a, c, _, _ = self.transfer(ells, ks)
        with np.errstate(over="ignore", invalid="ignore"):
            weight = 1.0 / (a ** 2 + c ** 2)
        return np.nan_to_num(weight, nan=0.0, posinf=0.0)

    def crossing(self, ells, ks):
        """
        Single-pass scattering off the throat from outside: inside, only
        the ingoing wave hat j - i hat y; outside, I hat h^- + O hat h^+,
        with hat h^-+ = hat j -+ i hat y. Returns (I, O), complex, so that
        flux conservation reads |I|^2 = |O|^2 + 1.
        """

        a_j, c_j, a_y, c_y = self.transfer(ells, ks)
        alpha = a_j - 1j * a_y
        gamma = c_j - 1j * c_y
        # alpha hat j + gamma hat y = I hat h^- + O hat h^+
        incoming = 0.5 * (alpha + 1j * gamma)
        outgoing = 0.5 * (alpha - 1j * gamma)
        return incoming, outgoing

    def transmission(self, ells, ks):
        """Gamma_l(k) = 1 / |I|^2, the fraction of flux crossing the throat."""

        incoming, _ = self.crossing(ells, ks)
        with np.errstate(over="ignore", invalid="ignore"):
            gamma = 1.0 / np.abs(incoming) ** 2
        return np.nan_to_num(gamma, nan=0.0, posinf=0.0)

    def round_trip(self, ells, ks):
        """
        Time for mode l to cross the flat pocket from its edge to its
        inner turning point (l + 1/2) / k and back [s].
        """

        ells = np.asarray(ells, dtype=float)[:, None]
        ks = np.asarray(ks, dtype=float)[None, :]
        turning = np.minimum((ells + 0.5) / ks, self.pocket_radius)
        return 2.0 * np.sqrt(self.pocket_radius ** 2 - turning ** 2) / C_LIGHT

    def fill_time(self, ells, ks):
        """
        e-folding time with which mode l fills the pocket once a stationary
        flux is switched on outside [s]: each round trip lets in a fraction
        Gamma_l of what is missing, so

            tau_l = t_round_trip / (-ln(1 - Gamma_l)),

        about t_round_trip / Gamma_l for an opaque throat and infinite
        when nothing crosses it.
        """

        gamma = np.clip(self.transmission(ells, ks), 0.0, 1.0 - 1.0e-15)
        with np.errstate(divide="ignore"):
            return self.round_trip(ells, ks) / -np.log1p(-gamma)

    def filled_intensity(self, rho, ks, t, lmax=None):
        """
        Intensity at proper distance rho from the centre of the pocket, a
        time t after a stationary, isotropic flux is switched on outside,
        relative to the flux itself and averaged over each resonance:

            S(rho, t) = sum_l (2l + 1) j_l(k rho)^2 (1 - exp(-t / tau_l)).

        It tends to 1 everywhere as t -> infinity, the closed-cavity limit.
        At the centre only l = 0 contributes. Off centre, after a few
        round trips, it reduces to the bright fraction of the sky seen by
        rays, 1 - sqrt(1 - (A_min / rho)^2): the modes with l > k A_min,
        trapped for rays, take exponentially long to tunnel in.
        """

        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        if lmax is None:
            lmax = int(np.max(ks) * max(rho, 1.0)) + 40
        ells = np.arange(lmax + 1)
        j, _, _, _ = riccati_bessel(lmax, ks * rho)
        weights = (2.0 * ells[:, None] + 1.0) * (j / (ks * rho)) ** 2
        with np.errstate(over="ignore", invalid="ignore"):
            filled = -np.expm1(-t / self.fill_time(ells, ks))
        return np.sum(weights * np.nan_to_num(filled, nan=0.0), axis=0)


def bright_fraction(rho, throat_radius):
    """
    Fraction of the sky lit for rays at proper distance rho from the
    centre of a flat pocket: the directions with angular momentum
    rho sin(theta) below the areal radius of the throat, the ones that
    came in through it,

        1 - sqrt(1 - (A_min / rho)^2),  or 1 for rho <= A_min.
    """

    if rho <= throat_radius:
        return 1.0
    return 1.0 - math.sqrt(1.0 - (throat_radius / rho) ** 2)
