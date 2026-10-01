# ==========================================================
# Tests for phonons in a two-dimensional condensate
#
# Section 5a: before modelling an acoustic pocket, the ray tools must
# reproduce a published geometry, the harmonically trapped condensate of
# Viermann et al. (2022). These pin the radial travel time and the
# curvature against closed forms derived independently, and the
# conserved quantities of the rays.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest
import sympy as sp

from warpdrive.acoustic import ThomasFermiDisc
from warpdrive.symbolic import christoffel, ricci_tensor


@pytest.fixture(scope="module")
def disc():
    return ThomasFermiDisc.from_viermann_2022()


def test_radial_rays_follow_the_arcsine_law(disc):
    """
    r(t) = R sin(arcsin(r_0/R) - c_0 t / R) for a ray aimed at the centre,
    integrated from the Hamiltonian and compared with the closed form.
    """

    R, c0, r0 = disc.radius, disc.sound_speed, 20.0e-6
    for t in (2.0e-3, 8.0e-3, 15.0e-3):
        position, _ = disc.trace_rays([[r0, 0.0]], [[-1.0, 0.0]], t)
        expected = R * math.sin(math.asin(r0 / R) - c0 * t / R)
        assert position[0, 0] == pytest.approx(expected, rel=1e-8)
        assert position[0, 1] == pytest.approx(0.0, abs=1e-15)


def test_travel_time_to_the_centre_in_the_experiment(disc):
    """
    From 20 um to the centre takes (R/c_0) arcsin(0.8) = 19.3 ms with the
    published 1.2 um/ms and 25 um, the time scale of their wave packets.
    """

    t = disc.radial_travel_time(20.0e-6, 0.0)
    assert t == pytest.approx(25.0 / 1.2 * math.asin(0.8) * 1e-3, rel=1e-12)
    assert t == pytest.approx(19.32e-3, rel=1e-3)


def test_rays_conserve_frequency_and_angular_momentum(disc):
    starts = [[15.0e-6, 0.0], [0.0, -10.0e-6], [5.0e-6, 5.0e-6]]
    aims = [[0.3, 1.0], [1.0, 0.2], [-1.0, 0.4]]
    position, momentum = disc.trace_rays(starts, aims, 10.0e-3)

    starts, aims = np.array(starts), np.array(aims)
    p0 = aims / np.linalg.norm(aims, axis=1)[:, None]
    w0 = disc.speed(np.hypot(*starts.T))
    L0 = starts[:, 0] * p0[:, 1] - starts[:, 1] * p0[:, 0]

    w = disc.speed(np.hypot(*position.T)) * np.hypot(*momentum.T)
    L = position[:, 0] * momentum[:, 1] - position[:, 1] * momentum[:, 0]
    np.testing.assert_allclose(w, w0, rtol=1e-8)
    np.testing.assert_allclose(L, L0, rtol=1e-8, atol=1e-20)


def test_uniform_condensate_gives_straight_rays():
    """With R much larger than the path, c_s is uniform and rays are lines."""

    flat = ThomasFermiDisc(sound_speed=1.0e-3, radius=1.0)
    position, _ = flat.trace_rays([[0.0, 0.0]], [[1.0, 1.0]], 1.0e-2)
    step = 1.0e-3 * 1.0e-2 / math.sqrt(2.0)
    np.testing.assert_allclose(position[0], [step, step], rtol=1e-6)


def test_curvature_of_the_optical_metric_from_the_christoffel_symbols(disc):
    """
    The 2D optical metric (dr^2 + r^2 dphi^2) / c_s^2 has Ricci scalar 2K;
    derived here from sympy and compared with -2 c_0^2 / (R^2 - r^2).
    """

    r, phi, R, c0 = sp.symbols("r phi R c_0", positive=True)
    c = c0 * sp.sqrt(1 - r ** 2 / R ** 2)
    g = sp.diag(1 / c ** 2, r ** 2 / c ** 2)
    ginv = sp.diag(c ** 2, c ** 2 / r ** 2)
    ricci = ricci_tensor(christoffel(g, ginv, (r, phi)), (r, phi))
    scalar = sum(ginv[i, i] * ricci[i, i] for i in range(2))
    curvature = sp.simplify(scalar / 2)
    assert sp.simplify(curvature + 2 * c0 ** 2 / (R ** 2 - r ** 2)) == 0

    points = np.array([0.0, 10.0e-6, 24.0e-6])
    numeric = sp.lambdify(r, curvature.subs({R: disc.radius,
                                             c0: disc.sound_speed}))
    np.testing.assert_allclose(disc.gaussian_curvature(points),
                               [numeric(p) for p in points], rtol=1e-12)


def test_centre_curvature_and_conformal_factor(disc):
    assert disc.gaussian_curvature(0.0) == pytest.approx(
        -2.0 * disc.sound_speed ** 2 / disc.radius ** 2)
    # n / c_s = (n_0 / c_0) sqrt(1 - r^2/R^2) for a Thomas-Fermi disc
    r = 15.0e-6
    assert disc.conformal_factor(r) == pytest.approx(
        disc.density / disc.sound_speed * math.sqrt(1 - (r / disc.radius)
                                                    ** 2))


# --- The acoustic pocket as a cavity (section 5c) ---
from warpdrive import BroeckMetric                          # noqa: E402
from warpdrive.acoustic import (AcousticPocket,             # noqa: E402
                                cylindrical_riccati)
from warpdrive.constants import C_LIGHT                     # noqa: E402
import mpmath as mp                                         # noqa: E402


def _bubble(inner, thickness, ratio, order=80):
    """A pocket with B = ratio inside; the shift wall is kept far away."""

    return BroeckMetric(speed=10.0 * C_LIGHT, radius=20.0 * (inner
                                                              + thickness),
                        sigma=10.0, inner_radius=inner, thickness=thickness,
                        alpha=ratio - 1.0, order=order)


def test_cylindrical_riccati_against_mpmath():
    x = np.array([0.3, 2.0, 17.0, 60.0])
    j, y, dj, dy = cylindrical_riccati(30, x)
    for m in (0, 1, 5, 20, 30):
        for i, xi in enumerate(x):
            scale = mp.sqrt(mp.pi * xi / 2)
            exact_j = float(scale * mp.besselj(m, xi))
            exact_y = float(scale * mp.bessely(m, xi))
            assert j[m, i] == pytest.approx(exact_j, rel=1e-10, abs=1e-280)
            assert y[m, i] == pytest.approx(exact_y, rel=1e-10)
    np.testing.assert_allclose(j * dy - y * dj, 1.0, rtol=1e-10)


@pytest.mark.parametrize("nu", [0.0, 1.0])
def test_flat_condensate_has_no_pocket(nu):
    flat = AcousticPocket(_bubble(10.0, 10.0, 1.0), 1.0, nu, 2000)
    a, c, _, _ = flat.transfer([0, 3, 10], [0.3, 1.0, 3.0])
    np.testing.assert_allclose(a, 1.0, atol=1e-7)
    np.testing.assert_allclose(c, 0.0, atol=1e-7)


@pytest.mark.parametrize("nu", [0.0, 1.0])
def test_zero_frequency_s_wave_is_sqrt_q(nu):
    """
    At w = 0 and m = 0, u'' = ((sqrt q)''/sqrt q) u is solved by
    u = sqrt(q), q = r n / c_s: pins the density term of the potential,
    the one a 3+1 conformal field would not have.
    """

    pocket = AcousticPocket(_bubble(20.0, 3.0, 2.0), 1.0, nu, 4000)
    a, b, eps = pocket.inner, pocket.outer, 1.0e-6
    sq = pocket.sqrt_q
    u, du = pocket._carry([0], [1.0e-9], np.array([[sq(a)]]),
                          np.array([[pocket.c_in * (sq(a) - sq(a - eps))
                                     / eps]]))
    assert u[0, 0] == pytest.approx(sq(b), rel=1e-6)
    assert du[0, 0] == pytest.approx(pocket.c_out * (sq(b + eps) - sq(b))
                                     / eps, rel=1e-4)


@pytest.mark.parametrize("nu", [0.0, 1.0])
def test_sharp_edge_reflects_by_impedance(nu):
    """
    A step thin against the wavelength and a pocket large against it
    transmit 1 - ((B - 1)/(B + 1))^2, with impedance n k = n w / c_s:
    the ratio is 1/B for uniform coupling and B for uniform density,
    the same |r| either way.
    """

    pocket = AcousticPocket(_bubble(200.0, 0.5, 2.0), 1.0, nu, 2000)
    gamma = pocket.transmission([0], [0.25])[0, 0]
    assert gamma == pytest.approx(1.0 - (1.0 / 3.0) ** 2, rel=1e-2)


def test_flux_is_conserved_and_the_cavity_fills_on_average():
    pocket = AcousticPocket(_bubble(20.0, 3.0, 2.0), 1.0, 1.0, 3000)
    incoming, outgoing = pocket.crossing([0, 4], [0.1, 0.2])
    np.testing.assert_allclose(np.abs(incoming) ** 2,
                               np.abs(outgoing) ** 2 + 1.0, rtol=1e-7)

    omegas = np.linspace(0.2, 0.6, 4000)
    weight = pocket.interior_weight([0], omegas)[0]
    assert weight.mean() == pytest.approx(1.0, abs=0.05)
    # a low-finesse comb whose contrast is the Fabry-Perot one,
    # ((1 + sqrt(1 - Gamma)) / (1 - sqrt(1 - Gamma)))^2 = 4 for Gamma = 8/9
    gamma = pocket.transmission([0], [0.4])[0, 0]
    root = math.sqrt(1.0 - gamma)
    expected = ((1.0 + root) / (1.0 - root)) ** 2
    assert weight.max() / weight.min() == pytest.approx(expected, rel=0.05)


@pytest.fixture(scope="module")
def sharp():
    """B = 2, a sharp edge: a throat with A_min half the edge radius."""

    return AcousticPocket(_bubble(20.0, 1.0, 2.0), 1.0, 1.0, 2000)


def test_comb_needs_about_two_round_trips(sharp):
    """
    Seen through a record of duration T, the comb is flat below one round
    trip, has contrast above 2 after two, and tends to the Fabry-Perot
    value for long records: the low-finesse cavity forms fast, which is
    what lowers the requirement of section 5b from mu T / h ~ 25 to ~ 6.
    """

    trip = 2.0 * sharp.inner / sharp.c_in
    omegas = np.linspace(0.005, 0.45, 2500)
    band = (omegas > 0.075) & (omegas < 0.25)

    def contrast(T):
        spectrum = sharp.measured_spectrum(omegas, T)[band]
        return spectrum.max() / spectrum.min()

    assert contrast(0.5 * trip) < 1.1
    assert 2.0 < contrast(2.0 * trip) < 2.5
    weight = sharp.interior_weight([0], omegas)[0][band]
    assert contrast(200.0 * trip) == pytest.approx(
        weight.max() / weight.min(), rel=0.1)


def test_throat_darkens_the_edge_of_the_pocket(sharp):
    """
    After two round trips, at 0.9 R~ a pocket behind a throat holds about
    two thirds of the outside intensity; a slow-sound cavity with the same
    B and a smooth edge, without a throat, is already full. Inside
    A_min c_in from the centre there is no difference.
    """

    smooth = AcousticPocket(_bubble(20.0, 60.0, 2.0, order=3), 1.0, 1.0,
                            2000)
    trip = 2.0 * sharp.inner / sharp.c_in
    omegas = np.linspace(0.125, 0.25, 20)

    edge = sharp.filled_intensity(18.0, omegas, 2.0 * trip).mean()
    later = sharp.filled_intensity(18.0, omegas, 100.0 * trip).mean()
    assert 0.6 < edge < 0.75
    assert edge < later < 1.0
    assert smooth.filled_intensity(18.0, omegas, 2.0 * trip).mean() \
        == pytest.approx(1.0, abs=0.02)
    assert sharp.filled_intensity(10.0, omegas, 2.0 * trip).mean() \
        == pytest.approx(1.0, abs=0.03)


def test_round_trip_is_the_chord_of_the_flat_pocket(sharp):
    """
    A ray of angular momentum m / w crosses the flat pocket along a chord,
    2 sqrt((R~/c_in)^2 - (m/w)^2) in optical length, i.e. in time.
    """

    edge = sharp.inner / sharp.c_in
    trip = sharp.round_trip([0, 3, 100], [0.2])[:, 0]
    assert trip[0] == pytest.approx(2.0 * edge)
    assert trip[1] == pytest.approx(2.0 * math.sqrt(edge ** 2 - 15.0 ** 2))
    assert trip[2] == 0.0


# --- Bogoliubov dispersion, local-index approximation ---
@pytest.fixture(scope="module")
def dispersive():
    """B = 2 with xi_out = 1 in units c_out = 1: cutoff at w = 1/4."""

    return AcousticPocket(_bubble(20.0, 1.0, 2.0), 1.0, 1.0, 2000,
                          healing=1.0)


def test_bogoliubov_wavenumber_limits(sharp):
    """w/c for long waves, sqrt(2 w / c xi) for short ones, and the
    dispersion relation w^2 = c^2 k^2 (1 + k^2 xi^2 / 4) in between."""

    c, xi = 0.5, 2.0
    assert sharp.bogoliubov_wavenumber(1.0e-6, c, xi) == pytest.approx(
        1.0e-6 / c, rel=1e-9)
    w = 1.0e4
    assert sharp.bogoliubov_wavenumber(w, c, xi) == pytest.approx(
        math.sqrt(2.0 * w / (c * xi)), rel=1e-3)
    k = sharp.bogoliubov_wavenumber(0.3, c, xi)
    assert c ** 2 * k ** 2 * (1.0 + k ** 2 * xi ** 2 / 4.0) == pytest.approx(
        0.09, rel=1e-12)
    assert sharp.bogoliubov_wavenumber(0.3, c, None) == pytest.approx(0.3 / c)


def test_dispersion_vanishes_at_low_frequency(sharp, dispersive):
    omegas = [1.0e-3, 2.0e-3]
    np.testing.assert_allclose(dispersive.transmission([0, 3], omegas),
                               sharp.transmission([0, 3], omegas),
                               rtol=1e-4)


def test_dispersive_flux_is_conserved(dispersive):
    """K_out |I|^2 = K_out |O|^2 + K_in, with K = c k_B on each side."""

    omegas = np.array([0.1, 0.2, 0.3])
    incoming, outgoing = dispersive.crossing([0, 4], omegas)
    inside, outside = dispersive._effective(omegas)
    np.testing.assert_allclose(outside * np.abs(incoming) ** 2,
                               outside * np.abs(outgoing) ** 2 + inside,
                               rtol=1e-8)


def test_dispersion_moves_the_comb_up_and_sharpens_it(sharp, dispersive):
    """
    Below the cutoff the same three resonances survive, shifted up as
    k_B grows faster than w / c; the larger jump of k_B at the edge
    reflects more, so after two round trips the comb is sharper (2.2 ->
    2.6). Behind the throat the edge darkens less, 0.67 -> 0.77, since
    more partial waves cross it, but still more than without a throat.
    """

    omegas = np.linspace(0.005, 0.45, 2500)

    def peaks(pocket):
        weight = pocket.interior_weight([0], omegas)[0]
        top = [i for i in range(1, len(omegas) - 1)
               if weight[i] > weight[i - 1] and weight[i] > weight[i + 1]
               and omegas[i] < 0.25]
        return omegas[top], weight

    plain, plain_weight = peaks(sharp)
    shifted, weight = peaks(dispersive)
    assert len(plain) == len(shifted) == 3
    assert np.all(shifted > plain)

    trip = 2.0 * sharp.inner / sharp.c_in
    band = (omegas > 0.075) & (omegas < 0.25)
    contrast = [p.measured_spectrum(omegas, 2.0 * trip, w)[band]
                for p, w in ((sharp, plain_weight), (dispersive, weight))]
    ratios = [c.max() / c.min() for c in contrast]
    assert ratios[1] == pytest.approx(2.6, rel=0.05)
    assert ratios[1] > ratios[0]

    band_w = np.linspace(0.125, 0.25, 20)
    edge = dispersive.filled_intensity(18.0, band_w, 2.0 * trip).mean()
    assert edge == pytest.approx(0.77, abs=0.01)


def test_dispersive_transmission_and_reflection_add_to_one(dispersive):
    omegas = np.array([0.1, 0.2, 0.3])
    incoming, outgoing = dispersive.crossing([0, 4], omegas)
    gamma = dispersive.transmission([0, 4], omegas)
    np.testing.assert_allclose(gamma + np.abs(outgoing / incoming) ** 2,
                               1.0, rtol=1e-8)


def test_dispersive_round_trip_runs_at_the_group_velocity(dispersive):
    """
    v_g / c = (1 + k^2 xi^2 / 2) / sqrt(1 + k^2 xi^2 / 4), faster than
    sound: the s-wave crosses the pocket in 2 R~ / v_g.
    """

    w = 0.2
    xi_in = 2.0
    k = dispersive.bogoliubov_wavenumber(w, dispersive.c_in, xi_in)
    group = dispersive.c_in * (1.0 + 0.5 * (k * xi_in) ** 2) / math.sqrt(
        1.0 + 0.25 * (k * xi_in) ** 2)
    assert dispersive.round_trip([0], [w])[0, 0] == pytest.approx(
        2.0 * dispersive.inner / group, rel=1e-12)
