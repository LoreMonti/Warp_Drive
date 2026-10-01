# ==========================================================
# Tests for the scalar waves in the pocket
#
# The pocket is a closed cavity behind a throat. These pin the special
# functions, flux conservation across the throat, the Fabry-Perot
# relation between the resonances and the single-pass transmission, the
# closed-cavity theorem (every partial wave fills the pocket on average),
# and the ray limit of the filling.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import mpmath as mp
import numpy as np
import pytest

from warpdrive import BroeckMetric
from warpdrive.constants import C_LIGHT
from warpdrive.waves import PocketWaves, bright_fraction, riccati_bessel


@pytest.fixture(scope="module")
def pocket():
    return BroeckMetric(speed=10.0 * C_LIGHT)


@pytest.fixture(scope="module")
def waves(pocket):
    return PocketWaves(pocket, n_steps=4000)


def test_riccati_bessel_against_mpmath():
    x = np.array([0.3, 2.0, 17.0, 80.0])
    j, y, dj, dy = riccati_bessel(40, x)
    for l in (0, 1, 5, 20, 40):
        for i, xi in enumerate(x):
            scale = xi * mp.sqrt(mp.pi / (2 * xi))
            exact_j = float(scale * mp.besselj(l + 0.5, xi))
            exact_y = float(scale * mp.bessely(l + 0.5, xi))
            assert j[l, i] == pytest.approx(exact_j, rel=1e-10, abs=1e-280)
            assert y[l, i] == pytest.approx(exact_y, rel=1e-10)
    # the Wronskian hat j hat y' - hat y hat j' is 1 for every l
    np.testing.assert_allclose(j * dy - y * dj, 1.0, rtol=1e-9)


def test_flat_space_transmits_everything():
    """With alpha = 0 there is no throat: a = 1, c = 0 and Gamma = 1."""

    flat = PocketWaves(BroeckMetric(speed=10.0 * C_LIGHT, alpha=0.0), 2000)
    a, c, _, _ = flat.transfer([0, 3, 10], [0.3, 1.0, 3.0])
    np.testing.assert_allclose(a, 1.0, atol=1e-7)
    np.testing.assert_allclose(c, 0.0, atol=1e-7)
    np.testing.assert_allclose(flat.transmission([0], [0.3, 2.0]), 1.0,
                               atol=1e-9)


def test_flux_is_conserved_across_the_throat(waves):
    """|I|^2 = |O|^2 + 1: what comes in is reflected or crosses."""

    incoming, outgoing = waves.crossing([0, 10, 22, 26], [0.5, 2.1])
    np.testing.assert_allclose(np.abs(incoming) ** 2,
                               np.abs(outgoing) ** 2 + 1.0, rtol=1e-7)


def test_rk4_converges_at_fourth_order(pocket):
    """B'' jumps at r = R~; sampling it from the pocket side would drop
    the transfer to first order."""

    reference = PocketWaves(pocket, 16000).transmission([0, 22], [0.5, 2.1])
    errors = [np.max(np.abs(PocketWaves(pocket, n).transmission(
        [0, 22], [0.5, 2.1]) / reference - 1.0)) for n in (1000, 2000)]
    assert errors[0] / errors[1] > 12.0


def test_throat_stops_high_angular_momentum(waves, pocket):
    """Modes with l well above k A_min tunnel; those well below cross."""

    _, throat = pocket.throat()
    k = 2.1
    gamma = waves.transmission([5, int(k * throat) + 10], [k])
    assert gamma[0, 0] > 0.5
    assert gamma[1, 0] < 1e-3


def test_resonance_peaks_are_four_over_the_transmission(waves):
    """
    Fabry-Perot: inside, T / |1 - sqrt(1 - T) e^{i phi}|^2 peaks at
    T / (1 - sqrt(1 - T))^2 ~ 4 / T. Ties the interior field to the
    single-pass transmission, two different solutions of the same
    transfer.
    """

    ks = np.linspace(2.0, 2.2, 20000)
    weight = waves.interior_weight([24], ks)[0]
    gamma = waves.transmission([24], ks)[0]
    k = int(np.argmax(weight))
    peak = gamma[k] / (1.0 - math.sqrt(1.0 - gamma[k])) ** 2
    assert weight[k] == pytest.approx(peak, rel=2e-2)


@pytest.mark.parametrize("ell", [0, 10, 22, 24])
def test_closed_cavity_fills_on_average(waves, ell):
    """
    The average of 1 / N_l over a band of resolved resonances is 1,
    whatever Gamma_l: the throat cannot shield the pocket in a steady
    state, only delay its filling.
    """

    ks = np.linspace(2.0, 2.2, 20000)
    weight = waves.interior_weight([ell], ks)[0]
    assert weight.mean() == pytest.approx(1.0, abs=0.04)


def test_filling_reaches_the_ray_limit_then_the_closed_cavity(waves,
                                                              pocket):
    """
    Off centre, after a few round trips the lit fraction is the one rays
    give, and it grows on as trapped modes tunnel in, but those with
    l >> k A_min take far longer than any trip. Near the centre, where
    every contributing l is below k A_min, the pocket fills completely.
    """

    _, throat = pocket.throat()
    ks = np.linspace(3.0, 3.5, 15)
    trip = 2.0 * waves.pocket_radius / C_LIGHT

    rho = 60.0
    early = waves.filled_intensity(rho, ks, 20.0 * trip).mean()
    late = waves.filled_intensity(rho, ks, 1.0e12).mean()
    assert early == pytest.approx(bright_fraction(rho, throat), rel=0.25)
    assert early < late < 0.1
    assert waves.filled_intensity(rho, ks, 0.0).max() == 0.0

    near = waves.filled_intensity(0.5 * throat, ks, 1.0e3 * trip).mean()
    assert near == pytest.approx(1.0, abs=0.02)


def test_bright_fraction():
    assert bright_fraction(5.0, 10.0) == 1.0
    assert bright_fraction(20.0, 10.0) == pytest.approx(1 - math.sqrt(0.75))


def test_zero_frequency_s_wave_is_the_areal_radius(waves):
    """
    At k = 0 and l = 0 the radial equation is u'' = (A''/A) u, solved
    exactly by u = A: starting from A = P, A' = 1 at the edge of the
    pocket, the transfer must land on A = b, A' = 1 outside. Pins the
    curvature term of the potential and the factor B in du/dr.
    """

    u, du = waves._carry(np.array([0]), np.array([1.0e-12]),
                         np.array([[waves.pocket_radius]]),
                         np.array([[1.0]]))
    assert u[0, 0] == pytest.approx(waves.outer_radius, rel=1e-6)
    assert du[0, 0] == pytest.approx(1.0, rel=1e-6)


def test_opaque_modes_fill_in_a_round_trip_over_the_transmission(waves,
                                                                 pocket):
    """tau_l -> t_round_trip / Gamma_l when the throat lets little in."""

    _, throat = pocket.throat()
    k = 2.1
    ells = [int(k * throat) + 5]
    gamma = waves.transmission(ells, [k])[0, 0]
    assert 1e-6 < gamma < 1e-2
    ratio = (waves.fill_time(ells, [k]) * gamma
             / waves.round_trip(ells, [k]))[0, 0]
    assert ratio == pytest.approx(1.0, rel=gamma)
