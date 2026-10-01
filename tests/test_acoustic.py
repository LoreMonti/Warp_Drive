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
