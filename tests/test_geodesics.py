# ==========================================================
# Tests for the null geodesics seen from the centre of the bubble
#
# Every check here is an exact statement that the ray integrator knows
# nothing about: the closed-form blueshift from the conserved
# Hamiltonian, the rear horizon from the bisection solver, its surface
# gravity from the shape function, and the identity of the two metrics
# for a ship at the centre.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest

from warpdrive import AlcubierreMetric, BroeckMetric
from warpdrive.constants import C_LIGHT
from warpdrive.geodesics import (
    ESCAPED,
    HORIZON,
    horizon_surface_gravity,
    sky_map,
    trace_rays,
    unlensed_brightness,
)

SPEED_RATIO = 10.0
ANGLES = np.linspace(0.0, math.pi, 61)


@pytest.fixture(scope="module")
def alcubierre():
    return AlcubierreMetric(speed=SPEED_RATIO * C_LIGHT, radius=100.0,
                            sigma=1.0)


@pytest.fixture(scope="module")
def bundle(alcubierre):
    return trace_rays(alcubierre, ANGLES)


def test_rays_are_traced_backwards(bundle, alcubierre):
    """
    Looking ahead, the light comes from ahead; looking behind, the ray
    stalls behind the ship. Tracing forwards in time would swap the two.
    """

    assert bundle.status[0] == ESCAPED
    assert bundle.end_position[0, 0] > alcubierre.radius
    assert bundle.status[-1] == HORIZON
    assert bundle.end_position[0, -1] < 0.0


def test_flat_space_leaves_the_sky_unchanged():
    slow = AlcubierreMetric(speed=1.0e-9 * C_LIGHT, radius=100.0, sigma=1.0)
    rays = trace_rays(slow, ANGLES)

    assert np.all(rays.status == ESCAPED)
    assert np.allclose(rays.source_angle, ANGLES, atol=1e-7)
    assert np.allclose(rays.frequency_ratio, 1.0, atol=1e-8)


def test_blueshift_follows_from_the_conserved_hamiltonian(bundle):
    """E_ship / E_far = 1 - (v_s/c) n_xi, for every escaped ray."""

    visible = bundle.escaped & (bundle.frequency_ratio > 1e-6)
    predicted = 1.0 - SPEED_RATIO * bundle.direction_xi[visible]

    assert visible.sum() > 20
    assert np.allclose(bundle.frequency_ratio[visible], predicted,
                       rtol=1e-8)
    assert bundle.frequency_ratio[0] == pytest.approx(1.0 + SPEED_RATIO,
                                                      rel=1e-9)


def test_hamiltonian_is_conserved(bundle):
    visible = bundle.escaped & (bundle.frequency_ratio > 1e-3)
    assert np.all(bundle.hamiltonian_drift[visible] < 1e-8)


def test_no_light_arrives_from_faster_than_the_bubble_outruns(bundle):
    """Only sources with n_xi < c / v_s can be seen at all."""

    assert np.all(bundle.direction_xi[bundle.escaped]
                  < 1.0 / SPEED_RATIO + 1e-9)


def test_rear_ray_stalls_at_the_horizon(bundle, alcubierre):
    """The bisection in the base class knows nothing about rays."""

    assert bundle.end_position[0, -1] == pytest.approx(
        -alcubierre.horizon_offset(), rel=1e-8)
    assert bundle.end_position[1, -1] == pytest.approx(0.0, abs=1e-9)


def test_rear_ray_fades_at_the_surface_gravity(alcubierre):
    """
    Held at the horizon, |p| grows as exp(kappa w) with
    kappa = (v_s/c) |f'(h)|, so the time to lose six more orders of
    magnitude is ln(1e6) / kappa.
    """

    behind = [math.pi]
    early = trace_rays(alcubierre, behind, min_ratio=1e-6)
    late = trace_rays(alcubierre, behind, min_ratio=1e-12)
    kappa = horizon_surface_gravity(alcubierre)

    assert kappa == pytest.approx(
        SPEED_RATIO * abs(float(alcubierre.shape_derivative(
            alcubierre.horizon_offset()))), rel=1e-9)
    assert late.elapsed[0] - early.elapsed[0] == pytest.approx(
        math.log(1e6) / kappa, rel=1e-3)


def test_subluminal_bubble_hides_nothing():
    slow = AlcubierreMetric(speed=0.5 * C_LIGHT, radius=100.0, sigma=1.0)
    rays = trace_rays(slow, ANGLES)

    assert np.all(rays.status == ESCAPED)
    assert np.all(rays.frequency_ratio > 0.0)
    assert horizon_surface_gravity(slow) is None


def test_pocket_does_not_change_the_sky_seen_from_the_centre(bundle,
                                                             alcubierre):
    """
    B is spherically symmetric, so rays leaving the centre cross the
    transition region radially and only change speed; and the blueshift
    formula does not contain B. Same wall, same sky.
    """

    broeck = BroeckMetric(speed=alcubierre.speed, radius=alcubierre.radius,
                          sigma=alcubierre.sigma)
    rays = trace_rays(broeck, ANGLES)

    assert np.array_equal(rays.status, bundle.status)
    visible = bundle.escaped & (bundle.frequency_ratio > 1e-6)
    assert np.allclose(rays.source_angle[visible],
                       bundle.source_angle[visible], atol=1e-8)
    assert np.allclose(rays.frequency_ratio[visible],
                       bundle.frequency_ratio[visible], rtol=1e-8)
    assert np.all(rays.elapsed[visible] > bundle.elapsed[visible])


def test_planck_thin_wall_is_refused():
    with pytest.raises(ValueError):
        trace_rays(BroeckMetric.from_paper(speed=10.0 * C_LIGHT), [0.0])


# --- Sky map ---
@pytest.fixture(scope="module")
def sky(alcubierre):
    return sky_map(alcubierre)


def test_visible_limit_is_the_closed_form(sky):
    """n_xi < c/v_s means a source angle below arccos(-c/v_s)."""

    assert sky.visible_limit == pytest.approx(math.acos(-1.0 / SPEED_RATIO))
    assert sky.source_angle[-1] == pytest.approx(sky.visible_limit,
                                                 abs=1e-5)


def test_apparent_positions_invert_the_traced_rays(sky, bundle):
    visible = bundle.escaped & (bundle.frequency_ratio > 1e-3)
    look, ratio = sky.apparent(bundle.source_angle[visible])

    assert np.allclose(look, bundle.look_angle[visible], atol=1e-3)
    assert np.allclose(ratio, bundle.frequency_ratio[visible], rtol=1e-3)


def test_sources_beyond_the_limit_are_hidden(sky):
    look, ratio = sky.apparent([0.0, sky.visible_limit + 0.01, math.pi])

    assert look[0] == 0.0 and ratio[0] == pytest.approx(1.0 + SPEED_RATIO)
    assert np.all(np.isnan(look[1:])) and np.all(np.isnan(ratio[1:]))


def test_subluminal_sky_is_complete():
    slow = sky_map(AlcubierreMetric(speed=0.5 * C_LIGHT, radius=100.0,
                                    sigma=1.0), n_rays=91)

    assert slow.visible_limit == math.pi
    assert slow.source_angle[-1] == pytest.approx(math.pi)
    assert slow.frequency_ratio[-1] == pytest.approx(0.5)


# --- Brightness ---
def test_flat_space_changes_no_brightness():
    rest = sky_map(AlcubierreMetric(speed=1.0e-9 * C_LIGHT, radius=100.0,
                                    sigma=1.0))
    angles = np.linspace(0.0, math.pi, 7)

    assert np.allclose(rest.magnification(angles), 1.0, atol=1e-6)
    assert np.allclose(rest.flux_ratio(angles), 1.0, atol=1e-6)
    assert rest.sky_brightness() == pytest.approx(1.0, rel=1e-5)


def test_star_ahead_is_brighter_by_the_fourth_power(sky):
    assert sky.flux_ratio(0.0) == pytest.approx(
        (1.0 + SPEED_RATIO) ** 4 * float(sky.magnification(0.0)),
        rel=1e-9)


def test_magnification_conserves_the_apparent_solid_angle(alcubierre):
    """
    \\int mu dOmega_source = \\int dOmega_look, over the part of the sky
    with R > 0.1. On a finite fan of rays the two differ by a
    discretisation error, which must fall as the square of the ray
    spacing: halving the spacing divides it by four.
    """

    def mismatch(n_rays):
        fan = sky_map(alcubierre, n_rays=n_rays)
        keep = fan.frequency_ratio > 0.1
        source = fan.source_angle[keep]
        covered = np.trapezoid(fan.magnification(source) * np.sin(source),
                               source)
        return covered / (1.0 - math.cos(fan.look_angle[keep][-1])) - 1.0

    coarse, fine = mismatch(361), mismatch(721)

    assert abs(fine) < 1e-4
    assert coarse / fine == pytest.approx(4.0, rel=0.1)


def test_received_light_is_the_same_on_either_sky(sky):
    """
    The light from an isotropic background, integrated over the apparent
    sky, must equal R^4 mu integrated over the true one: two routes, one
    through the traced rays and one through the magnification.
    """

    source = sky.source_angle
    through_source = 0.5 * np.trapezoid(
        sky.frequency_ratio ** 4 * sky.magnification(source)
        * np.sin(source), source)

    assert through_source == pytest.approx(sky.sky_brightness(), rel=1e-6)


def test_lensing_is_what_separates_the_received_light_from_the_unlensed():
    """
    Slow bubbles barely distort the sky, so the closed form without
    magnification holds; at 10c the distortion takes 6 % off it.
    """

    for speed, tolerance in ((0.5, 1e-3), (2.0, 1e-2)):
        slow = sky_map(AlcubierreMetric(speed=speed * C_LIGHT, radius=100.0,
                                        sigma=1.0))
        assert slow.sky_brightness() == pytest.approx(
            unlensed_brightness(speed), rel=tolerance)

    assert unlensed_brightness(SPEED_RATIO) == pytest.approx(
        (1.0 + SPEED_RATIO) ** 5 / (10.0 * SPEED_RATIO))
