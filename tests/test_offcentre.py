# ==========================================================
# Tests for observers away from the centre of the bubble
#
# The three-dimensional tracer is checked against the two-dimensional
# one at the centre, against Bouguer's invariant for the rays that leave
# the pocket and those that stay in it, against the closed-form
# blueshift, and against an Alcubierre bubble, which has no pocket and
# traps nothing.
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
    TRAPPED,
    UNFINISHED,
    throat_radius,
    trace_rays,
    trace_rays_3d,
    visible_cone,
)

SPEED_RATIO = 10.0


@pytest.fixture(scope="module")
def broeck():
    return BroeckMetric(speed=SPEED_RATIO * C_LIGHT)


def _in_plane(psi):
    """Lines of sight at angle psi from +y towards +x, in the x-y plane."""

    psi = np.asarray(psi, dtype=float)
    return np.array([np.sin(psi), np.cos(psi), np.zeros_like(psi)])


def test_centre_reproduces_the_meridional_tracer(broeck):
    """
    Two integrators, one tolerance each of ~1e-8 over a path that crosses
    the pocket at c/B: they must agree to 1e-7.
    """

    angles = np.linspace(0.0, math.pi, 13)
    flat = trace_rays(broeck, angles)
    field = trace_rays_3d(broeck, [0.0, 0.0, 0.0],
                          np.array([np.cos(angles), np.sin(angles),
                                    np.zeros_like(angles)]))

    visible = flat.escaped & (flat.frequency_ratio > 1e-6)
    source = np.arccos(-field.far_direction[0])
    assert np.array_equal(field.status, flat.status)
    assert np.allclose(source[visible], flat.source_angle[visible],
                       atol=1e-7)
    assert np.allclose(field.frequency_ratio[visible],
                       flat.frequency_ratio[visible], rtol=1e-7)


def test_centre_view_is_symmetric_about_the_motion(broeck):
    """Rotating a line of sight about x rotates the source with it."""

    tilt = np.radians(35.0)
    turn = np.radians(np.array([0.0, 70.0, 160.0]))
    look = np.array([np.full(3, math.cos(tilt)),
                     math.sin(tilt) * np.cos(turn),
                     math.sin(tilt) * np.sin(turn)])
    field = trace_rays_3d(broeck, [0.0, 0.0, 0.0], look)

    assert np.all(field.escaped)
    assert np.allclose(field.far_direction[0], field.far_direction[0, 0],
                       atol=1e-9)
    azimuth = np.arctan2(-field.far_direction[2], -field.far_direction[1])
    assert np.allclose(np.unwrap(azimuth), turn, atol=1e-8)


def test_throat_is_the_narrowest_areal_radius(broeck):
    r = np.linspace(broeck.inner_radius, broeck.inner_radius
                    + broeck.thickness, 200001)
    areal = broeck.conformal_profile(r) * r

    assert throat_radius(broeck, 5.0) == pytest.approx(areal.min(),
                                                       rel=1e-6)
    alcubierre = AlcubierreMetric(speed=broeck.speed, radius=broeck.radius,
                                  sigma=broeck.sigma)
    assert throat_radius(alcubierre, 5.0) == pytest.approx(5.0)


@pytest.mark.parametrize("r0", [2.0, 5.0, 8.0])
def test_visible_cone_follows_bouguer(broeck, r0):
    """
    Rays just inside either cone leave the pocket; rays just outside
    stay in it, whether or not the Bouguer shortcut is used.
    """

    cone = visible_cone(broeck, r0)
    assert math.sin(cone) == pytest.approx(
        throat_radius(broeck, r0) / (broeck.conformal_profile(r0) * r0))

    inside = [0.95 * cone, math.pi - 0.95 * cone]
    outside = [1.05 * cone, math.pi - 1.05 * cone]
    observer = [0.0, r0, 0.0]

    kept = trace_rays_3d(broeck, observer, _in_plane(inside))
    assert np.all(kept.status != TRAPPED)
    assert np.all(np.linalg.norm(kept.end_position, axis=0)
                  > broeck.inner_radius + broeck.thickness)

    shortcut = trace_rays_3d(broeck, observer, _in_plane(outside))
    brute = trace_rays_3d(broeck, observer, _in_plane(outside),
                          classify_trapped=False, max_steps=12000)
    assert np.all(shortcut.status == TRAPPED)
    assert np.all(brute.status == UNFINISHED)
    assert np.all(np.linalg.norm(brute.end_position, axis=0)
                  < broeck.inner_radius + broeck.thickness)


def test_bouguer_invariant_is_the_initial_angular_momentum(broeck):
    psi = np.radians([0.0, 30.0, 90.0])
    field = trace_rays_3d(broeck, [0.0, 5.0, 0.0], _in_plane(psi))

    expected = broeck.conformal_profile(5.0) * 5.0 * np.sin(psi)
    assert np.allclose(field.bouguer, expected, atol=1e-12)


def test_blueshift_holds_off_the_centre(broeck):
    """The observer is at rest in the bubble, where b = 0, as at the centre."""

    cone = visible_cone(broeck, 5.0)
    psi = np.concatenate([np.linspace(0.0, 0.9 * cone, 7),
                          np.linspace(math.pi - 0.9 * cone, math.pi, 7)])
    field = trace_rays_3d(broeck, [0.0, 5.0, 0.0], _in_plane(psi))

    visible = field.escaped & (field.frequency_ratio > 1e-6)
    assert visible.sum() >= 10
    assert np.allclose(field.frequency_ratio[visible],
                       1.0 - SPEED_RATIO * field.far_direction[0, visible],
                       rtol=1e-8)
    assert np.all(field.hamiltonian_drift[visible] < 1e-7)


def test_alcubierre_traps_nothing_and_moves_the_sky_continuously():
    """
    No pocket, no throat. Moving the observer shifts the sky smoothly:
    the change in far-field direction is proportional to the offset.
    """

    alcubierre = AlcubierreMetric(speed=SPEED_RATIO * C_LIGHT, radius=100.0,
                                  sigma=1.0)
    angles = np.radians([0.0, 30.0, 60.0, 90.0])
    look = np.array([np.cos(angles), np.sin(angles), np.zeros_like(angles)])

    centre = trace_rays_3d(alcubierre, [0.0, 0.0, 0.0], look)
    near = trace_rays_3d(alcubierre, [0.0, 0.0, 5.0e-4], look)
    far = trace_rays_3d(alcubierre, [0.0, 0.0, 1.0e-3], look)

    assert visible_cone(alcubierre, 5.0) == pytest.approx(math.pi / 2.0)
    assert np.all(far.status == ESCAPED)

    shift_near = np.linalg.norm(near.far_direction - centre.far_direction,
                                axis=0)
    shift_far = np.linalg.norm(far.far_direction - centre.far_direction,
                               axis=0)
    moving = shift_far > 1e-6
    assert moving.sum() >= 3
    assert np.allclose(shift_far[moving] / shift_near[moving], 2.0,
                       rtol=0.02)


def test_paper_pocket_shows_the_universe_through_a_pinhole():
    """
    With alpha = 1e17 the throat is 1.46e-15 m across: a metre from the
    centre the whole outside fits in a cone of 1e-15 rad.
    """

    paper = BroeckMetric.from_paper()
    r0 = 1.0 / (1.0 + paper.alpha)          # one metre of proper distance
    assert throat_radius(paper, r0) == pytest.approx(1.463e-15, rel=1e-3)
    assert visible_cone(paper, r0) < 2.0e-15


def test_observer_outside_the_shift_wall_is_refused(broeck):
    with pytest.raises(ValueError):
        trace_rays_3d(broeck, [0.0, broeck.radius, 0.0], _in_plane([0.0]))
