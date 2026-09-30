# ==========================================================
# Tests for the Van Den Broeck metric
#
# The comparison with the symbolic derivation lives in test_symbolic.py;
# these pin the energy budget against a closed form, against the
# published numbers, and against the Alcubierre limit.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest

from warpdrive import (
    AlcubierreMetric,
    BroeckMetric,
    ConePocket,
    curvature_radius_by_order,
    format_profile,
    neck_scaling,
    pocket_energy_floor,
    profile_mission,
    quantum_inequality_check,
    quantum_inequality_threshold,
)
from warpdrive.constants import C_LIGHT, G, L_PLANCK
from warpdrive.geodesics import throat_radius
from warpdrive.shapes import broeck_volume_profile_derivative


@pytest.fixture(scope="module")
def metric():
    return BroeckMetric(speed=10.0 * C_LIGHT)


@pytest.fixture(scope="module")
def paper():
    return BroeckMetric.from_paper()


def test_net_transition_energy_matches_the_closed_form(paper):
    """
    Writing psi = sqrt(B), the static density is -(c^4/8 pi G) 8 psi^-5
    lap(psi) / 2 and sqrt(gamma) = psi^6, so integrating by parts gives

        E_B = (c^4 / 2 G) \\int (B'^2 / B) r^2 dr > 0,

    a positive net budget however the density is distributed in sign.
    """

    budget = paper.transition_energy_budget()

    n = 200000
    step = paper.thickness / n
    r = paper.inner_radius + (np.arange(n) + 0.5) * step
    first = broeck_volume_profile_derivative(
        r, paper.inner_radius, paper.thickness, paper.alpha, paper.order)
    closed = (C_LIGHT ** 4 / (2.0 * G)
              * np.sum(first ** 2 / paper.conformal_profile(r) * r ** 2)
              * step)

    assert budget.net > 0.0
    assert budget.net == pytest.approx(closed, rel=1e-5)


def test_published_transition_energies(paper):
    """
    Eqs. (15)-(16) of the 1999 paper, given to two significant figures:
    E_II,- = -1.4e30 kg and E_II,+ = 4.9e30 kg.
    """

    budget = paper.transition_energy_budget()

    assert budget.negative_mass == pytest.approx(-1.4e30, rel=0.05)
    assert budget.positive_mass == pytest.approx(4.9e30, rel=0.05)


def test_published_sign_change(paper):
    """The density turns positive at w = 0.981, w = (R~ + D~ - r) / D~."""

    def density_at(w):
        r = paper.inner_radius + paper.thickness * (1.0 - w)
        return float(paper.energy_density(r, 0.0, 0.0))

    assert density_at(0.975) < 0.0
    assert density_at(0.987) > 0.0


def test_published_pocket_size(paper):
    """A 200 m pocket diameter behind a 3e-15 m shift wall."""

    assert 2.0 * paper.pocket_proper_radius() == pytest.approx(200.0)


def test_paper_budget_is_finite_and_the_wall_is_refused(paper):
    """
    The Planck-thin shift wall is out of reach of any grid in r: the
    closed form handles it, the generic routines must refuse it.
    """

    budget = paper.energy_budget_analytic()
    transition = paper.transition_energy_budget()

    assert np.isfinite(budget.negative) and np.isfinite(budget.positive)
    assert budget.negative < transition.negative
    assert budget.positive == transition.positive
    with pytest.raises(ValueError):
        paper.energy_budget()


def test_generic_quadrature_matches_the_closed_form(metric):
    """Independent paths through sqrt(gamma) = B^3, for both signs."""

    analytic = metric.energy_budget_analytic()
    numeric = metric.energy_budget()

    assert numeric.negative == pytest.approx(analytic.negative, rel=1e-4)
    assert numeric.positive == pytest.approx(analytic.positive, rel=1e-3)


def test_density_is_not_sign_definite(metric):
    """The Alcubierre invariant eps <= 0 must not carry over."""

    r = np.linspace(metric.inner_radius, metric.inner_radius
                    + metric.thickness, 20001)
    eps = metric.energy_density(r, 0.0, 0.0)

    assert eps.min() < 0.0
    assert eps.max() > 0.0


def test_no_inflation_reduces_to_alcubierre():
    broeck = BroeckMetric(speed=10.0 * C_LIGHT, alpha=0.0)
    alcubierre = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=broeck.radius,
                                  sigma=broeck.sigma)

    x, y = np.meshgrid(np.linspace(-150.0, 150.0, 61),
                       np.linspace(0.0, 150.0, 31))
    assert np.array_equal(broeck.energy_density(x, y, 0.0),
                          alcubierre.energy_density(x, y, 0.0))
    assert np.all(broeck.conformal_factor(x, y, 0.0) == 1.0)

    budget = broeck.energy_budget_analytic()
    reference = alcubierre.energy_budget_analytic()
    assert budget.negative == reference.negative
    assert budget.positive == 0.0


def test_interface_quantities_see_only_the_shift_wall(metric):
    """
    The ship sits in the pocket but comoves with it, so B drops out of
    dtau/dt; the horizon lies in the shift wall, where B = 1, so it is
    Alcubierre's.
    """

    alcubierre = AlcubierreMetric(speed=metric.speed, radius=metric.radius,
                                  sigma=metric.sigma)

    assert metric.proper_time_rate() == pytest.approx(1.0)
    assert metric.horizon_offset() == pytest.approx(
        alcubierre.horizon_offset(), rel=1e-9
    )


def test_overlapping_regions_are_rejected():
    with pytest.raises(ValueError):
        BroeckMetric(speed=C_LIGHT, radius=100.0, sigma=0.1)


def test_invalid_profile_parameters_are_rejected():
    with pytest.raises(ValueError):
        BroeckMetric(speed=C_LIGHT, alpha=-2.0)


# --- Neck radius scan ---
NECKS = np.logspace(math.log10(3.0e-15), 1.0, 8)


@pytest.fixture(scope="module")
def scan():
    return neck_scaling(NECKS)


def test_neck_scan_starts_from_the_paper(scan, paper):
    """R = 3e-15 m with a 100 m pocket is exactly the 1999 configuration."""

    transition = paper.transition_energy_budget()
    total = paper.energy_budget_analytic()

    assert scan.transition_negative[0] == pytest.approx(transition.negative,
                                                        rel=1e-6)
    assert scan.transition_positive[0] == pytest.approx(transition.positive,
                                                        rel=1e-6)
    assert scan.wall[0] == pytest.approx(total.negative - transition.negative,
                                         rel=1e-9)


def test_wall_energy_scales_with_the_square_of_the_neck(scan):
    slopes = np.diff(np.log(-scan.wall)) / np.diff(np.log(scan.neck_radius))
    assert np.allclose(slopes, 2.0, atol=1e-6)


def test_transition_energy_does_not_depend_on_the_neck(scan):
    """
    With R~ = D~ and (1 + alpha) R~ fixed, the transition region only sees
    the pocket, as long as the pocket is much larger than the neck.
    """

    small = scan.neck_radius < 1.0e-2
    for part in (scan.transition_negative, scan.transition_positive):
        values = part[small]
        assert np.ptp(values) < 0.01 * np.abs(values).mean()


def test_alcubierre_reference_is_a_pocket_sized_bubble(scan):
    reference = AlcubierreMetric(speed=C_LIGHT, radius=100.0,
                                 sigma=1.0 / (1.0e2 * L_PLANCK))

    assert scan.alcubierre == reference.energy_budget_analytic().negative
    assert np.all(scan.total_negative > scan.alcubierre)


def test_neck_larger_than_the_pocket_is_rejected():
    with pytest.raises(ValueError):
        neck_scaling([400.0])


def test_mission_report_for_the_paper_configuration():
    """
    At 10c the horizon of the paper's bubble sits in a Planck-thin wall
    that cannot be bisected: the report must say so rather than fail.
    """

    profile = profile_mission(BroeckMetric.from_paper(speed=10.0 * C_LIGHT))
    report = format_profile(profile)

    assert profile.horizon is None
    assert not profile.horizon_resolved
    assert profile.pocket_radius == pytest.approx(100.0)
    assert profile.energy.positive > 0.0
    assert "too thin" in report
    assert "pocket" in report


# --- The pocket as a static throat ---
@pytest.mark.parametrize("alpha, order", [(2.0, 80), (10.0, 10), (10.0, 80),
                                          (1.0e3, 300)])
def test_null_energy_is_violated_at_the_throat(alpha, order):
    """
    A pocket larger inside than outside has a minimal sphere, and the
    flare-out there forces eps + p_r < 0, whatever the profile.
    """

    pocket = BroeckMetric(speed=C_LIGHT, alpha=alpha, order=order)
    assert (1.0 + alpha) * pocket.inner_radius > (pocket.inner_radius
                                                 + pocket.thickness)

    throat = pocket.throat()
    assert throat is not None
    radius, areal = throat
    assert pocket.inner_radius < radius < (pocket.inner_radius
                                           + pocket.thickness)
    assert areal < pocket.inner_radius + pocket.thickness
    assert float(pocket.pocket_null_energy(radius)) < 0.0


def test_throat_matches_the_ray_tracer_throat():
    """
    Two searches for the same minimum: throat() refines it with a
    parabola, throat_radius takes the smallest grid sample, which sits
    slightly above it.
    """

    pocket = BroeckMetric(speed=10.0 * C_LIGHT)
    refined = pocket.throat()[1]
    sampled = throat_radius(pocket, pocket.inner_radius)

    assert refined <= sampled
    assert refined == pytest.approx(sampled, rel=1e-8)


def test_null_energy_is_the_convexity_of_the_areal_radius():
    """
    eps + p_r = -(c^4/8 pi G) (2/A) d^2A/dl^2, checked against a finite
    difference of A along the proper radial distance l.
    """

    pocket = BroeckMetric(speed=C_LIGHT)
    r = np.linspace(pocket.inner_radius + 0.05, pocket.inner_radius
                    + pocket.thickness - 0.05, 200001)
    areal = pocket.areal_radius(r)
    proper = np.concatenate([[0.0], np.cumsum(
        0.5 * (pocket.conformal_profile(r[1:])
               + pocket.conformal_profile(r[:-1])) * np.diff(r))])
    convexity = np.gradient(np.gradient(areal, proper), proper)
    geometric = -C_LIGHT ** 4 / (8.0 * math.pi * G) * 2.0 * convexity / areal

    inner = slice(1000, -1000)
    scale = np.abs(geometric[inner]).max()
    assert np.allclose(pocket.pocket_null_energy(r)[inner], geometric[inner],
                       atol=1e-4 * scale)


def test_null_energy_integrates_to_zero_across_the_pocket():
    """
    dA/dl is 1 in the flat pocket and 1 outside, so
    \\int (eps + p_r) A dl = -(c^4/4 pi G) [dA/dl] = 0: the violation at
    the throat is balanced exactly by the region round the maximum of A.
    """

    pocket = BroeckMetric(speed=C_LIGHT)
    n = 400000
    step = pocket.thickness / n
    r = pocket.inner_radius + (np.arange(n) + 0.5) * step
    weight = (pocket.pocket_null_energy(r) * pocket.areal_radius(r)
              * pocket.conformal_profile(r))

    assert abs(weight.sum()) < 1e-5 * np.abs(weight).sum()


def test_null_energy_holds_where_the_areal_radius_peaks():
    """The violation is localised: near the maximum of A, A'' < 0."""

    pocket = BroeckMetric(speed=C_LIGHT)
    r = np.linspace(pocket.inner_radius, pocket.inner_radius
                    + pocket.thickness, 200001)
    null = pocket.pocket_null_energy(r)
    peak = r[np.argmax(pocket.areal_radius(r))]

    assert null.max() > 0.0
    assert float(pocket.pocket_null_energy(peak)) > 0.0
    assert peak < pocket.throat()[0]


def test_no_inflation_no_throat_no_stress():
    flat = BroeckMetric(speed=C_LIGHT, alpha=0.0)
    r = np.linspace(1.0, 30.0, 50)

    assert flat.throat() is None
    assert np.all(flat.pocket_null_energy(r) == 0.0)
    assert all(np.all(p == 0.0) for p in flat.pocket_pressures(r))


def test_paper_pocket_violates_the_null_energy_at_its_throat(paper):
    radius, areal = paper.throat()

    assert areal == pytest.approx(1.463e-15, rel=1e-3)
    assert float(paper.pocket_null_energy(radius)) < 0.0



# --- A lower bound on E_tot ---
def _dirichlet(psi_value, psi_slope, inner, outer, n=400000):
    """(2 c^4/G) int psi'^2 r^2 dr by the midpoint rule, for callables."""

    step = (outer - inner) / n
    r = inner + (np.arange(n) + 0.5) * step
    return 2.0 * C_LIGHT ** 4 / G * np.sum(psi_slope(r) ** 2 * r ** 2) * step


@pytest.mark.parametrize("alpha, order", [(2.0, 3), (10.0, 10), (10.0, 80),
                                          (10.0, 300), (1.0e3, 80)])
def test_every_profile_lies_above_the_dirichlet_bound(alpha, order):
    pocket = BroeckMetric(speed=C_LIGHT, alpha=alpha, order=order)
    assert pocket.transition_energy_budget().net > pocket.pocket_energy_bound()


def test_paper_profile_is_six_times_the_bound(paper):
    ratio = (paper.transition_energy_budget().net
             / paper.pocket_energy_bound())
    assert ratio == pytest.approx(6.47, rel=0.01)


def test_smooth_profiles_approach_the_harmonic_bound():
    """
    The harmonic psi = C1 + C2/r, rounded off at both edges over a width
    delta by cubic Hermite pieces with zero slope, is a smooth admissible
    profile. Its E_tot lies above the bound and tends to it linearly as
    delta -> 0: 3.3 % at delta = 1 m, 3.3e-4 at 1 cm.
    """

    a, b, alpha = 10.0, 20.0, 10.0
    top = math.sqrt(1.0 + alpha)
    c2 = (top - 1.0) / (1.0 / a - 1.0 / b)
    c1 = 1.0 - c2 / b
    harmonic = lambda r: c1 + c2 / r
    harmonic_slope = lambda r: -c2 / r ** 2
    bound = BroeckMetric(speed=C_LIGHT, inner_radius=a, thickness=b - a,
                         alpha=alpha).pocket_energy_bound()

    def rounded_slope(delta):
        def hermite_slope(r, r0, r1, v0, v1, s0, s1):
            h = r1 - r0
            t = (r - r0) / h
            dh00 = 6 * t ** 2 - 6 * t
            dh10 = 3 * t ** 2 - 4 * t + 1
            dh01 = -6 * t ** 2 + 6 * t
            dh11 = 3 * t ** 2 - 2 * t
            return (dh00 * v0 + dh10 * h * s0 + dh01 * v1 + dh11 * h * s1) / h

        def slope(r):
            left = r < a + delta
            right = r > b - delta
            out = harmonic_slope(r)
            out = np.where(left, hermite_slope(
                r, a, a + delta, top, harmonic(a + delta), 0.0,
                harmonic_slope(a + delta)), out)
            out = np.where(right, hermite_slope(
                r, b - delta, b, harmonic(b - delta), 1.0,
                harmonic_slope(b - delta), 0.0), out)
            return out
        return slope

    energies = [_dirichlet(None, rounded_slope(d), a, b)
                for d in (1.0, 0.1, 0.01)]

    excess = [e / bound - 1.0 for e in energies]
    assert all(x > 0.0 for x in excess)
    # the excess falls linearly with the width of the rounding
    assert excess[0] / excess[1] == pytest.approx(10.0, rel=0.05)
    assert excess[1] / excess[2] == pytest.approx(10.0, rel=0.05)


def test_floor_is_the_minimum_of_the_bound_over_the_inner_radius():
    """
    At fixed pocket radius P and outer radius b the bound over the inner
    radius a is smallest at a* = b^2/P, where it equals (2c^4/G)(P - b).
    """

    P, b = 110.0, 20.0
    floor, best = pocket_energy_floor(P, b)
    assert floor == pytest.approx(2.0 * C_LIGHT ** 4 / G * (P - b))
    assert best == pytest.approx(b ** 2 / P)

    for a in (1.0, best, 6.0, 12.0, 18.0):
        pocket = BroeckMetric(speed=C_LIGHT, inner_radius=a, thickness=b - a,
                              alpha=P / a - 1.0, radius=100.0)
        assert pocket.pocket_energy_bound() >= floor * (1.0 - 1e-12)
    optimal = BroeckMetric(speed=C_LIGHT, inner_radius=best,
                           thickness=b - best, alpha=P / best - 1.0)
    assert optimal.pocket_energy_bound() == pytest.approx(floor, rel=1e-12)


def test_floor_numbers_and_no_floor_without_excess(paper):
    floor, _ = pocket_energy_floor(paper.pocket_proper_radius(),
                                   paper.inner_radius + paper.thickness)
    assert floor / C_LIGHT ** 2 == pytest.approx(2.693e29, rel=1e-3)
    assert paper.transition_energy_budget().net > floor

    assert pocket_energy_floor(15.0, 20.0) == (0.0, None)


# --- The quantum-inequality check of the 1999 paper ---
def _pocket(alpha, thickness):
    """R~ = D~, R = 3 D~, as in the paper, with a wall thin enough to fit."""

    return BroeckMetric(speed=C_LIGHT, radius=3.0 * thickness,
                        sigma=1.0e2 / thickness, inner_radius=thickness,
                        thickness=thickness, alpha=alpha)


@pytest.fixture(scope="module")
def checked():
    """The parameters the paper's quantum-inequality check actually uses."""

    return quantum_inequality_check(_pocket(1.0e34, 1.0e-32))


def test_curvature_radius_of_the_paper_check(checked):
    """Eq. (20): r_c = D~ / 72.5 = 1.4e-34 m, near w = 0.349."""

    assert 1.0e-32 / checked.curvature_radius == pytest.approx(72.46,
                                                               rel=1e-3)
    w = (2.0e-32 - checked.peak_radius) / 1.0e-32
    assert w == pytest.approx(0.349, abs=1e-3)


def test_quantum_inequality_numbers_of_the_paper(checked):
    """Eq. (22): -6.6e93 kg/m^3 against a limit of -9.2e94 kg/m^3."""

    assert checked.peak_density == pytest.approx(-6.63e93, rel=2e-3)
    assert checked.limit == pytest.approx(-9.21e94, rel=2e-3)
    assert not checked.violated


def test_printed_parameters_violate_the_inequality(paper):
    """
    With the alpha = 1e17, D~ = 1e-15 m of eq. (7), which give the paper's
    energies, the same check fails by 33 orders of magnitude.
    """

    check = quantum_inequality_check(paper)
    assert paper.thickness / check.curvature_radius == pytest.approx(
        43.39, rel=1e-3)
    assert check.peak_density == pytest.approx(-2.37e59, rel=2e-3)
    assert check.limit == pytest.approx(-1.184e26, rel=1e-3)
    assert check.margin == pytest.approx(2.0e33, rel=1e-2)
    assert check.violated


def test_margin_scales_with_the_square_of_the_thickness(checked):
    """Peak ~ 1/D~^2 and limit ~ 1/r_c^4 ~ 1/D~^4 at fixed alpha."""

    for scale in (0.5, 2.0):
        other = quantum_inequality_check(_pocket(1.0e34, scale * 1.0e-32))
        assert other.margin / checked.margin == pytest.approx(scale ** 2,
                                                              rel=1e-6)


def test_radial_curvature_is_minus_the_tangential_pressure(paper):
    """R_{r th r th} = -p_t in units of c^4 / 8 pi G, as derive_pocket finds."""

    r = paper._transition_grid(1000)
    radial, _ = paper.riemann_components(r)
    _, tangential = paper.pocket_pressures(r)
    scale = C_LIGHT ** 4 / (8.0 * math.pi * G)
    np.testing.assert_allclose(radial, -tangential / scale, rtol=1e-10)


def test_no_inflation_no_curvature():
    flat = BroeckMetric(speed=C_LIGHT, alpha=0.0)
    r = flat._transition_grid(100)
    for component in flat.riemann_components(r):
        assert np.all(component == 0.0)


def test_observers_at_rest_are_the_most_restrictive(checked):
    assert checked.worst_speed == 0.0


class _SlabPocket:
    """Uniform eps = -1 and p_r = -3: the margin peaks at v^2 = 1/3."""

    def _transition_grid(self, n_points):
        return np.linspace(1.0, 2.0, n_points)

    def _conformal_density(self, r):
        return -np.ones_like(r) * C_LIGHT ** 2

    def pocket_pressures(self, r):
        return -3.0 * np.ones_like(r) * C_LIGHT ** 2, np.zeros_like(r)

    def curvature_radius(self, n_points):
        return 1.0, 1.5


def test_moving_observers_win_when_null_contraction_dominates():
    """
    (eps + v^2 p_r)(1 - v^2) = (1 + 3 v^2)(1 - v^2) is largest at
    v^2 = 1/3, where it is 4/3 of its value at rest.
    """

    at_rest = quantum_inequality_check(_SlabPocket(), n_speeds=1)
    moving = quantum_inequality_check(_SlabPocket(), n_speeds=10000)
    assert moving.worst_speed == pytest.approx(1.0 / math.sqrt(3.0),
                                               abs=1e-4)
    assert moving.margin / at_rest.margin == pytest.approx(4.0 / 3.0,
                                                           rel=1e-6)


# --- The cheapest pocket under a curvature bound ---
def test_areal_form_of_the_energy_matches_the_quadrature(metric, paper):
    """E_tot = (c^4/2G) int (1 + A'^2) dl + (c^4/G)(P - b), for any pocket."""

    for pocket in (metric, paper, BroeckMetric(speed=C_LIGHT, alpha=3.0,
                                               order=10)):
        assert pocket.areal_energy() == pytest.approx(
            pocket.transition_energy_budget().net, rel=1e-6)


def test_cone_approaches_the_floor_as_the_square_of_r_min():
    """
    Excess over the floor (L_1 + L_2) / 3 (P - b), with L_1 = 2 r^2 / P
    and L_2 ~ 2 r^2 / b: 5e-8 for P = 100 m, b = 20 m, r = 1 cm.
    """

    floor, _ = pocket_energy_floor(100.0, 20.0)
    excess = [ConePocket(100.0, 20.0, r).energy() / floor - 1.0
              for r in (1.0e-2, 1.0e-3)]
    assert excess[0] == pytest.approx(5.0e-8, rel=1e-3)
    assert excess[0] / excess[1] == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("r_min", [1.0e-3, 0.1, 5.0])
def test_cone_has_exactly_the_requested_curvature_radius(r_min):
    """The radial curvature |A''|/A reaches 1/r_min^2 and nothing exceeds it."""

    cone = ConePocket(100.0, 20.0, r_min)
    assert cone.curvature_radius() == pytest.approx(r_min, rel=1e-6)
    radial, tangential = cone.riemann_components()
    assert tangential <= radial

    # the inner corner reaches 1/r_min^2 by construction; so must the
    # outer one, at the bottom of its dip
    _, areal, _ = cone._outer_corner(10001)
    assert np.max(cone.outer_rate / areal) * r_min ** 2 == pytest.approx(
        1.0, rel=1e-6)


@pytest.mark.parametrize("alpha, order", [(10.0, 80), (3.0, 10),
                                          (1.0e3, 4)])
def test_no_profile_beats_the_cone(alpha, order):
    """At the same P, b and curvature radius, polynomials cost more."""

    pocket = BroeckMetric(speed=C_LIGHT, alpha=alpha, order=order)
    outer = pocket.inner_radius + pocket.thickness
    r_c, _ = pocket.curvature_radius()
    cone = ConePocket(pocket.pocket_proper_radius(), outer, r_c)
    assert pocket.transition_energy_budget().net > cone.energy()


def test_quantum_inequality_caps_the_curvature_radius():
    """
    Corner density -c^4 / 4 pi G r^2 against a limit ~ 1/r^4: the ratio is
    (8 pi / 3) beta^4 (r / l_P)^2, so the check holds only below
    sqrt(3 / 8 pi) l_P / beta^2 = 34.5 Planck lengths.
    """

    threshold = quantum_inequality_threshold()
    assert threshold / L_PLANCK == pytest.approx(34.549, rel=1e-4)

    for ratio in (0.3, 0.9, 1.1):
        check = ConePocket(100.0, 2.0e-15,
                           ratio * threshold).quantum_inequality()
        assert check.margin == pytest.approx(ratio ** 2, rel=1e-6)
        assert check.worst_speed == 0.0


def test_best_order_of_the_polynomial():
    """
    n = 80 maximises r_c for alpha = 1e34 to 0.1%, the paper's claim for
    its first configuration; for the alpha = 1e17 of eq. (7) it is n = 44.
    """

    orders = np.arange(38, 92)
    for alpha, best, ratio in ((1.0e34, 84, 72.38), (1.0e17, 44, 37.29)):
        ratios = curvature_radius_by_order(alpha, orders)
        k = int(np.argmin(ratios))
        assert orders[k] == best
        assert ratios[k] == pytest.approx(ratio, rel=1e-3)

    at_80 = curvature_radius_by_order(1.0e34, [80])[0]
    assert at_80 == pytest.approx(72.46, rel=1e-3)


def test_cone_rejects_impossible_pockets():
    with pytest.raises(ValueError):
        ConePocket(10.0, 20.0, 0.1)
    with pytest.raises(ValueError):
        ConePocket(100.0, 20.0, 15.0)
