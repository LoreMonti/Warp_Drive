# ==========================================================
# Tests for the symbolic derivation of the stress-energy tensor
#
# The derivation in `symbolic.py` is the source of truth for the physics;
# the expressions in `metrics/` are the fast numpy path. These tests are
# what holds the two together, and they are the only place in the suite
# that constrains the overall prefactor: everything else pins signs,
# symmetries and scaling, none of which fix a constant.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import pytest
import sympy as sp

from warpdrive import AlcubierreMetric, BroeckMetric
from warpdrive.constants import C_LIGHT, G
from warpdrive.shapes import (
    broeck_volume_profile,
    broeck_volume_profile_derivative,
    broeck_volume_profile_second_derivative,
    tanh_top_hat,
    tanh_top_hat_derivative,
)
from warpdrive.symbolic import (
    alcubierre_expansion_reference,
    alcubierre_reference,
    check_inverse,
    derive,
    numeric_lambda,
)


RADIUS = 100.0
SIGMA = 0.1
SPEED_RATIO = 10.0


@pytest.fixture(scope="module")
def abstract():
    """Derivation with the shape function left abstract."""

    return derive()


@pytest.fixture(scope="module")
def numeric(abstract):
    """
    The derived expressions as numpy callables, with the very same shape
    functions the metric itself uses plugged into the abstract profile.
    """

    def shape(r):
        return tanh_top_hat(r, RADIUS, SIGMA)

    def derivative(r):
        return tanh_top_hat_derivative(r, RADIUS, SIGMA)

    profiles = {"f": (shape, derivative, None)}
    return {
        "energy_density": numeric_lambda(abstract, abstract["energy_density"],
                                         profiles),
        "expansion": numeric_lambda(abstract, abstract["expansion"], profiles),
    }


@pytest.fixture(scope="module")
def metric():
    return AlcubierreMetric(speed=SPEED_RATIO * C_LIGHT, radius=RADIUS,
                            sigma=SIGMA)


def test_analytic_inverse_is_the_actual_inverse(abstract):
    """The ADM inverse is written by hand rather than computed; check it."""

    assert check_inverse(abstract["metric"], abstract["inverse"]) == sp.eye(4)


def test_metric_keeps_its_time_dependence(abstract):
    """
    The bubble moves, so the slices are not static. If the w dependence
    were dropped the derivation would silently lose the time derivative
    of the spatial metric, which matters as soon as B is not constant.
    """

    w = abstract["coords"][0]
    assert abstract["metric"][0, 1].has(w)


def test_energy_density_reproduces_alcubierre(abstract):
    """
    The Einstein tensor of the ansatz, contracted on the Eulerian normal,
    must return the published density

        eps = -(c^4/8 pi G)(v_s^2/c^2)(rho^2/4 r_s^2)(df/dr_s)^2

    up to the c^4/8 pi G factor carried outside the symbolic part.
    """

    difference = sp.simplify(
        abstract["energy_density"] - alcubierre_reference(abstract)
    )
    assert difference == 0


def test_expansion_reproduces_alcubierre(abstract):
    """theta = v_s (x_s/r_s) df/dr_s, per unit w rather than per second."""

    difference = sp.simplify(
        abstract["expansion"] - alcubierre_expansion_reference(abstract)
    )
    assert difference == 0


def test_numerical_energy_density_matches_the_derivation(numeric, metric):
    """
    The bridge between the two implementations, and the only test in the
    suite that constrains the c^4/8 pi G prefactor: everything else pins
    signs, symmetries and scaling, none of which fix an overall constant.
    """

    factor = C_LIGHT ** 4 / (8.0 * math.pi * G)
    samples = [(30.0, 40.0, 0.0), (0.0, 100.0, 0.0), (60.0, 60.0, 60.0),
               (-90.0, 25.0, 10.0), (120.0, 5.0, 5.0)]

    for x, y, z in samples:
        derived = factor * float(
            numeric["energy_density"](0.0, x, y, z, SPEED_RATIO)
        )
        implemented = float(metric.energy_density(x, y, z))
        assert derived == pytest.approx(implemented, rel=1e-10)
        assert implemented < 0.0


def test_numerical_expansion_matches_the_derivation(numeric, metric):
    """theta is per unit w in the derivation, so it needs one factor of c."""

    for x, y, z in [(90.0, 20.0, 0.0), (-110.0, 0.0, 30.0), (100.0, 0.0, 0.0)]:
        derived = C_LIGHT * float(numeric["expansion"](0.0, x, y, z,
                                                       SPEED_RATIO))
        implemented = float(metric.expansion(x, y, z))
        assert derived == pytest.approx(implemented, rel=1e-10)


def test_flat_interior_and_exterior_carry_no_energy(numeric):
    """
    Both far inside and far outside the wall f is constant, so the
    derived density vanishes: the exotic matter lives only in the wall.
    """

    inside = float(numeric["energy_density"](0.0, 1.0, 1.0, 1.0, SPEED_RATIO))
    outside = float(numeric["energy_density"](0.0, 400.0, 30.0, 0.0,
                                              SPEED_RATIO))
    assert inside == pytest.approx(0.0, abs=1e-12)
    assert outside == pytest.approx(0.0, abs=1e-12)


@pytest.fixture(scope="module")
def general():
    """Derivation with both profiles abstract; the slow one, ~9 s."""

    return derive(conformal=True)


def test_conformal_factor_enters_the_energy_density(general):
    """
    With B carried abstractly the density must actually depend on it,
    otherwise the Van Den Broeck variant could never differ from
    Alcubierre and the whole interface would be pointless.
    """

    assert general["energy_density"].has(sp.Function("B"))
    assert general["energy_density"] != alcubierre_reference(general)


@pytest.fixture(scope="module")
def broeck():
    return BroeckMetric(speed=SPEED_RATIO * C_LIGHT)


@pytest.fixture(scope="module")
def broeck_numeric(general, broeck):
    """
    The general derivation with the Van Den Broeck profiles plugged in.
    It still carries the coupling terms proportional to B' (1 - f), which
    `BroeckMetric` drops; in the separated configuration they are below
    e^-100 of the terms kept.
    """

    m = broeck
    shape = (lambda r: tanh_top_hat(r, m.radius, m.sigma),
             lambda r: tanh_top_hat_derivative(r, m.radius, m.sigma),
             None)
    parameters = (m.inner_radius, m.thickness, m.alpha, m.order)
    conformal = (
        lambda r: broeck_volume_profile(r, *parameters),
        lambda r: broeck_volume_profile_derivative(r, *parameters),
        lambda r: broeck_volume_profile_second_derivative(r, *parameters),
    )
    profiles = {"f": shape, "B": conformal}
    return {
        "energy_density": numeric_lambda(general, general["energy_density"],
                                         profiles),
        "expansion": numeric_lambda(general, general["expansion"], profiles),
    }


# pocket, transition region (both signs of eps_B), flat interior, shift wall
BROECK_SAMPLES = [(3.0, 4.0, 0.0), (12.0, 5.0, 0.0), (0.0, 0.0, 19.9),
                  (-14.0, 8.0, 3.0), (50.0, 0.0, 10.0), (95.0, 20.0, 0.0),
                  (-70.0, 70.0, 0.0), (0.0, 101.0, 0.0)]


def test_broeck_energy_density_matches_the_derivation(broeck_numeric,
                                                      broeck):
    """
    Pins the c^4/8 pi G prefactor and every coefficient of the static
    term of the transition region, which nothing else in the suite
    constrains: the closed form for the net budget follows from the same
    expression, and the published numbers only fix two figures.
    """

    factor = C_LIGHT ** 4 / (8.0 * math.pi * G)
    for x, y, z in BROECK_SAMPLES:
        derived = factor * float(
            broeck_numeric["energy_density"](0.0, x, y, z, SPEED_RATIO)
        )
        implemented = float(broeck.energy_density(x, y, z))
        assert derived == pytest.approx(implemented, rel=1e-10,
                                        abs=1e-10 * factor)


def test_broeck_expansion_matches_the_derivation(broeck_numeric, broeck):
    for x, y, z in BROECK_SAMPLES:
        derived = C_LIGHT * float(broeck_numeric["expansion"](0.0, x, y, z,
                                                              SPEED_RATIO))
        implemented = float(broeck.expansion(x, y, z))
        assert derived == pytest.approx(implemented, rel=1e-10, abs=1e-12)
