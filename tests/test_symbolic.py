# ==========================================================
# Tests for the symbolic derivation of the stress-energy tensor
#
# These close the loop on the numerical code: the expressions in
# `metrics/alcubierre.py` were written by hand from the 1994 paper, and
# nothing else in the suite constrains their overall prefactor. Deriving
# the Einstein tensor from the metric and comparing is what pins it.
#
# Author: Lorenzo Monti
# ==========================================================

import math

import numpy as np
import pytest

sp = pytest.importorskip("sympy", reason="needs the [symbolic] extra")

from warpdrive import AlcubierreMetric                       # noqa: E402
from warpdrive.constants import C_LIGHT, G                   # noqa: E402
from warpdrive.shapes import (                                # noqa: E402
    tanh_top_hat,
    tanh_top_hat_derivative,
)
from warpdrive.symbolic import (                             # noqa: E402
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


def test_conformal_factor_enters_the_energy_density():
    """
    With B carried abstractly the density must actually depend on it,
    otherwise the Van Den Broeck variant could never differ from
    Alcubierre and the whole interface would be pointless.
    """

    general = derive(conformal=True)
    assert general["energy_density"].has(sp.Function("B"))
    assert general["energy_density"] != alcubierre_reference(general)
