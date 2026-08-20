# ==========================================================
# Symbolic derivation of the stress-energy tensor
#
# Derives the Einstein tensor of the warp bubble ansatz directly, rather
# than transcribing results from papers, and uses it to check the
# hand-written expressions in `metrics/`.
#
# Coordinates are (w, x, y, z) with w = c t, so every coordinate is a
# length and every metric component is dimensionless. The shift is then
# dimensionless too,
#
#     b(r_s) = beta / c = u f(r_s),      u = v_s / c,
#
# and the line element of the whole family reads
#
#     ds^2 = -dw^2 + B(r_s)^2 [ (dx - b(r_s) dw)^2 + dy^2 + dz^2 ]
#
# with r_s = sqrt((x - u w)^2 + y^2 + z^2): the bubble travels along x at
# dx/dw = u, so the metric genuinely depends on w and the time
# derivatives are kept rather than assumed away.
#
# This module needs sympy, which is an optional dependency:
#
#     pip install -e ".[symbolic]"
#
# It is deliberately not imported by `warpdrive/__init__.py`, so the
# runtime stays numpy + matplotlib.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import sympy as sp


# --- Simplification ---
def reduce_expr(expression):
    """
    Canonicalise a curvature expression.

    `sympy.simplify` on the Einstein contraction of this ansatz takes
    over a minute and returns the same thing; putting the expression
    over a common denominator and cancelling takes well under a second,
    because what needs collapsing is a rational function of the shape
    function and its derivatives, not a transcendental tangle.
    """

    return sp.cancel(sp.together(expression))


# --- Coordinates and profiles ---
def coordinates():
    """The chart (w, x, y, z), all four with dimensions of length."""

    return sp.symbols("w x y z", real=True)


def bubble_radius(coords, speed):
    """
    Distance from the centre of the bubble,

        r_s = sqrt((x - u w)^2 + y^2 + z^2)

    Keeping the w dependence explicit is the point: the slices are not
    static, and for a non-trivial B the time derivative of the spatial
    metric contributes to the extrinsic curvature.
    """

    w, x, y, z = coords
    return sp.sqrt((x - speed * w) ** 2 + y ** 2 + z ** 2)


# --- Metric ---
def metric(shift, conformal):
    """
    Covariant metric of the ansatz.

    In ADM language the lapse is alpha = 1, the shift is beta^x = -b and
    the spatial metric is gamma_ij = B^2 delta_ij.
    """

    g = sp.zeros(4, 4)
    g[0, 0] = -1 + conformal ** 2 * shift ** 2
    g[0, 1] = g[1, 0] = -conformal ** 2 * shift
    g[1, 1] = conformal ** 2
    g[2, 2] = conformal ** 2
    g[3, 3] = conformal ** 2
    return g


def inverse_metric(shift, conformal):
    """
    Contravariant metric, written analytically from the ADM relations

        g^{00} = -1/alpha^2,  g^{0i} = beta^i/alpha^2,
        g^{ij} = gamma^{ij} - beta^i beta^j / alpha^2

    rather than obtained by symbolic inversion, which is slow and
    produces an unreadable expression. `check_inverse` verifies it.
    """

    ginv = sp.zeros(4, 4)
    ginv[0, 0] = -1
    ginv[0, 1] = ginv[1, 0] = -shift
    ginv[1, 1] = 1 / conformal ** 2 - shift ** 2
    ginv[2, 2] = 1 / conformal ** 2
    ginv[3, 3] = 1 / conformal ** 2
    return ginv


def check_inverse(g, ginv):
    """Return g . g^{-1} simplified; must be the identity."""

    return sp.simplify(g * ginv)


# --- Curvature ---
def christoffel(g, ginv, coords):
    """
    Christoffel symbols of the second kind,

        Gamma^l_{mn} = 1/2 g^{ls} (d_m g_{sn} + d_n g_{sm} - d_s g_{mn})

    returned as a nested list indexed [l][m][n].
    """

    n = len(coords)
    dg = [[[sp.diff(g[i, j], coords[k]) for k in range(n)]
           for j in range(n)] for i in range(n)]

    gamma = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for m in range(n):
            for nu in range(m, n):
                total = sp.S.Zero
                for s in range(n):
                    if ginv[l, s] == 0:
                        continue
                    total += ginv[l, s] * (dg[s][nu][m] + dg[s][m][nu]
                                           - dg[m][nu][s])
                value = sp.together(total / 2)
                gamma[l][m][nu] = value
                gamma[l][nu][m] = value
    return gamma


def ricci_tensor(gamma, coords):
    """
    Ricci tensor contracted straight from the connection,

        R_{mn} = d_l Gamma^l_{mn} - d_n Gamma^l_{lm}
                 + Gamma^l_{ls} Gamma^s_{mn} - Gamma^l_{ns} Gamma^s_{lm}

    Going through the full Riemann tensor would cost 256 components for
    no gain: only the Ricci trace enters the Einstein tensor.
    """

    n = len(coords)
    ricci = sp.zeros(n, n)

    for m in range(n):
        for nu in range(m, n):
            total = sp.S.Zero
            for l in range(n):
                total += sp.diff(gamma[l][m][nu], coords[l])
                total -= sp.diff(gamma[l][l][m], coords[nu])
                for s in range(n):
                    total += gamma[l][l][s] * gamma[s][m][nu]
                    total -= gamma[l][nu][s] * gamma[s][l][m]
            value = sp.together(total)
            ricci[m, nu] = value
            ricci[nu, m] = value
    return ricci


def einstein_tensor(g, ginv, coords):
    """
    G_{mn} = R_{mn} - 1/2 g_{mn} R.

    Returns (G, R_{mn}, R).
    """

    gamma = christoffel(g, ginv, coords)
    ricci = ricci_tensor(gamma, coords)

    scalar = sp.S.Zero
    n = len(coords)
    for m in range(n):
        for nu in range(n):
            if ginv[m, nu] != 0:
                scalar += ginv[m, nu] * ricci[m, nu]
    scalar = reduce_expr(scalar)

    return ricci - g * scalar / 2, ricci, scalar


# --- Eulerian observers ---
def eulerian_normal(shift):
    """
    Future-pointing unit normal to the slices,

        n^mu = (1/alpha)(1, -beta^i) = (1, b, 0, 0),

    normalised so that g_{mn} n^m n^n = -1.
    """

    return sp.Matrix([1, shift, 0, 0])


def energy_density(einstein, normal):
    """
    Energy density measured by the Eulerian observers, in units of
    c^4 / 8 pi G:

        eps_hat = G_{mn} n^m n^n

    Multiply by c^4/(8 pi G) for joules per cubic metre.
    """

    total = sp.S.Zero
    for m in range(4):
        for nu in range(4):
            if einstein[m, nu] != 0:
                total += einstein[m, nu] * normal[m] * normal[nu]
    return reduce_expr(total)


def expansion(g, normal, coords):
    """
    Expansion of the normal congruence, computed as a divergence,

        theta_hat = nabla_mu n^mu = (1/sqrt(-g)) d_mu (sqrt(-g) n^mu)

    This is per unit w; multiply by c for units of s^-1.

    Using the divergence formula avoids any convention ambiguity in the
    sign of the extrinsic curvature.
    """

    root = sp.sqrt(-g.det())
    total = sp.S.Zero
    for m in range(4):
        total += sp.diff(root * normal[m], coords[m])
    return reduce_expr(total / root)


# --- High level derivation ---
def _profile(name, radius, r_s, active=True):
    """
    An abstract radial profile together with its first two derivatives,
    each evaluated at r_s in the unevaluated form sympy produces when
    differentiating through the chain rule.

    A derivative cannot be taken with respect to a compound expression,
    so it is taken with respect to a dummy radius and substituted back.
    """

    if not active:
        return sp.S.One, sp.S.Zero, sp.S.Zero

    func = sp.Function(name)
    return (
        func(r_s),
        sp.diff(func(radius), radius).subs(radius, r_s),
        sp.diff(func(radius), radius, 2).subs(radius, r_s),
    )


def derive(conformal=False, speed=None):
    """
    Derive the Eulerian energy density and the expansion scalar for the
    ansatz, keeping the radial profiles abstract.

    Parameters
    ----------
    conformal : False for the Alcubierre case B = 1; True to carry an
                abstract B(r_s), which is what the Van Den Broeck variant
                needs.
    speed     : the dimensionless u = v_s/c; defaults to a symbol.

    The profiles are deliberately left abstract. Substituting a concrete
    tanh before the algebra expands the expression into something
    thousands of terms long that neither simplifies nor lambdifies;
    `numeric_lambda` supplies the profiles numerically instead, at the
    point of evaluation, which is both faster and closer to what the
    numpy implementation actually does.

    Returns a dict with the symbols, the metric, the Einstein tensor and
    the two scalars, all still functions of (w, x, y, z).
    """

    coords = coordinates()
    u = sp.Symbol("u", positive=True) if speed is None else speed
    r_s = bubble_radius(coords, u)
    radius = sp.Symbol("r", positive=True)

    f_expr, f_prime, f_second = _profile("f", radius, r_s)
    B_expr, B_prime, B_second = _profile("B", radius, r_s, active=conformal)

    b = u * f_expr
    g = metric(b, B_expr)
    ginv = inverse_metric(b, B_expr)
    normal = eulerian_normal(b)

    G, ricci, scalar = einstein_tensor(g, ginv, coords)

    return {
        "coords": coords,
        "speed": u,
        "radius": r_s,
        "shape": f_expr,
        "shape_derivative": f_prime,
        "shape_second_derivative": f_second,
        "conformal": B_expr,
        "conformal_derivative": B_prime,
        "conformal_second_derivative": B_second,
        "metric": g,
        "inverse": ginv,
        "normal": normal,
        "einstein": G,
        "ricci": ricci,
        "curvature_scalar": scalar,
        "energy_density": energy_density(G, normal),
        "expansion": expansion(g, normal, coords),
    }


def _resolve_profile(expression, name):
    """
    Replace an abstract profile and its derivatives by named callables.

    sympy writes the chain rule as ``Subs(Derivative(f(_xi), _xi), _xi,
    r_s)``, where ``_xi`` is a fresh Dummy each time. Two such objects
    compare equal under `simplify` but not under `subs`, so the
    replacement has to be done by structure rather than by value.
    """

    func = sp.Function(name)
    named = {
        1: sp.Function(f"{name}_prime"),
        2: sp.Function(f"{name}_second"),
    }

    def handle(node):
        inner = node.expr
        if not isinstance(inner, sp.Derivative) or inner.expr.func != func:
            return node
        order = inner.derivative_count
        if order not in named:
            raise ValueError(f"derivative of order {order} is not supported")
        return named[order](node.point[0])

    expression = expression.replace(lambda node: isinstance(node, sp.Subs),
                                    handle)
    return expression.replace(func, sp.Function(f"{name}_value"))


def numeric_lambda(result, expression, profiles):
    """
    Turn a symbolic expression into a numpy callable ``(w, x, y, z, u)``,
    supplying the abstract profiles as real functions.

    `profiles` maps a profile name to the tuple of callables
    ``(value, first derivative, second derivative)``; a trailing None is
    allowed when that order does not appear in the expression.

    This is how the derivation is compared against `metrics/`: the very
    same numpy shape functions the metric uses are fed into the
    symbolically derived structure, so any mismatch can only come from
    the algebra.
    """

    table = {}
    for name, callables in profiles.items():
        expression = _resolve_profile(expression, name)
        for suffix, function in zip(("value", "prime", "second"), callables):
            if function is not None:
                table[f"{name}_{suffix}"] = function

    if expression.has(sp.Derivative) or expression.has(sp.Subs):
        raise ValueError("expression still contains unresolved derivatives")

    arguments = (*result["coords"], result["speed"])
    return sp.lambdify(arguments, expression, modules=[table, "numpy"])


def alcubierre_reference(result):
    """
    The published Alcubierre energy density, in the same units as
    `energy_density`:

        eps_hat = -u^2 rho^2 (df/dr_s)^2 / (4 r_s^2),   rho^2 = y^2 + z^2

    Reference: M. Alcubierre, Class. Quantum Grav. 11, L73 (1994), eq. 8.
    """

    _, _, y, z = result["coords"]
    u, r_s = result["speed"], result["radius"]
    derivative = result["shape_derivative"]

    return -u ** 2 * (y ** 2 + z ** 2) * derivative ** 2 / (4 * r_s ** 2)


def alcubierre_expansion_reference(result):
    """
    The published expansion scalar, per unit w:

        theta_hat = u (x_s / r_s) df/dr_s
    """

    w, x, _, _ = result["coords"]
    u, r_s = result["speed"], result["radius"]
    derivative = result["shape_derivative"]

    return u * (x - u * w) * derivative / r_s
