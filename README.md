# Warp Drive

A numerical study of the **Alcubierre (1994) warp drive** metric — the geometry
behind every "IXS Enterprise"-style concept ship, with its two coaxial rings
wrapped around a central hull.

The metric, in ADM (3+1) form, is

```math
ds^2 = -c^2 dt^2 + \left(dx - v_s f(r_s)\,dt\right)^2 + dy^2 + dz^2
```

with the bubble centred on $x_s(t)$, $v_s = dx_s/dt$, and
$r_s = \sqrt{(x - x_s)^2 + y^2 + z^2}$. The shape function $f$ goes from 1
inside the bubble to 0 outside:

```math
f(r_s) = \frac{\tanh\left(\sigma(r_s + R)\right) - \tanh\left(\sigma(r_s - R)\right)}{2\tanh(\sigma R)}
```

Both regions are **exactly flat**. All the curvature lives in a wall of
thickness $\sim 1/\sigma$ at $r_s = R$. The ship never moves through space: it
sits at rest in a flat patch while the wall contracts space ahead of it and
expands it behind. Special relativity is never violated, because $v_s$ is a
*coordinate* velocity, not a local one.

![Alcubierre bubble sweeping past a field of test particles](images/readme/alcubierre_flyby.gif)

## Install

```bash
git clone https://github.com/LoreMonti/Warp_Drive.git
cd Warp_Drive
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

That one install command reads `pyproject.toml` and pulls in everything the
package needs: `numpy`, `matplotlib` and `sympy`, plus `pytest` from the `dev`
extra. Nothing at runtime imports sympy — it carries the derivation the rest of
the package is checked against, so it is a hard dependency rather than an
optional extra. Python 3.10 or newer; the results below were produced with
Python 3.10, numpy 2.2.6, matplotlib 3.10.9 and sympy 1.14.0.

## Usage

Three drivers reproduce the studies in this README. Each builds its bubbles,
writes its figures and prints a report, needs no arguments for the default run,
and lists its options with `--help`.

| script | what it does |
| --- | --- |
| `run_alcubierre.py` | the 1994 bubble: shape function, expansion scalar, exotic-matter shell, flyby animation and mission profile |
| `run_broeck.py` | the two-scale bubble: the same figures, the comparison with Alcubierre, the neck scan and the configuration of the 1999 paper |
| `run_sky.py` | null rays from the centre: the sky the crew sees at several speeds |
| `make_readme_images.py` | rebuilds the images shown in this README |

```bash
python scripts/run_sky.py
```

Figures go to `images/local/`, reports to `reports/`; neither is tracked, and
both are rebuilt by rerunning a script. The only tracked images are the ones in
this README, in `images/readme/`, with their parameters fixed in code.

As a library:

```python
from warpdrive import AlcubierreMetric, profile_mission, format_profile
from warpdrive.constants import C_LIGHT

metric = AlcubierreMetric(speed=10 * C_LIGHT, radius=100.0, sigma=0.1)
print(format_profile(profile_mission(metric)))
```

Van Den Broeck's bubble takes the same calls; `BroeckMetric.from_paper()` builds
the configuration of the 1999 paper:

```python
from warpdrive import BroeckMetric

paper = BroeckMetric.from_paper()
budget = paper.energy_budget_analytic()
print(budget.negative_mass, budget.positive_mass, paper.pocket_proper_radius())
```

## Layout

```
Warp_Drive/
├── README.md
├── ROADMAP.md
├── LICENSE
├── pyproject.toml
├── src/warpdrive/
│   ├── constants.py           # physical constants and unit conversions
│   ├── shapes.py              # radial profiles f(r), B(r) and derivatives
│   ├── integrators.py         # RK4 and adaptive RK45 over an ensemble
│   ├── tracers.py             # Eulerian congruence dragged by the bubble
│   ├── diagnostics.py         # travel times, energy budget, neck scan, quantum inequality, cone pocket
│   ├── symbolic.py            # Einstein tensor: source of truth
│   ├── geodesics.py           # null rays: the sky from the bubble
│   ├── waves.py               # scalar waves: the pocket as a cavity behind the throat
│   ├── acoustic.py            # phonon rays in a 2D condensate, checked on Viermann et al.
│   ├── metrics/
│   │   ├── base.py            # WarpMetric: the 3+1 interface
│   │   ├── alcubierre.py      # the 1994 metric
│   │   └── broeck.py          # Van Den Broeck's two-scale bubble
│   └── viz/
│       ├── style.py           # shared palette
│       ├── figures.py         # static figures
│       ├── comparison.py      # Alcubierre against Van Den Broeck
│       ├── sky.py             # the view from the bridge
│       └── animation.py       # flyby animation
├── scripts/
│   ├── run_alcubierre.py      # command line driver
│   ├── run_broeck.py          # Van Den Broeck study
│   ├── run_sky.py             # null rays and the sky
│   └── make_readme_images.py  # rebuilds images/readme
├── tests/                     # pytest suite
├── images/
│   ├── readme/                # images used by this README, tracked
│   └── local/                 # figures written by the drivers, ignored
└── reports/                   # text reports written by the drivers, ignored
```

Everything listed is in place; open work is tracked in
[ROADMAP.md](ROADMAP.md).

Every metric is written in the ADM form

```math
ds^2 = -c^2 dt^2 + B(r_s)^2\left[\left(dx - \beta(r_s)\,dt\right)^2 + dy^2 + dz^2\right]
```

so a concrete spacetime is fixed by two radial profiles: the shift $\beta$,
which drags the coordinates, and the conformal factor $B$, which inflates
spatial volume. Alcubierre has $B = 1$ and $\beta = v_s f(r_s)$; Van Den Broeck
keeps that shift and adds a non-trivial $B$. Everything that can be derived
from the interface — proper time, the total energy budget, the horizon — lives
in `WarpMetric` and is written once, so the figures and the driver never need
to know which metric they were handed.

## Tests

```bash
pytest
```

Errors in general relativity rarely crash: a wrong sign or a missing factor of
$c$ produces plausible numbers. The suite pins the invariants that would catch
that — $f(0) = 1$, $d\tau/dt = 1$ at any $v_s$, $\varepsilon \leq 0$ everywhere,
$\varepsilon = 0$ on the axis, $\theta$ antisymmetric under $x \to -x$,
superluminal motion in the *exterior* rejected as spacelike,
$E \propto v_s^2 R^2 \sigma$, and the horizon solver agreeing with the analytic
condition $f = 1 - c/v_s$. The closed-form energy budget is also checked against
the generic quadrature in the base class, which is the test that keeps the
interface honest now that a second metric exists: for Van Den Broeck the two
agree on both signs of the budget.

For the 1999 paper the suite pins every number it prints, each against the
configuration that actually produces it, and the factor $2 \times 10^{33}$ by
which the printed parameters violate the quantum inequality. The curvature is
derived symbolically from the Christoffel symbols and checked against two
closed forms, and the quantum-inequality margin against its $\tilde\Delta^2$
scaling at fixed $\alpha$; the search over moving observers is checked on a
slab where the answer, $v^2 = 1/3$, is known, so a lost $\gamma$ would show.
The areal form of the energy is checked against the quadrature on three
pockets; the rounded cone must sit above the floor by exactly its closed-form
excess, which scales as $r_\mathrm{min}^2$, reach the requested curvature radius
at *both* corners, cost less than every polynomial with the same curvature
radius, and cross the quantum inequality at $35\,\ell_P$ with a margin going as
$r_c^2$.
The wave solver is checked against mpmath for the Riccati–Bessel functions, on
flat space ($a = 1$, $c = 0$, $\Gamma = 1$), on flux conservation
$`|I|^2 = |O|^2 + 1`$, on the exact zero-frequency s-wave $u = A$, which
pins the curvature term of the potential, on fourth-order convergence (a
first-order solver would mean the jump of $B''$ at the pocket edge is sampled
wrongly), on the Fabry–Pérot peak height against the single-pass transmission,
and on the closed-cavity average of 1 for open and tunnelling modes alike.
The phonon rays of `acoustic.py` are checked against the arcsine law of the
radial travel time, against a sympy derivation of the curvature of the
optical metric from its Christoffel symbols, and on the conservation of
frequency and angular momentum.

### Where the physics comes from

None of those invariants constrain an overall constant. A density off by a
factor of two would keep every sign, every symmetry and every scaling law, and
the suite would stay green.

So the physics is not transcribed from papers. `symbolic.py` builds the
curvature of the ansatz from scratch with sympy — Christoffel symbols, Ricci
tensor, curvature scalar, Einstein tensor — keeping the full dependence on $w$,
and contracts it on the Eulerian normal $n^\mu = (1, b, 0, 0)$. That derivation
is the source of truth; the expressions in `metrics/` are the fast numpy path,
written by reading it.

Holding the two together:

| check | result |
| --- | --- |
| derived $\varepsilon$ vs the published Alcubierre density | difference exactly `0` |
| derived $\theta$ vs the published expansion | difference exactly `0` |
| derived vs `AlcubierreMetric.energy_density`, on a 400x400 grid | agreement to `5e-16` |
| cost of one derivation | 1.3 s ($B = 1$), 8.8 s (general $B$) |
| cost of evaluating the grid | 3.3 ms derived, 4.2 ms hand-written |

The numerical bridge feeds the package's own numpy shape functions into the
symbolically derived structure, so a mismatch can only come from the algebra of
one side or the other. It is the only test that pins the $c^4/8\pi G$ prefactor.

Both implementations are kept deliberately. The derivation is not the truth
either — it is a second program, with its own possible mistakes in the ansatz,
the analytic inverse or the contraction conventions. What counts as evidence is
that two independent routes agree, and deleting one would delete the evidence
rather than strengthen it.

## What the code computes

**Expansion scalar** — negative ahead of the ship (space contracting) and
positive behind it (space expanding), reproducing the surface from the original
paper:

```math
\theta = v_s \frac{x_s}{r_s} \frac{df}{dr_s}
```

![Expansion scalar](images/readme/alcubierre_expansion_scalar.png)

**Energy density** seen by Eulerian observers, with $\rho^2 = y^2 + z^2$:

```math
\varepsilon = -\frac{c^4}{8\pi G}\frac{v_s^2}{c^2}\frac{\rho^2}{4 r_s^2}\left(\frac{df}{dr_s}\right)^2
```

It is **negative everywhere it is non-zero** — the drive violates the weak
energy condition — and its distribution is a **torus** around the axis of
motion. That torus is the physical reason concept ships are drawn with rings:
the rings mark where the exotic matter has to be held.

![Exotic matter distribution](images/readme/alcubierre_energy_density.png)
![Negative-energy shell and twin-ring hull](images/readme/alcubierre_shell_3d.png)

**Energy budget** — the angular integral is analytic,
$`\int (\rho^2/r^2)\,d\Omega = 8\pi/3`$, which collapses the budget to a single
radial quadrature:

```math
E = -\frac{c^2 v_s^2}{12 G}\int_0^\infty \left(\frac{df}{dr}\right)^2 r^2\,dr
```

The integral is taken over the distance from the wall, $s = r - R$, rather
than over $r$. A wall of $10^2$ Planck lengths around a femtometre bubble has
$R\sigma \sim 10^{18}$, beyond the range of double precision: every $r$ across
it rounds to the same number, while $s$ stays representable. The budget is
always reported as a negative and a positive part, $E_-$ and $E_+$, never as a
single net number; for Alcubierre $E_+ = 0$, but for a metric with a conformal
factor the two have different physical meaning and summing them would hide the
exotic matter.

**Causal structure** — for $v_s > c$ a photon emitted forward along the axis
obeys $\dot{x}_s = c - v_s\left[1 - f\right]$, which vanishes where
$f = 1 - c/v_s$. A horizon forms inside the bubble wall: the crew cannot signal
the front of their own bubble, so it cannot be steered, slowed, or switched off
from the inside.

**Proper time** — evaluated from the line element rather than assumed. On the
ship worldline $f = 1$ and $\dot{x} = v_s$, so $d\tau = dt$ exactly: no time
dilation at any $v_s$.

## Van Den Broeck's two-scale bubble

`BroeckMetric` keeps Alcubierre's shift and adds a conformal factor $B(r_s)$
equal to $1 + \alpha$ inside a pocket of radius $\tilde R$ and to $1$ outside a
transition region of thickness $\tilde\Delta$. A coordinate radius $\tilde R$
then encloses a proper radius $(1 + \alpha)\tilde R$: with $\alpha = 10^{17}$ and
$\tilde R = 10^{-15}$ m, a 200 m pocket sits behind a shift wall of radius
$3 \times 10^{-15}$ m. Only the separated configuration is implemented, with the
transition inside the flat interior of $f$.

Running the derivation with $B$ abstract gives, in that configuration,

```math
\varepsilon = \frac{c^4}{8\pi G}\left[\frac{B'^2}{B^4} - \frac{2B''}{B^3} - \frac{4B'}{r_s B^3}\right] - \frac{c^2 v_s^2}{32\pi G}\,\frac{\rho^2}{r_s^2}\,f'^2
```

and the expansion is Alcubierre's. The first term does not depend on $v_s$:
inflating volume costs energy even at rest. It changes sign inside the
transition region, but with $\psi = \sqrt{B}$ an integration by parts shows
its total is strictly positive,

```math
E_B = \frac{c^4}{2G}\int_0^\infty \frac{B'^2}{B}\,r^2\,dr > 0
```

so the budget is reported as a negative and a positive part. For the paper's
configuration ($n = 80$, $v_s = c$, a shift wall of $10^2$ Planck lengths):

| part | computed | published |
| --- | --- | --- |
| transition region, $E_-$ | $-1.38 \times 10^{30} \~ \text{kg}$ | $-1.4 \times 10^{30} \~ \text{kg}$ |
| transition region, $E_+$ | $+4.87 \times 10^{30} \~ \text{kg}$ | $+4.9 \times 10^{30} \~ \text{kg}$ |
| shift wall, $E_-$ | $-2.08 \times 10^{29} \~ \text{kg}$ | $-6.3 \times 10^{29} \~ \text{kg}$ |
| total | $-0.80 \~ M_\odot$ and $+2.45 \~ M_\odot$ | |

The shift wall does not match because the paper uses a different profile $f$;
only its order of magnitude is comparable. For scale: an Alcubierre bubble of
100 m radius with the same Planck-thin wall needs about $-6 \times 10^{62}$ kg
(Pfenning & Ford), ten orders of magnitude above the mass of the visible
universe. The two-scale bubble brings that below a solar mass of negative
energy, but more than twice as much positive energy comes with it, and every
obstruction listed below still applies.

The peak density of eq. (14) and the curvature radius of eq. (20) are *not*
reproduced with these parameters: the paper computes them with a different
configuration, see the checks of the 1999 paper below.

**The pocket is a throat.** Inside the shift wall, in the frame of the bubble,
$f = 1$ and the metric is ultrastatic,

```math
ds^2 = -c^2 dt^2 + B(r)^2\left(dr^2 + r^2 d\Omega^2\right)
```

so its stress-energy depends on the geometry of space alone, not on $v_s$.
With the proper radial distance $`d\ell = B\,dr`$ and the areal radius
$`A = B\,r`$, the circumference of a sphere over $2\pi$, the null energy along
radial light rays is

```math
\varepsilon + p_r = -\frac{c^4}{8\pi G}\,\frac{2}{A}\,\frac{d^2 A}{d\ell^2}
```

the relation of a Morris–Thorne wormhole throat. In the flat pocket and outside
the transition $dA/d\ell = 1$; a pocket larger inside than outside,
$(1 + \alpha)\tilde R > \tilde R + \tilde\Delta$, forces $A$ to fall to a
minimum and rise again. That minimal sphere flares out, so by Hochberg and
Visser's theorem, whose definition of a throat is geometric and covers trivial
topology, the null energy condition fails there: **for any profile of $B$, at
any speed, for every observer**. The pocket needs exotic matter even at rest,
for a reason that does not depend on how the energy of a warp bubble is
defined. For the default pocket the throat has an areal radius of 11.2 m and
$`\varepsilon + p_r = -1.35\,c^4/8\pi G`$ per square metre there; for the 1999
configuration the throat is $1.46 \times 10^{-15}$ m across. Since
$dA/d\ell$ is 1 on both sides, $`\int (\varepsilon + p_r)\,A\,d\ell = 0`$: the
violation is balanced exactly by the region round the maximum of $A$, where the
condition holds. The throat form itself is not new: Krasnikov (2003) writes a
Van Den Broeck pocket in proper distance and finds the same
$`G_{\hat t\hat t} + G_{\hat r\hat r} = -2r''/r`$; what is added here is the
link to the flare-out theorem and its check against the full moving metric.
The derivation is done twice, in the moving Cartesian chart of
`symbolic.derive` and in the static spherical chart of `symbolic.derive_pocket`,
and the two agree to $10^{-9}$.

![The areal radius and the null energy across the pocket](images/readme/broeck_throat.png)

**A floor on the energy of the pocket.** Writing $\psi = \sqrt{B}$, the static
density is a Laplacian, $`\varepsilon = -(c^4/2\pi G)\,\nabla^2\psi/\psi^5`$, and
integrated over the proper volume $`\psi^6\,dV`$ by parts the net energy of the
transition region becomes a Dirichlet integral. Among all profiles with the same
end values it is smallest for the harmonic $\psi = C_1 + C_2/r$, and minimising
also over the inner radius at fixed proper pocket radius $P$ and outer radius
$b$ gives a floor for every profile:

```math
E_\mathrm{tot} = \frac{2c^4}{G}\int_a^b \psi'^2\,r^2\,dr \;\ge\; \frac{2c^4}{G}\,(P - b)
```

reached, as an infimum, at $a = b^2/P$ by a harmonic profile whose kinks are a
positive shell on the pocket and a negative one on the neck. The floor grows
with how much larger the pocket is inside than outside, the same excess that
forces the throat: 0.122 solar masses for the default pocket and 0.135 for the
1999 configuration, whose profile lies 13 times above it. A polynomial of order
10 with a well-chosen inner radius comes within a factor of 1.5.

This bounds $E_\mathrm{tot}$, the Eulerian energy integrated over a slice, which
is the quantity Van Den Broeck and Pfenning and Ford use and which Barzegar,
Buchert and Vigneron (2026) show to depend on the foliation and not to be a
mass. It says how far any profile could improve on the 1999 number; the
observer-independent statement is the null energy violation at the throat.
The floor is on the *net* energy. The negative part alone has no such floor:
Krasnikov (2003) shrinks it to about $-10^{-3}$ g with a profile of his own,
and for pockets of the conformally flat form used here the floor then says the
positive part must carry at least 0.13 solar masses more than that.

![Net energy of the transition region against its lower bounds](images/readme/broeck_energy_floor.png)

**The checks of the 1999 paper.** Van Den Broeck tests his transition region
against the quantum inequality of Ford and Roman, which bounds the density an
observer can measure over a sampling time $\tau_0$ short compared with the
smallest curvature radius $r_c$:

```math
\varepsilon \;\ge\; -\frac{3\hbar}{32\pi^2 c^3 \tau_0^4}, \qquad \tau_0 = \beta\,\frac{r_c}{c}, \quad \beta = 0.1
```

The metric of the pocket is ultrastatic, so its Riemann tensor is purely
spatial, with two independent orthonormal components,

```math
R_{\hat r\hat\theta\hat r\hat\theta} = -\frac{1}{A}\frac{d^2 A}{d\ell^2} = \frac{r B'^2 - r B B'' - B B'}{r B^4}, \qquad R_{\hat\theta\hat\phi\hat\theta\hat\phi} = \frac{1 - (dA/d\ell)^2}{A^2}
```

and $`r_c = 1/\sqrt{\max|R_{\hat a\hat b\hat c\hat d}|}`$ (densities in the table
are $\varepsilon/c^2$). The radial one, the
curvature of the throat, is the larger and equals $-p_t$ in units of
$`c^4/8\pi G`$. Every number the paper prints is reproduced, but only with **two
different configurations**, both with a 100 m pocket:

| | eq. (7): $\alpha = 10^{17}$, $\tilde R = \tilde\Delta = 10^{-15}$ m | the check: $\alpha = 10^{34}$, $\tilde R = \tilde\Delta = 10^{-32}$ m |
| :--- | :--- | :--- |
| net energies | $-1.38$, $+4.87 \times 10^{30}$ kg, the paper's eqs. (15), (16) | $-1.33$, $+4.82 \times 10^{30}$ kg |
| curvature radius | $\tilde\Delta/43.4$ | $\tilde\Delta/72.5$, the paper's eq. (20) |
| peak density | $-2.37 \times 10^{59}$ kg/m³ | $-6.63 \times 10^{93}$ kg/m³, the paper's eq. (22) |
| limit | $-1.18 \times 10^{26}$ kg/m³ | $-9.21 \times 10^{94}$ kg/m³, the paper's eq. (22) |
| quantum inequality | violated by a factor $2 \times 10^{33}$ | holds, margin 0.072 |

The giveaway is $\tilde\Delta/72.5 = 1.4 \times 10^{-34}$ m, which needs
$\tilde\Delta \approx 10^{-32}$ m. The arXiv history explains it: versions
1 to 4 of the paper (May and June 1999) use $\alpha = 10^{34}$ and
$\tilde R = \tilde\Delta = 10^{-32}$ m throughout, with energies of a few
milligrams; version 5, "error in calculation corrected", moves eq. (7) to the
femtometre configuration and recomputes the energies, but keeps the curvature
radius, "about ten Planck lengths", and the quantum-inequality numbers of the
earlier one. Krasnikov (2003) remarks in a footnote that the profile violates
the inequality; the size of the violation and its origin are, as far as we
found, not documented. The conclusions survive, because at fixed
proper pocket radius the energies barely depend on $\tilde\Delta$, as the floor
above predicts; but the parameters printed in eq. (7) violate the inequality by
33 orders of magnitude. It is an inconsistency at the level of an erratum, not a
refutation. At fixed pocket the margin grows roughly as $\tilde\Delta^2$: the
inequality holds only below $\tilde\Delta = 3.7 \times 10^{-32}$ m, and
$r_c$ reaches the Planck length at $1.2 \times 10^{-33}$ m, so the check
passes in a window of a decade and a half, with curvature radii of a few to a
few tens of Planck lengths, where the semiclassical inequality is itself
questionable.

The paper checks only observers at rest. One moving radially at speed $v$ sees
$`\gamma^2(\varepsilon + v^2 p_r)`$ and, with $r_c$ shortened by $\gamma$, a limit
$\gamma^4$ times larger, so the margin goes as
$`(\varepsilon + v^2 p_r)(1 - v^2)`$. Since
$`\varepsilon + v^2 p_r = (1 - v^2)\,\varepsilon + v^2(\varepsilon + p_r)`$, a moving
observer does worse only where the null contraction is more than twice as
negative as the density. It is not, for any polynomial profile tried: the
observers at rest are already the most restrictive.

![Quantum-inequality margin of the transition region against its thickness](images/readme/broeck_quantum_inequality.png)

**The cheapest pocket under a curvature bound.** Any static spherical pocket can
be written as $`dl^2 + A(l)^2 d\Omega^2`$, with $l$ the proper radial distance
and $A$ the areal radius, and then

```math
\varepsilon = \frac{c^4}{8\pi G}\,\frac{1 - A'^2 - 2AA''}{A^2}, \qquad E_\mathrm{tot} = \int \varepsilon\,4\pi A^2\,dl = \frac{c^4}{2G}\int \left(1 + A'^2\right) dl + \frac{c^4}{G}\,(P - b)
```

after one integration by parts, with $A' = 1$ in the flat pocket and outside.
Since $1 + A'^2 \ge 2|A'|$ and $A$ must fall from $P$ to its minimum and rise
back to $b$,

```math
E_\mathrm{tot} \;\ge\; \frac{2c^4}{G}\,(P - A_\mathrm{min}) \;\ge\; \frac{2c^4}{G}\,(P - b)
```

for every spherical pocket, not only the conformally flat ones of the previous
paragraph. The proof also shows what saturates it: a cone, $A' = -1$ from
$A = P$ down to $A = b$, where the density vanishes and all the energy sits at
the two corners. Rounding the corners so that the curvature radius never falls
below $r_\mathrm{min}$ costs, in closed form,

```math
E_\mathrm{tot} = \frac{2c^4}{G}\,(P - b) + \frac{2c^4}{3G}\,(L_1 + L_2), \qquad L_1 = \frac{2r_\mathrm{min}^2}{P}, \quad L_2 \simeq \frac{2r_\mathrm{min}^2}{b}
```

a relative excess of $10^{-55}$ for the 1999 pocket at ten Planck lengths. The
bound on the curvature radius Van Den Broeck used to choose his profile does not
force the factor of 13 his profile pays; the shape does. Krasnikov (2003) already
builds a pocket of this kind, with a parabolic corner a Planck length thick, to
minimise the *negative* energy; what is added here is the identity for the net
energy, the floor for every spherical pocket, and the closed-form cost of the
curvature bound.

At the outer corner the density is set by the curvature,
$`\varepsilon \simeq -c^4/4\pi G r_c^2`$, while the quantum-inequality limit
grows as $1/r_c^4$. Their ratio is $`(8\pi/3)\,\beta^4 (r_c/\ell_P)^2`$, so the
inequality caps the curvature radius instead of bounding it from below:

```math
r_c \;\le\; \sqrt{\frac{3}{8\pi}}\;\frac{\ell_P}{\beta^2} \approx 35\,\ell_P \qquad (\beta = 0.1)
```

This is why the check of the 1999 paper passes only in a narrow window near the
Planck length. It is the same reasoning by which Pfenning and Ford bound the
shift wall to about $10^2\,\ell_P$, applied to the pocket.

Finally, the paper says $n = 80$ is the order that maximises $r_c$. It is, to
0.1%, for $\alpha = 10^{34}$, the configuration of its first four arXiv
versions (the exact optimum is $n = 84$); for the $\alpha = 10^{17}$ of the
published eq. (7) the best order is $n = 44$, one more trace of the same
revision.

![Net energy against the curvature bound, and the best order of the polynomial](images/readme/broeck_curvature_bound.png)

**The pocket is a cavity, not a shield.** Above $c$ the rear wall of the bubble
is a horizon, and Finazzi, Liberati and Barceló (2009) find a thermal flux at
the centre of the bubble at

```math
T_H = \frac{\hbar c\,\kappa}{2\pi k_B}, \qquad \kappa = \frac{v_s}{c}\,\left|f'(h)\right|
```

a calculation done in 1+1 dimensions, which they expect to hold near the axis
in 3+1. Taking that flux as given, stationary and arriving at the neck from
outside, does the throat keep it out of the pocket? For a massless scalar
$`\Phi = e^{-i\omega t}\,Y_{\ell m}\,u(l)/A`$ in the ultrastatic metric of the
pocket, the radial equation is a one-dimensional scattering problem,

```math
u'' + \left[k^2 - V_\ell(l)\right] u = 0, \qquad V_\ell = \frac{\ell(\ell + 1)}{A^2} + \frac{A''}{A}, \qquad k = \frac{\omega}{c}
```

with primes in proper distance. The second term is the curvature of the throat;
the first peaks there too, so only $`\ell \lt k A_\mathrm{min}`$ crosses it
classically (left panel below). `waves.PocketWaves` carries Riccati–Bessel
solutions across the transition region by RK4 and reads off, for each $\ell$,
the single-pass transmission $\Gamma_\ell(k)$ and the interior intensity
$1/N_\ell$ per unit incident intensity.

But the pocket has no exit on the far side. In a steady state what enters must
leave, and the interior intensity of each partial wave is a Fabry–Pérot comb
whose average over one resonance is exactly

```math
\left\langle \frac{T}{\left|1 - \sqrt{1 - T}\,e^{i\varphi}\right|^2} \right\rangle_\varphi = \frac{T}{1 - (1 - T)} = 1
```

for any transmission $T$: an opaque throat makes the peaks higher, as
$4/\Gamma_\ell$, and narrower, as $\Gamma_\ell$, but does not lower the average
(centre panel). **In a steady state the pocket does not shield anywhere.** What
the throat controls is time: each round trip $t_\mathrm{rt}$ lets in a fraction
$\Gamma_\ell$ of what is missing, so mode $\ell$ fills in

```math
\tau_\ell = \frac{t_\mathrm{rt}}{-\ln(1 - \Gamma_\ell)} \;\approx\; \frac{t_\mathrm{rt}}{\Gamma_\ell}
```

At the centre only $\ell = 0$ reaches, it crosses the throat, and the crew is
reached at once: the view from the centre of section 2 already showed every
direction there opens onto the outside sky. Off centre, at proper distance
$\rho$, the modes with $\ell \lt k A_\mathrm{min}$ fill within a few round trips
and give the lit fraction rays predict, $`1 - \sqrt{1 - (A_\mathrm{min}/\rho)^2}`$;
the rest tunnel in on times that grow exponentially with $\ell$ (right panel).
For the default pocket ($\Xi = \kappa A_\mathrm{min}/2\pi \approx 3$ at $10c$)
this is visible. For a bubble that satisfies the quantum inequalities,
$\Xi \sim 10^{17}$, the ray result holds for any realistic time, and the wave
calculation adds nothing to it.

![Transmission of the throat, the cavity comb at the centre, and the filling off centre](images/readme/broeck_cavity.png)

**Towards an acoustic pocket.** In a Bose–Einstein condensate the wave regime
$\Xi \sim 1$ is natural, so the comb of the cavity is the laboratory signature
to look for. Every ingredient exists separately: a 1D acoustic warp drive
(Finazzi 2012), engineered spatial curvature in 2D condensates (Viermann et al.
2022), acoustic throats (Vaidya and Kruczenski 2024), cavity resonances between
horizons (black-hole lasers), and measured Hawking radiation (Steinhauer and
collaborators); an acoustic pocket behind a throat was not found.

A first estimate says it is hard. A pocket of size $L$ holds about
$`N \simeq L/\pi\xi_\mathrm{in}`$ resonances below the Bogoliubov cutoff, with
$`\xi_\mathrm{in} = \hbar/m c_\mathrm{in}`$ the healing length inside, and a time
$T$ allows $`n_\mathrm{rt} = T c_\mathrm{in}/2L`$ round trips. Their product does
not depend on the size of the pocket,

```math
N\,n_\mathrm{rt} \simeq \frac{\mu_\mathrm{in}\,T}{h}, \qquad \mu_\mathrm{in} = m\,c_\mathrm{in}^2
```

and a pocket of slow sound, $B = c_\mathrm{out}/c_\mathrm{in}$, lowers it as
$1/B^2$. A clear comb, five resonances over five round trips, needs about 25.
The rubidium black hole of Muñoz de Nova et al. (2019), $c = 0.52$ mm/s and
about 0.1 s of stationary flow, and the potassium disc of Viermann et al.,
$c = 1.2$ mm/s over tens of milliseconds, both give about 7 without a pocket
and 1.8 with $B = 2$. The comb needs condensates denser or longer-lived by an
order of magnitude.

The ray tools for the acoustic pocket are first checked on the published
geometry: `acoustic.ThomasFermiDisc` reproduces the harmonically trapped disc
of Viermann et al., whose optical metric $`(dr^2 + r^2 d\phi^2)/c_s^2`$ has
curvature $`K = -2c_0^2/(R^2 - r^2)`$, negative and nearly constant near the
centre, and in which a phonon aimed at the centre from 20 µm arrives after
$`(R/c_0)\arcsin(0.8) = 19.3`$ ms, the time scale of their wave packets.

With the same shift wall, the pocket adds a thin shell carrying both signs of
energy and holds more space than its coordinate size: at the default parameters
10 m of coordinate radius hold 110 m of proper radius.

![Alcubierre and Van Den Broeck on identical axes](images/readme/broeck_comparison.png)

Where the thirty orders of magnitude come from, and where they stop: holding the
pocket at 100 m of proper radius and the wall at $10^2$ Planck lengths, and
keeping the paper's proportions $\tilde R = \tilde\Delta = R/3$, the shift wall
costs $\propto R^2$ while the transition region depends only on
$(1 + \alpha)\tilde R$ and stays at $-0.69$ and $`+2.45\,M_\odot`$ for any neck
much smaller than the pocket. Shrinking the neck pays off until the wall drops
below the transition region, near the paper's $R = 3 \times 10^{-15}$ m; below
that the budget has a floor.

![Exotic mass against neck radius](images/readme/broeck_neck_scaling.png)

## The view from the bridge

For constant $v_s$ the metric is static in the rest frame of the bubble,
$\xi = x - v_s t$:

```math
ds^2 = -c^2 dt^2 + B^2\left[\left(d\xi - \tilde\beta\,dt\right)^2 + dy^2 + dz^2\right], \qquad \tilde\beta = -v_s\,(1 - f)
```

so a photon has a conserved Hamiltonian, and `geodesics.py` integrates
Hamilton's equations backwards from the ship:

```math
H = \frac{c}{B}\,|p| + \tilde\beta\,p_\xi, \qquad \dot x^i = \frac{\partial H}{\partial p_i}, \qquad \dot p_i = -\frac{\partial H}{\partial x^i}
```

An Eulerian observer measures a photon energy proportional to $|p|/B$. At the
ship $\tilde\beta = 0$, far away $B = 1$ and $\tilde\beta = -v_s$, so the
conservation of $H$ fixes the blueshift of every source without integrating
anything:

```math
\frac{E_\mathrm{ship}}{E_\mathrm{far}} = 1 - \frac{v_s}{c}\,n_\xi
```

with $n$ the direction in which the light travels far from the bubble. Three
consequences, each checked against the ray integrator:

- a star straight ahead appears $1 + v_s/c$ times bluer, eleven times at $10c$;
- no light with $n_\xi > c/v_s$ reaches the ship. Behind a superluminal bubble
  the sky is **dark**, because the bubble outruns the light chasing it: only
  sources within $\arccos(-c/v_s)$ of the direction of travel can be seen, and
  the hidden fraction of the sky is $(1 - c/v_s)/2$, 45 % at $10c$;
- a ray traced towards the rear stalls at the horizon, exactly where
  `horizon_offset` puts it, and its momentum grows as $e^{\kappa c t}$ with
  $`\kappa = (v_s/c)\,|f'(h)|`$, the analogue of a surface gravity.

From the centre the sky is symmetric about the direction of travel, so a fan of
rays in one plane describes all of it. The views below are fisheye images of a
procedural star field: straight ahead at the centre of each disc, straight
behind at the rim, stars coloured by $E_\mathrm{ship}/E_\mathrm{far}$ and
sized by the flux they deliver.

![The sky seen from the centre of the bubble](images/readme/sky_fisheye.png)

Apart from the visible limit, the stars barely move: the apparent angle stays
close to the true one until the last few degrees, where the sky is stretched
over the rest of the view and redshifted to nothing.

**Brightness.** $I_\nu/\nu^3$ is conserved along a ray, so the surface
brightness of the sky scales as the fourth power of the frequency ratio
$R = E_\mathrm{ship}/E_\mathrm{far}$. A star is a point, so its flux also
carries the magnification $\mu$, the ratio of the solid angles it covers with
and without the bubble:

```math
\frac{F_\mathrm{ship}}{F_\mathrm{far}} = R^4\,\mu, \qquad \mu = \frac{\sin\theta_\mathrm{look}}{\sin\theta_\mathrm{source}}\,\frac{d\theta_\mathrm{look}}{d\theta_\mathrm{source}}
```

A star straight ahead arrives $`(1 + v_s/c)^4\,\mu`$ times brighter:
$1.34 \times 10^4$ times at $10c$, where $11^4 = 14641$ is reduced by
$\mu = 0.91$. Integrated over the view, an isotropic background of starlight
reaches the crew

```math
\frac{1}{4\pi}\int R^4\,d\Omega_\mathrm{look} = \frac{1}{4\pi}\int R^4\,\mu\,d\Omega_\mathrm{source}
```

times brighter than at rest: 1.51 at $0.5c$, 12.1 at $2c$ and **1507 at
$10c$**, most of it shifted into the ultraviolet. If the bubble shifted
frequencies but left the stars in place ($\mu = 1$) the integral would have the
closed form $[(1 + u)^5 - \max(0, 1 - u)^5]/(10u)$, with $u = v_s/c$; the
distortion of the sky takes 6 % off it at $10c$. Both integrals are computed
from the traced rays and agree to $10^{-6}$.

![Apparent angle, blueshift and flux against the true angle](images/readme/sky_mapping.png)

For a ship at the centre, Van Den Broeck's pocket changes nothing: $B$ is
spherically symmetric, so rays leaving the centre cross it radially and only
slow down, and the blueshift above does not contain $B$. With the same shift
wall the two skies agree to $10^{-12}$ rad.

**Away from the centre** the pocket takes over. Inside the shift wall the
spatial metric is $B^2\delta_{ij}$, so for light $B$ acts as a spherically
symmetric refractive index, and Bouguer's invariant holds along every ray:

```math
B(r)\,r\,\sin\psi = L
```

with $\psi$ the angle to the radial direction. $`B\,r`$ is the areal radius, the
circumference of a sphere over $2\pi$, and its minimum outside the pocket is a
**throat**: a ray leaves the pocket only if $L$ is below the throat radius.
From a proper distance $\ell_0$ from the centre the outside is therefore seen
only through two windows about the radial line, one outwards and one through
the centre, of half-angle

```math
\sin\psi_c = \frac{R_\mathrm{throat}}{\ell_0}
```

and every other line of sight stays inside the pocket. For the default pocket,
110 m across with an 11.2 m throat, the windows are 21.9° at 30 m from the
centre and 7.2° at 90 m, and the whole sky is squeezed into them.
`trace_rays_3d` traces the rays in three dimensions from any point inside the
bubble; the edge of each window falls on the closed-form cone.

![The sky from four points of the pocket](images/readme/sky_offcentre.png)

For the configuration of the 1999 paper the throat is $1.46 \times 10^{-15}$ m.
One metre from the centre of the 100 m pocket, the crew would see the entire
universe through a window of $10^{-15}$ rad, and the inside of the pocket in
every other direction.

## Sample output

For $R = 100$ m, $1/\sigma = 10$ m, $v_s = 10c$:

```
Proxima Centauri, 4.2465 ly
  coordinate time  t   = 0.4246 yr
  crew proper time tau = 0.4246 yr        (dtau/dt = 1.000000)
  1g relativistic rocket, same trip: tau = 3.54 yr, t = 5.87 yr

  negative energy  E-  = -3.37e+47 J       (positive part E+ = 0)
  exotic mass      M-  = -3.75e+30 kg  =  -1.89 solar masses
  future horizon at 89.0 m ahead of the ship
```

Scaling of the required exotic mass, in solar masses:

| $v_s/c$ | $R = 10 \~ \text{m}$ | $R = 100 \~ \text{m}$ | $R = 1000 \~ \text{m}$ |
| --- | --- | --- | --- |
| $0.5$ | $-9.7 \times 10^{-5}$ | $-4.7 \times 10^{-3}$ | $-4.7 \times 10^{-1}$ |
| $1$ | $-3.9 \times 10^{-4}$ | $-1.9 \times 10^{-2}$ | $-1.9$ |
| $2$ | $-1.6 \times 10^{-3}$ | $-7.6 \times 10^{-2}$ | $-7.5$ |
| $10$ | $-3.9 \times 10^{-2}$ | $-1.9$ | $-1.9 \times 10^{2}$ |
| $100$ | $-3.9$ | $-1.9 \times 10^{2}$ | $-1.9 \times 10^{4}$ |

$E \propto v_s^2 R^2 \sigma$. Alcubierre's own thin-wall estimate gave a
*negative* mass larger than the whole visible universe; the numbers above are
milder only because the wall here is 10 m thick rather than sub-nuclear.

## The honest part

The simulation is an exact solution of Einstein's equations — but that is a
weaker statement than it sounds. General relativity lets you write down *any*
metric and then read off, from
$G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$, the stress-energy needed to hold
it up. Alcubierre ran the equations backwards: he chose the geometry he wanted
and computed the matter it demands. The matter it demands does not exist.

Known obstructions, in rough order of severity:

1. **Exotic matter.** The required $T_{\mu\nu}$ violates the weak, null, and
   dominant energy conditions. Casimir-type effects give tiny, static,
   laboratory-scale negative energy densities — nothing remotely like a
   macroscopic shell.
2. **Quantum inequalities.** Pfenning & Ford (1997) showed the wall must be
   thinner than $\sim 10^2$ Planck lengths, which drives the energy requirement
   back up to absurd values (the mission report prints the wall thickness in
   Planck lengths so you can see how far off it is). Van Den Broeck's pocket
   passes the same test only with curvature radii of a few tens of Planck
   lengths, and not with the parameters it prints (see above).
3. **The horizon.** Computed above: the bubble cannot be controlled from
   within, so it must be laid down in advance along the entire route — which
   requires something already at the destination.
4. **Causality.** Two superluminal bubbles on different trajectories can be
   combined into a closed timelike curve.
5. **Semiclassical instability.** The wall behaves like a horizon and radiates;
   Finazzi, Liberati & Barceló (2009) found the interior temperature diverges,
   destabilising the bubble.
6. **The bulldozer problem.** The animation shows it directly: particles near
   the axis are captured, since $f = 1$ forces them to $\dot{x} = v_s$. The
   bubble sweeps up interstellar matter and releases it, extremely blueshifted,
   at the destination (McMonigal, Lewis & O'Byrne 2012). Light does the same
   to the crew: at $10c$ the ship receives about 1500 times the ambient
   starlight, blueshifted by up to a factor of 11.

Modern work moves toward *subluminal* solitons with positive energy —
Bobrick & Martire (2021), Lentz (2021), Fell & Heisenberg (2021),
Schuster et al. (2023) — which are physically far more respectable but no
longer faster than light.

## Roadmap

The next step, an acoustic analogue of the pocket in a Bose–Einstein condensate,
where the wave regime of the cavity is natural, lives in
**[ROADMAP.md](ROADMAP.md)**.

## References

- M. Alcubierre, *The warp drive: hyper-fast travel within general relativity*,
  Class. Quantum Grav. **11**, L73 (1994)
- M. J. Pfenning & L. H. Ford, Class. Quantum Grav. **14**, 1743 (1997)
- C. Van Den Broeck, Class. Quantum Grav. **16**, 3973 (1999)
- L. H. Ford & T. A. Roman, *Quantum field theory constrains traversable
  wormhole geometries*, Phys. Rev. D **53**, 5496 (1996)
- S. Krasnikov, *Quantum inequalities do not forbid spacetime shortcuts*,
  Phys. Rev. D **67**, 104013 (2003)
- S. Finazzi, S. Liberati & C. Barceló, Phys. Rev. D **79**, 124017 (2009)
- W. A. Hiscock, *Quantum effects in the Alcubierre warp drive spacetime*,
  Class. Quantum Grav. **14**, L183 (1997)
- S. Finazzi, *Analogue gravitational phenomena in Bose–Einstein condensates*,
  PhD thesis, SISSA (2012), arXiv:1208.4729
- J. R. Muñoz de Nova, K. Golubkov, V. I. Kolobov & J. Steinhauer,
  *Observation of thermal Hawking radiation and its temperature in an analogue
  black hole*, Nature **569**, 688 (2019)
- C. Viermann et al., *Quantum field simulator for dynamics in curved
  spacetime*, Nature **611**, 260 (2022)
- S. Vaidya & M. Kruczenski, *Acoustic black holes, white holes, and wormholes
  in Bose–Einstein condensates in two dimensions*, arXiv:2412.02727 (2024)
- R. F. Rosato, S. Biswas & S. Chakraborty, *Greybody factors, reflectionless
  scattering modes, and echoes of ultracompact horizonless objects*,
  arXiv:2501.16433 (2025)
- B. McMonigal, G. F. Lewis & P. O'Byrne, Phys. Rev. D **85**, 064024 (2012)
- A. Bobrick & G. Martire, Class. Quantum Grav. **38**, 105009 (2021)
