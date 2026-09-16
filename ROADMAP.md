# Roadmap

Status of the `warpdrive` package. Checked items are implemented and covered by
the test suite; unchecked ones are planned.

## Done

- [x] **Alcubierre metric** — shape function $f(r_s)$, shift $\beta = v_s f$,
      expansion scalar, Eulerian energy density.
- [x] **`WarpMetric` interface** — the 3+1 form
      $ds^2 = -c^2 dt^2 + B^2[(dx - \beta\,dt)^2 + dy^2 + dz^2]$, so a metric is
      fixed by two radial profiles and a second spacetime can be added without
      touching the driver or the figures.
- [x] **Energy budget** — generic quadrature carrying $\sqrt{\gamma} = B^3$ in
      the base class, plus the Alcubierre closed form
      $E = -\dfrac{c^2 v_s^2}{12G}\displaystyle\int_0^\infty \left(\frac{df}{dr}\right)^2 r^2\,dr$.
- [x] **Causal structure** — horizon solver for $\beta + c/B = v_s$, reducing to
      $f = 1 - c/v_s$ for Alcubierre.
- [x] **Proper time** — evaluated from the line element rather than assumed.
- [x] **Eulerian tracers** — RK4 congruence showing drag, release, and on-axis
      capture (the bulldozer problem).
- [x] **Visualisation** — shape function, expansion surface, exotic-matter
      torus, 3D shell with an IXS-style hull, and the flyby animation.
- [x] **Symbolic derivation** — `symbolic.py` builds the Einstein tensor of the
      ansatz with sympy and reproduces the published energy density and
      expansion exactly, prefactor included. It is now the source of truth for
      the physics: the expressions in `metrics/` are the fast numpy path,
      written by reading the derivation, and a test holds the two together.
- [x] **Packaging and tests** — `src` layout, CLI drivers, and a test suite
      pinning the invariants a sign error would not crash on.

## 1. Van Den Broeck's two-scale bubble

- [x] $B(r_s)$ volume profile in `shapes.py`: the polynomial of the 1999 paper,
      $B = 1 + \alpha\left[n w^{n-1} - (n-1) w^n\right]$ with
      $w = (\tilde R + \tilde\Delta - r_s)/\tilde\Delta$
- [x] `BroeckMetric` overriding `conformal_factor`, written from the derivation
      rather than transcribed from the 1999 paper, with the transition region of
      $B$ kept inside the flat interior of $f$
- [x] Energy budget split into its negative and positive parts, on a radial
      grid that resolves each wall
- [x] Closed-form check of the net budget of the $B$ region (see below)
- [x] Cross-check against the published numbers, as an independent third
      opinion
- [x] Energy scaling plot: exotic mass vs neck radius, at fixed pocket and
      wall (`neck_scaling`, `plot_neck_scaling`)
- [x] Side-by-side comparison with Alcubierre on identical axes
      (`plot_metric_comparison`, `scripts/run_broeck.py`)

Van Den Broeck (1999) noticed that the energy requirement scales with the
*surface* of the bubble, not with the volume it encloses. Adding a second shape
function $B(r_s)$ that inflates the spatial volume inside a microscopic neck
produces a pocket with a large interior volume — metres across, enough for the
ship — hidden behind an outer surface of nuclear size. Since
$E \propto v_s^2 R^2 \sigma$ and $R$ is now the radius of the *neck*, the
negative energy of the shift wall collapses.

The headline figure needs its context. Van Den Broeck compares against an
Alcubierre bubble whose wall is only $\sim 10^2$ Planck lengths thick, as the
quantum inequalities demand, and whose negative energy is far larger than the
mass of the visible universe. His geometry brings that down to **a few solar
masses of negative energy, accompanied by a comparable amount of positive
energy** stored in the transition region of $B$. The reduction is relative to
that Planck-thin wall, not to the 10 m wall used in this package, which already
gives $-1.9\,M_\odot$ at $v_s = 10c$; applying the same factor to our numbers
would be meaningless. The budget for this package's parameters will come out of
the computation, not out of the paper.

This is also the first real test of the `WarpMetric` abstraction, which is why
it comes before the ray tracer: it is the cheapest way to find out whether the
interface was designed correctly, before anything larger is built on top of it.

Implementation notes:

- the energy budget is an `EnergyBudget` with separate negative and positive
  parts. `WarpMetric.energy_budget` integrates each radial region declared by
  `energy_regions` on its own grid and refuses regions too thin for double
  precision; `AlcubierreMetric.energy_budget_analytic` integrates over the
  wall offset $s = r - R$, which reaches Planck-thin walls and matches the
  thin-wall limit. `BroeckMetric` adds the transition region of $B$ to
  `energy_regions`. The radial quadratures use the midpoint rule, because
  $B''$ jumps at the inner edge of the transition region and a trapezoid rule
  converges only as $1/n$ there;
- the energy density is no longer sign-definite: the transition region of $B$
  carries $\varepsilon > 0$, so the $\varepsilon \leq 0$ invariant of the
  Alcubierre tests must not be reused for `BroeckMetric`;
- with the two regions separated, `symbolic.py` gives
  $$\varepsilon = \frac{c^4}{8\pi G}\left[\frac{B'^2}{B^4} - \frac{2B''}{B^3} - \frac{4B'}{r_s B^3}\right] - \frac{c^2 v_s^2}{32\pi G}\,\frac{\rho^2}{r_s^2}\,f'^2$$
  i.e. a static term from the curvature of $\gamma_{ij} = B^2\delta_{ij}$ plus
  the Alcubierre term, and the expansion is the Alcubierre one. The static term
  matches eq. (11) of the paper. It changes sign pointwise, but writing
  $\psi = \sqrt{B}$ and integrating by parts gives a strictly positive net
  budget,
  $$E_B = \frac{c^4}{2G}\int_0^\infty \frac{B'^2}{B}\,r^2\,dr$$
  which is the closed-form check above;
- the published numbers to reproduce, for $n = 80$, $\alpha = 10^{17}$,
  $\tilde R = \tilde\Delta = 10^{-15}$ m: $E_{II,-} = -1.4 \times 10^{30}$ kg,
  $E_{II,+} = 4.9 \times 10^{30}$ kg, sign change at $w = 0.981$. A
  `BroeckMetric.from_paper()` reproduces all three ($-1.38$, $4.87$,
  $0.981$) and the closed form gives the same net $3.49 \times 10^{30}$ kg,
  all pinned by tests. The same quadrature does *not* reproduce the peak
  density of eq. (14) or the curvature radius of eq. (20), which it finds near
  $w \approx 0.575$ rather than $0.349$; at $w = 0.349$ the printed profile has
  $B - 1 \sim 10^{-18}$. Those two numbers are not used as checks until the
  discrepancy is understood;
- `horizon_offset` already solves $\beta + c/B = v_s$, so the causal structure
  needs no changes either;
- the figures take a metric, but three of them silently assumed Alcubierre:
  a one-sided colour scale and a 'negative everywhere' title for the density,
  and a 3D shell sampled by $|\varepsilon|$. They now use a symmetric log scale,
  a title read from the data, and the negative part only;
- the honest counterpart: the quantum inequalities of Pfenning & Ford still
  apply to the neck, the interior volume has to be seeded somehow, and the
  causal and stability problems are untouched. The variant makes the drive
  cheap, not physical.

Reference: C. Van Den Broeck, *A "warp drive" with more reasonable total energy
requirements*, Class. Quantum Grav. **16**, 3973 (1999).

### 1b. Overlapping regions (later)

- [ ] `BroeckMetric` variant with the transition region of $B$ overlapping the
      wall of $f$
- [ ] Coupling terms of the energy density and the modified expansion, from
      the same derivation
- [ ] Budget comparison against the separated configuration

When the two regions overlap, the derivation adds

$$\varepsilon_{\mathrm{coupling}} = \frac{c^2 v_s^2}{8\pi G}\,\frac{x_s^2}{r_s^2}\left[3\left(\frac{B'}{B}\right)^2(1-f)^2 - 2\,\frac{B'}{B}\,f'\,(1-f)\right], \qquad \theta = v_s\,\frac{x_s}{r_s}\left[f' - 3\,\frac{B'}{B}\,(1-f)\right]$$

Both vanish identically when $B' = 0$ wherever $f \neq 1$, which is why the
separated case comes first: it can be checked term by term against the paper.
The overlapping case has no published reference to compare with, so it is the
one where the derivation carries the whole weight, and it exercises terms of
`symbolic.py` that nothing else in the suite reaches.

## 2. Null-geodesic ray tracing — the view from the bridge

- [ ] Hamiltonian ray integrator for the ADM form
- [ ] Adaptive RK45 in `integrators.py`
- [ ] Backwards integration from the observer, one ray per pixel
- [ ] Blueshift map at the front wall
- [ ] Horizon shadow as a correctness check
- [ ] Rendered star field at several $v_s$, for both metrics

Integrate null geodesics backwards from the ship to reconstruct what the crew
would actually see through a window.

The metric is not static, so the geodesics have to be integrated in the full 4D
spacetime rather than reduced to an effective potential.

A sign convention has to be fixed first. In this package $\beta = v_s f(r_s)$
is the *drag velocity*, entering the line element as $(dx - \beta\,dt)$. The
standard ADM shift enters as $(dx^i + \beta^i_{\mathrm{ADM}}\,dt)$, so
$\beta^x_{\mathrm{ADM}} = -\beta$. With lapse $\alpha = c$ and spatial metric
$\gamma_{ij} = B^2 \delta_{ij}$, the general photon Hamiltonian
$H = \alpha\sqrt{\gamma^{ij} p_i p_j} - \beta^i_{\mathrm{ADM}}\, p_i$ becomes

$$H = \frac{c}{B}\sqrt{\delta^{ij} p_i p_j} + \beta\, p_x$$

so the ray equations are $\dot{x}^i = \partial H / \partial p_i$ and
$\dot{p}_i = -\partial H / \partial x^i$, with the shift and $B$ supplying all
the coupling. A photon sent forward along the axis gets
$\dot{x} = \beta + c/B$, the same condition `horizon_offset` solves; at the
centre of a $v_s = 10c$ bubble that is $11c$, i.e. $c$ relative to the ship, as
it must be in a flat region. Writing the shift term with the opposite sign
would give $-9c$ there and move the horizon shadow. Practical notes:

- integrate **backwards** in time from the observer, one ray per pixel, and map
  the escaping direction onto a background star field or an equirectangular sky
  texture;
- `integrators.py` already vectorises RK4 over an ensemble, so a batch of rays
  costs about the same as one; the wall, however, needs adaptive stepping —
  $f'$ is sharply peaked and a fixed step will walk straight through it, so an
  embedded RK45 belongs in the same module;
- carry the photon frequency along each ray to get the **blueshift map** at the
  front wall, which is the observable that makes the bulldozer problem
  quantitative rather than anecdotal;
- for $v_s > c$ the horizon computed by `horizon_offset` shows up as a region no
  backwards ray can reach — a black disc ahead of the ship. That is a useful
  check on the integrator: the disc must appear exactly where the bisection puts
  it.

By this point there are two metrics to render, so the same camera can show what
changes when the bubble is Van Den Broeck's rather than Alcubierre's.
