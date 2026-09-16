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
- [x] **Packaging and tests** — `src` layout, CLI driver, 36 tests pinning the
      invariants a sign error would not crash on.

## 1. Van Den Broeck's two-scale bubble

- [ ] $B(r_s)$ volume profile in `shapes.py`
- [ ] `BroeckMetric` overriding `conformal_factor`, written from the derivation
      rather than transcribed from the 1999 paper
- [ ] Cross-check the result against the published expression, as an
      independent third opinion
- [ ] Expansion and energy density from `symbolic.py`, run with
      `conformal=True`
- [ ] Energy scaling plot: exotic mass vs neck radius
- [ ] Side-by-side comparison with Alcubierre on identical axes

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

- the generic quadrature in `WarpMetric.total_exotic_energy` already carries
  the $\sqrt{\gamma} = B^3$ factor, so the budget needs no new integrator, and
  the test comparing it against the Alcubierre closed form guards the shared
  path;
- the energy density is no longer sign-definite: the transition region of $B$
  carries $\varepsilon > 0$, so the $\varepsilon \leq 0$ invariant of the
  Alcubierre tests must not be reused for `BroeckMetric`, and the budget has to
  be reported as separate negative and positive parts rather than a single net
  number;
- `horizon_offset` already solves $\beta + c/B = v_s$, so the causal structure
  needs no changes either;
- the figures and the driver take a metric, so the two spacetimes can be
  compared side by side on identical axes with no changes to `viz/`;
- the honest counterpart: the quantum inequalities of Pfenning & Ford still
  apply to the neck, the interior volume has to be seeded somehow, and the
  causal and stability problems are untouched. The variant makes the drive
  cheap, not physical.

Reference: C. Van Den Broeck, *A "warp drive" with more reasonable total energy
requirements*, Class. Quantum Grav. **16**, 3973 (1999).

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
