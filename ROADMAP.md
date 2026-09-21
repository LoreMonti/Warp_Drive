# Roadmap

Status of the `warpdrive` package. Checked items are implemented and covered by
the test suite; unchecked ones are planned.

## Done

- [x] **Alcubierre metric** — shape function $f(r_s)$, shift $\beta = v_s f$,
      expansion scalar, Eulerian energy density.
- [x] **`WarpMetric` interface** — the 3+1 form
      $`ds^2 = -c^2 dt^2 + B^2[(dx - \beta\,dt)^2 + dy^2 + dz^2]`$, so a metric is
      fixed by two radial profiles and a second spacetime can be added without
      touching the driver or the figures.
- [x] **Energy budget** — generic quadrature carrying $\sqrt{\gamma} = B^3$ in
      the base class, plus the Alcubierre closed form
      $`E = -\dfrac{c^2 v_s^2}{12G}\displaystyle\int_0^\infty \left(\frac{df}{dr}\right)^2 r^2\,dr`$.
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
gives $`-1.9\,M_\odot`$ at $v_s = 10c$; applying the same factor to our numbers
would be meaningless. The budget for this package's parameters will come out of
the computation, not out of the paper.

This is also the first real test of the `WarpMetric` abstraction, which is why
it comes before the ray tracer: it is the cheapest way to find out whether the
interface was designed correctly, before anything larger is built on top of it.

With the two regions separated, `symbolic.py` gives

```math
\varepsilon = \frac{c^4}{8\pi G}\left[\frac{B'^2}{B^4} - \frac{2B''}{B^3} - \frac{4B'}{r_s B^3}\right] - \frac{c^2 v_s^2}{32\pi G}\,\frac{\rho^2}{r_s^2}\,f'^2
```

i.e. a static term from the curvature of $\gamma_{ij} = B^2\delta_{ij}$ plus the
Alcubierre term, and the expansion is the Alcubierre one. The static term
matches eq. (11) of the paper. It changes sign pointwise, but writing
$\psi = \sqrt{B}$ and integrating by parts gives a strictly positive net budget,
which is the closed-form check in the list above:

```math
E_B = \frac{c^4}{2G}\int_0^\infty \frac{B'^2}{B}\,r^2\,dr
```

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
- the published numbers to reproduce, for $n = 80$, $\alpha = 10^{17}$,
  $\tilde R = \tilde\Delta = 10^{-15}$ m: $E_{II,-} = -1.4 \times 10^{30}$ kg,
  $E_{II,+} = 4.9 \times 10^{30}$ kg, sign change at $w = 0.981$.
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

```math
\varepsilon_{\mathrm{coupling}} = \frac{c^2 v_s^2}{8\pi G}\,\frac{x_s^2}{r_s^2}\left[3\left(\frac{B'}{B}\right)^2(1-f)^2 - 2\,\frac{B'}{B}\,f'\,(1-f)\right], \qquad \theta = v_s\,\frac{x_s}{r_s}\left[f' - 3\,\frac{B'}{B}\,(1-f)\right]
```

Both vanish identically when $B' = 0$ wherever $f \neq 1$, which is why the
separated case comes first: it can be checked term by term against the paper.
The overlapping case has no published reference to compare with, so it is the
one where the derivation carries the whole weight, and it exercises terms of
`symbolic.py` that nothing else in the suite reaches.

## 2. Null-geodesic ray tracing — the view from the bridge

- [x] Hamiltonian ray integrator in the rest frame of the bubble
      (`geodesics.py`)
- [x] Adaptive Dormand–Prince RK45 in `integrators.py`, one step size per ray
- [x] Backwards integration from the ship; one fan of rays in a meridional
      plane instead of one ray per pixel
- [x] Frequency ratio along every line of sight, against the closed form
- [x] Rear horizon as a correctness check, with its surface gravity
- [x] Rendered star field at several $v_s$, for both metrics
      (`viz/sky.py`, `scripts/run_sky.py`)
- [x] Observer away from the centre of the pocket: three-dimensional rays,
      the Bouguer throat and the two windows (`trace_rays_3d`,
      `plot_offcentre_sky`)
- [x] Brightness of the sources: flux $R^4\mu$ for a star, light received
      from an isotropic background, and the effect of lensing on it

The plan written before the implementation had two things wrong, recorded here
because the tests now pin both.

**The metric is static.** In the frame of the bubble, $\xi = x - v_s t$ with
constant $v_s$, nothing depends on time:

```math
ds^2 = -c^2 dt^2 + B^2\left[\left(d\xi - \tilde\beta\,dt\right)^2 + dy^2 + dz^2\right], \qquad \tilde\beta = \beta - v_s = -v_s\,(1-f)
```

so the photon Hamiltonian is conserved,

```math
H = \frac{c}{B}\,|p| + \tilde\beta\,p_\xi
```

The sign convention is the one fixed above: $\beta$ is the drag velocity and
$\beta^x_{\mathrm{ADM}} = -\beta$. Conservation of $H$ gives the frequency
ratio between the ship and a distant source in closed form,
$`E_\mathrm{ship}/E_\mathrm{far} = 1 - (v_s/c)\,n_\xi`$, and the integrator is
tested against it for every ray.

**The dark region is behind the ship, not ahead.** The horizon found by
`horizon_offset` stops signals sent forward from the ship; light arriving from
ahead reaches it freely. Light from behind advances on the ship at
$c/B - v_s(1-f)$, which is $c - v_s < 0$ far away: a superluminal bubble
outruns it. Only sources with $n_\xi < c/v_s$ are visible, a ray traced
towards the rear stalls at $\xi = -h$, and its momentum grows as
$e^{\kappa c t}$ with $\kappa = (v_s/c)|f'(h)|$.

Implementation notes:

- from the centre the sky is axisymmetric, so rays in one plane, interpolated
  over the look angle, place every star; a per-pixel tracer is only needed off
  the centre;
- a step may not cross more than half a wall thickness, because an error
  estimate cannot see a wall that falls between its samples;
- rays whose frequency ratio falls below $10^{-12}$ are stopped as lost at the
  horizon; the Hamiltonian drift is measured relative to $|p|/B$, since near
  the horizon $|p|$ reaches $10^{12}$ and rounding dominates;
- an early draft reversed time both in the right-hand side and in the
  integration limit, tracing rays forwards; the blueshift still matched, since
  $H$ is conserved either way, and only the side on which the rear ray stalled
  exposed it;
- brightness: $I_\nu/\nu^3$ is conserved, so surface brightness scales as
  $R^4$ and a star's flux as $R^4\mu$. The plan expected $\mu$ to drop out of
  the light received from an isotropic sky, leaving the closed form
  $[(1+u)^5 - \max(0,1-u)^5]/(10u)$; it does not, since
  $`\int R^4\,d\Omega_\mathrm{look} = \int R^4\mu\,d\Omega_\mathrm{source}`$. The
  closed form is the unlensed value, 6 % above the traced one at $10c$;
- $\mu$ needs the slope of the map from source to apparent angle. Taken as
  $d\theta_\mathrm{source}/d\theta_\mathrm{look}$ on the evenly spaced look
  angles and inverted it converges at second order, which a test checks;
  differentiated along the uneven source angles it was far less accurate;
- for a ship at the centre the two metrics show the same sky: $B$ is
  spherically symmetric and absent from the blueshift;
- off the centre, $B$ acts on light as a spherically symmetric refractive
  index, so $`B\,r\sin\psi`$ is conserved and the minimum areal radius $`B\,r`$
  outside the pocket is a throat: a ray leaves only if its invariant is below
  it, and the sky is seen through two windows of half-angle
  $\arcsin(R_\mathrm{throat}/\ell_0)$. The invariant is known from the initial
  state, so trapped rays are classified without being integrated; a test
  checks the shortcut against brute-force integration on both sides of the
  cone;
- the windows of different offsets hold nearly the same picture, rescaled with
  the cone. With the paper's parameters the throat is $1.46 \times 10^{-15}$ m
  and the window a metre from the centre is $10^{-15}$ rad across.
