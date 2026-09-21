# Roadmap

Status of the `warpdrive` package. Checked items are implemented and covered by
the test suite; unchecked ones are planned.

| section | topic | status |
| --- | --- | --- |
| 1 | Van Den Broeck's two-scale bubble | done |
| 1b | Overlapping regions of $B$ and $f$ | to do |
| 2 | Null-geodesic ray tracing, the view from the bridge | done |
| 3 | The pocket: throat geometry, energy floor, the 1999 checks | to do |
| 4 | Hawking radiation through the throat | research |
| 5 | An acoustic Van Den Broeck bubble | research |
| – | Minor items | to do |

Every open section starts with a literature check: a result is only worth
pursuing once the full texts of the closest papers show it is not already
known.

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

## 1b. Overlapping regions of $B$ and $f$

- [ ] Literature: check whether configurations with the transition of $B$
      inside the shift wall have been studied, in Van Den Broeck's follow-ups,
      Loup's Natário–Broeck papers and Gauthier, Gravel and Melanson (2002)
- [ ] `BroeckMetric` variant that allows the transition region of $B$ to
      overlap the wall of $f$
- [ ] Coupling terms of the energy density and the modified expansion, written
      from `symbolic.py`
- [ ] Test against `derive(conformal=True)` at points inside the overlap, where
      every coupling coefficient is non-zero
- [ ] Energy budget against the separated configuration
- [ ] Rays: inside the overlap $f \neq 1$, so Bouguer's invariant is no longer
      exact; check what survives of the throat and the two windows

When the two regions overlap, the derivation adds

```math
\varepsilon_{\mathrm{coupling}} = \frac{c^2 v_s^2}{8\pi G}\,\frac{x_s^2}{r_s^2}\left[3\left(\frac{B'}{B}\right)^2(1-f)^2 - 2\,\frac{B'}{B}\,f'\,(1-f)\right], \qquad \theta = v_s\,\frac{x_s}{r_s}\left[f' - 3\,\frac{B'}{B}\,(1-f)\right]
```

Both vanish identically when $B' = 0$ wherever $f \neq 1$, which is why the
separated case came first: it can be checked term by term against the paper.
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

Prior art: the view from the centre of an Alcubierre bubble was computed by
Clark, Hiscock and Larson (Class. Quantum Grav. **16**, 3965, 1999) and studied
in detail by Müller and Weiskopf (Gen. Rel. Grav., arXiv:1107.5650); this
section reproduces it independently. The brightness of the sky and the view
from away from the centre of a Van Den Broeck pocket were not found in the
literature, and are the starting point of sections 3 to 5.

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

## 3. The pocket: throat geometry, energy floor, the 1999 checks

What the first round of reading showed, and how it reshaped this section:

- the vanishing ADM energy of the pocket is **not new**: Barzegar, Buchert and
  Vigneron (2026, Theorem IV.19) prove it for warp drives with flat slices,
  and for Van Den Broeck it follows from the definition, since the spatial
  metric is flat outside a compact region;
- the Eulerian energy integrated over a slice, $E_\mathrm{tot}$, the quantity
  Van Den Broeck and Pfenning and Ford use, is called **ambiguous** in the same
  paper (Error 9): it depends on the foliation, is not conserved and is not the
  mass of the bubble. A lower bound on it is only meaningful as a statement
  about that quantity;
- no lower bound of Dirichlet type was found in Lobo and Visser (2004),
  Schuster, Santiago and Visser (2023) or Barzegar et al. (2026); Schuster et
  al. list conformally flat slices, the class Van Den Broeck belongs to, as
  future work;
- the throat of the pocket is a surface of minimal area, like the throat of a
  wormhole. Hochberg and Visser showed that such a flaring-out throat forces a
  violation of the null energy condition. That statement is independent of the
  observer and of the foliation, and does not involve $v_s$; no one was found
  to have applied it to Van Den Broeck's pocket.

Literature:

- [x] Barzegar, Buchert and Vigneron (2026): ADM energy, positive-energy
      theorem, and the ambiguity of $E_\mathrm{tot}$
- [x] Lobo and Visser (2004): no bound on the energy of the pocket
- [x] Schuster, Santiago and Visser (2023): no treatment of Van Den Broeck; the
      conformally flat class is left as future work
- [ ] Gauthier, Gravel and Melanson, *New lower bounds for warp drive energy*,
      Int. J. Mod. Phys. A **17**, 2761 (2002): a single page, not freely
      available; needs library access
- [x] Hochberg and Visser, *Geometric structure of the generic static
      traversable wormhole throat* (1997): the throat is defined geometrically,
      as a minimal surface in a static slice with a flare-out condition, and
      throats of trivial topology are explicitly included, with a closed region
      joined to flat space by a narrow neck as their example; the strong
      flare-out condition implies a violation of the null energy condition at
      the throat
- [ ] Morris and Thorne (1988) and Hochberg and Visser's dynamic follow-up
      (1998), for completeness
- [ ] A Baylor thesis on curvature invariants of wormholes and warped
      spacetimes, flagged by a search, to confirm no one has described the
      pocket as a wormhole of trivial topology
- [ ] Loup's Natário–Broeck papers and *Warp drive basics* (2021), for any
      statement on the throat or on a lower energy bound
- [ ] Initial-data literature on conformally flat, time-symmetric slices
      (Brill, Brill–Lindquist): whether the Dirichlet bound below is known in
      that setting

3a — The throat as a minimal surface:

- [x] Show that inside the shift wall, in the frame of the bubble, the metric
      is ultrastatic, $`-c^2 dt^2 + B^2\delta_{ij}\,dx^i dx^j`$, so the slices are
      time-symmetric there and the throat, where the areal radius $`B\,r`$ is
      smallest, is a minimal surface
- [x] Derive the radial pressure from the Einstein tensor in `symbolic.py`,
      and the null contraction $\varepsilon + p_r$ for radial light rays
- [x] Verify that $\varepsilon + p_r < 0$ at the throat for every profile, and
      at every $v_s$, as the flare-out theorem requires; locate where the
      violation sits
- [x] Tests: the sign at the throat for several profiles, the null
      contraction against the derivation, and flat space
- [x] Figure of the areal radius and the null energy across the pocket
      (`plot_pocket_throat`)

What 3a found: the radial null contraction inside the shift wall is
$`\varepsilon + p_r = -(c^4/8\pi G)\,(2/A)\,d^2A/d\ell^2`$, derived twice, in the
moving Cartesian chart and in a static spherical one, which agree to $10^{-9}$.
A pocket larger inside than outside has a minimal sphere where the condition
fails, for any profile and any $v_s$; the condition is sufficient, not
necessary. The violation integrates to zero against $`A\,d\ell`$, balanced by the
region round the maximum of $A$.

3b — A lower bound on $E_\mathrm{tot}$:

- [ ] Show from the derivation that the net $E_\mathrm{tot}$ of the pocket is a
      Dirichlet integral of $\psi = \sqrt{B}$
- [ ] Prove the lower bound below for every profile, with equality for the
      harmonic $\psi$, and the floor at fixed proper pocket radius $P$
- [ ] `pocket_energy_bound(metric)`, and a test that every profile tried lies
      above it
- [ ] Minimum-$E_\mathrm{tot}$ profile under a lower bound on the curvature
      radius, the constraint Van Den Broeck used to choose $n = 80$
- [ ] State explicitly, following Barzegar et al., that this bounds the
      quantity used by Van Den Broeck and Pfenning and Ford, not a mass

3c — The checks of the 1999 paper:

- [ ] Re-derive the peak density and the curvature radius of eqs. (14) and (20)
      of the 1999 paper, which do not match its own profile
- [ ] Redo its quantum-inequality check with the corrected numbers
- [ ] The same check for observers crossing the transition region at high
      speed, not only the Eulerian ones

Write-up:

- [ ] Note in LaTeX, compiled with tectonic, with 3a, 3b and 3c
- [ ] README section and figure

With $\psi = \sqrt{B}$, $a = \tilde R$ and $b = \tilde R + \tilde\Delta$, the
net $E_\mathrm{tot}$ of the transition region and its lower bound are

```math
E_B = \frac{2c^4}{G}\int_a^b \psi'^2\,r^2\,dr \;\ge\; \frac{2c^4}{G}\,\left(\sqrt{B_\mathrm{max}} - 1\right)^2\frac{ab}{b - a}
```

and for a pocket of proper radius $P$ much larger than the neck the bound tends
to a floor that depends on neither the profile nor the neck,

```math
M_B \gtrsim \frac{2c^2 P}{G} \approx 0.14\,M_\odot \times \frac{P}{100\ \mathrm{m}}
```

A first numerical check: the profile of the 1999 paper lies 6.5 times above
the bound, the polynomial of order 10 1.46 times, and no profile tried falls
below it.

## 4. Hawking radiation through the throat

The question: does the pocket shield the crew from the Hawking radiation of
the warp horizon, and by how much?

Literature:

- [ ] Finazzi, Liberati and Barceló (2009, 2012) on the semiclassical warp
      drive: where the Hawking flux goes and what fills the interior
- [ ] Hiscock (1997) on quantum effects in the Alcubierre spacetime
- [ ] Greybody factors for spherically symmetric barriers: the Schrödinger-like
      radial equation, WKB and numerical methods used for black holes
- [ ] Any work combining a Van Den Broeck pocket with quantum fields, to
      confirm the question is open

Theory:

- [ ] Justify a stationary horizon in the frame of the bubble and its
      temperature, and state the 3+1 assumption explicitly, since the Hawking
      flux of a warp drive is established in 1+1 dimensions
- [ ] Scalar field in the static region inside the shift wall: separation in
      spherical harmonics, radial equation and effective potential built from
      $B(r)$
- [ ] Transmission $\Gamma_\ell(\omega)$ through the throat; show that its
      eikonal limit reproduces the Bouguer windows of section 2
- [ ] Flux reaching the pocket: Planck spectrum times $\Gamma_\ell(\omega)$, as
      a function of $\Xi$ and of the position of the observer

Code and tests:

- [ ] Radial wave solver, tested on flat space ($\Gamma = 1$), on a barrier
      with an analytic transmission, and against the Bouguer windows at high
      frequency
- [ ] Suppression of the Hawking flux against $\Xi$ and the offset, table and
      figure

Write-up:

- [ ] Results written as the first half of the paper shared with section 5

The temperature and the parameter that decides between wave and ray optics are

```math
T_H = \frac{\hbar c\,\kappa}{2\pi k_B}, \qquad \kappa = \frac{v_s}{c}\,\left|f'(h)\right|, \qquad \Xi = \frac{\kappa\,R_\mathrm{throat}}{2\pi}
```

A mode of frequency $\omega$ and angular momentum $\ell$ crosses the throat
only if $\ell \lesssim \omega R_\mathrm{throat}/c$. For the default bubble
$\Xi \approx 3$ and the wave calculation is needed; for the 1999 configuration
$\Xi \sim 10^{15}$ and the ray result applies, which predicts a suppression of
order $(R_\mathrm{throat}/\ell_0)^2$, about $10^{-30}$ a metre from the centre.

## 5. An acoustic Van Den Broeck bubble

The question: does a pocket of slow sound filter analogue Hawking radiation
measurably, and with what signature in the correlation spectrum?

Literature:

- [ ] Barceló, Liberati and Visser, *Analogue gravity* (Living Reviews)
- [ ] Finazzi et al. (2012) on warp drives in Bose–Einstein condensates
- [ ] Steinhauer and collaborators (2016, 2019, 2021) on the measured analogue
      Hawking radiation and its correlation spectrum
- [ ] Greybody factors in acoustic black holes, and condensates with a
      position-dependent speed of sound
- [ ] Any acoustic pocket or slow-sound region used as a filter, to confirm
      the question is open

Theory:

- [ ] Acoustic metric with flow $\mathbf v(\mathbf x)$ and speed of sound
      $c_s(\mathbf x)$, and its conformal relation to the Van Den Broeck
      metric through $c_s = c/B$
- [ ] The phonon wave equation: rays coincide with the relativistic ones, but
      the conformal factor changes the effective potential, so the greybody
      factor differs from section 4 and has to be computed on its own
- [ ] Realistic condensate parameters: speed of sound, healing length and the
      frequency above which Bogoliubov dispersion breaks the analogy, compared
      with the frequencies that cross the throat
- [ ] Realisable geometry: a one- or two-dimensional pocket first, the density
      profile that produces it and the transonic flow that makes the horizon
- [ ] Decision point: if no realistic configuration exists, the paper rests on
      section 4 alone

Observable:

- [ ] Density–density correlation spectrum with and without the pocket, and
      the signature of the filtering

Machine learning (optional):

- [ ] Simulation-based inference of $\kappa$ and $R_\mathrm{throat}$ from
      noisy correlation spectra, with normalising flows trained on the forward
      model

Write-up:

- [ ] One paper covering sections 4 and 5: the same ray optics, two greybody
      factors, and the laboratory prediction

The acoustic line element is conformal to the Van Den Broeck one,

```math
ds^2 \propto -c_s^2\,dt^2 + \left(d\mathbf x - \mathbf v\,dt\right)^2, \qquad c_s = \frac{c}{B}, \quad \mathbf v = \text{shift}
```

so sound rays follow exactly the light rays computed in section 2, while the
waves feel a different potential.

## Minor items

- [ ] README scaling table: write $-4.7 \times 10^{-1}$ and
      $-1.9 \times 10^{2}$ as plain numbers, as was done for $10^0$
- [ ] 3D shell figure: sample points near the walls, which are nearly empty
      for thin walls with uniform sampling
- [ ] `--help` of the drivers: say that default paths are relative to the
      repository root, not to the working directory
