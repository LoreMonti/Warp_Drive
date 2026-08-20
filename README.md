# Warp Drive

A numerical study of the **Alcubierre (1994) warp drive** metric — the geometry
behind every "IXS Enterprise"-style concept ship, with its two coaxial rings
wrapped around a central hull.

The metric, in ADM (3+1) form, is

```
ds² = -c² dt² + (dx - v_s f(r_s) dt)² + dy² + dz²
```

with the bubble centred on `x_s(t)`, `v_s = dx_s/dt`, and
`r_s = √((x - x_s)² + y² + z²)`. The shape function `f` goes from 1 inside the
bubble to 0 outside:

```
f(r_s) = [tanh(σ(r_s + R)) - tanh(σ(r_s - R))] / [2 tanh(σR)]
```

Both regions are **exactly flat**. All the curvature lives in a wall of
thickness `~1/σ` at `r_s = R`. The ship never moves through space: it sits at
rest in a flat patch while the wall contracts space ahead of it and expands it
behind. Special relativity is never violated, because `v_s` is a *coordinate*
velocity, not a local one.

![Alcubierre bubble sweeping past a field of test particles](docs/assets/warp_flyby.gif)

## Install

```bash
git clone https://github.com/lorenzomonti/warp-drive.git
cd warp-drive
pip install -e ".[dev]"
```

Runtime dependencies are `numpy` and `matplotlib` only; `pytest` is needed
just for the test suite.

## Usage

```bash
python scripts/run_alcubierre.py
python scripts/run_alcubierre.py --speed 2 --radius 50 --sigma 0.5
python scripts/run_alcubierre.py --no-animation
```

The script writes four figures, one animation and a mission report to
`output/`, which is not tracked. As a library:

```python
from warpdrive import AlcubierreMetric, profile_mission, format_profile
from warpdrive.constants import C_LIGHT

metric = AlcubierreMetric(speed=10 * C_LIGHT, radius=100.0, sigma=0.1)
print(format_profile(profile_mission(metric)))
```

## Layout

```
src/warpdrive/
├── constants.py       physical constants and unit conversions
├── shapes.py          radial profiles f(r), and the B(r) of the roadmap
├── integrators.py     RK4, vectorised over an ensemble of particles
├── tracers.py         Eulerian congruence dragged by a passing bubble
├── diagnostics.py     travel times, energy budget, causal structure
├── metrics/
│   ├── base.py        WarpMetric: the 3+1 interface
│   └── alcubierre.py  the 1994 metric
└── viz/
    ├── style.py       shared palette
    ├── figures.py     static figures
    └── animation.py   flyby animation
scripts/               command line drivers
tests/                 pytest suite
```

Every metric is written in the ADM form

```
ds² = -c² dt² + B(r_s)² [ (dx - β(r_s) dt)² + dy² + dz² ]
```

so a concrete spacetime is fixed by two radial profiles: the shift `β`, which
drags the coordinates, and the conformal factor `B`, which inflates spatial
volume. Alcubierre has `B = 1` and `β = v_s f(r_s)`; Van Den Broeck keeps that
shift and adds a non-trivial `B`. Everything that can be derived from the
interface — proper time, the total energy budget, the horizon — lives in
`WarpMetric` and is written once, so the figures and the driver never need to
know which metric they were handed.

## Tests

```bash
pytest
```

Errors in general relativity rarely crash: a wrong sign or a missing factor of
`c` produces plausible numbers. The suite pins the invariants that would catch
that — `f(0) = 1`, `dτ/dt = 1` at any `v_s`, `ε ≤ 0` everywhere, `ε = 0` on the
axis, `θ` antisymmetric under `x → -x`, superluminal motion in the *exterior*
rejected as spacelike, `E ∝ v_s²R²σ`, and the horizon solver agreeing with the
analytic condition `f = 1 - c/v_s`. The closed-form energy budget is also
checked against the generic quadrature in the base class, which is the test
that keeps the interface honest when a second metric is added.

## What the code computes

**Expansion scalar** — `θ = v_s (x_s/r_s) f'(r_s)`, negative ahead of the ship
(space contracting) and positive behind it (space expanding), reproducing the
surface from the original paper.

![Expansion scalar](docs/assets/02_expansion_scalar.png)

**Energy density** seen by Eulerian observers,

```
ε = -(c⁴/8πG)(v_s²/c²)(ρ²/4r_s²)(df/dr_s)²,    ρ² = y² + z²
```

It is **negative everywhere it is non-zero** — the drive violates the weak
energy condition — and its distribution is a **torus** around the axis of
motion. That torus is the physical reason concept ships are drawn with rings:
the rings mark where the exotic matter has to be held.

![Exotic matter distribution](docs/assets/03_energy_density.png)
![Negative-energy shell and twin-ring hull](docs/assets/04_shell_3d.png)

**Energy budget** — the angular integral is analytic, leaving

```
E = -(c² v_s² / 12G) ∫₀^∞ (df/dr)² r² dr
```

**Causal structure** — for `v_s > c` a photon emitted forward along the axis
obeys `dx_s/dt = c - v_s[1 - f]`, which vanishes where `f = 1 - c/v_s`. A
horizon forms inside the bubble wall: the crew cannot signal the front of their
own bubble, so it cannot be steered, slowed, or switched off from the inside.

**Proper time** — evaluated from the line element rather than assumed. On the
ship worldline `f = 1` and `dx/dt = v_s`, so `dτ = dt` exactly: no time
dilation at any `v_s`.

## Sample output (R = 100 m, 1/σ = 10 m, v_s = 10c)

```
Proxima Centauri, 4.2465 ly
  coordinate time  t   = 0.4246 yr
  crew proper time τ   = 0.4246 yr        (dτ/dt = 1.000000)
  1g relativistic rocket, same trip: τ = 3.54 yr, t = 5.87 yr

  exotic energy    E   = -3.37e+47 J
  mass equivalent  M   = -3.75e+30 kg  =  -1.89 solar masses
  future horizon at 89.0 m ahead of the ship
```

Scaling of the required exotic mass (solar masses):

| v_s/c | R = 10 m | R = 100 m | R = 1000 m |
| ---: | ---: | ---: | ---: |
| 0.5 | -9.7e-05 | -4.7e-03 | -4.7e-01 |
| 1 | -3.9e-04 | -1.9e-02 | -1.9e+00 |
| 2 | -1.6e-03 | -7.6e-02 | -7.5e+00 |
| 10 | -3.9e-02 | -1.9e+00 | -1.9e+02 |
| 100 | -3.9e+00 | -1.9e+02 | -1.9e+04 |

`E ∝ v_s² R² σ`. Alcubierre's own thin-wall estimate gave a *negative* mass
larger than the whole visible universe; the numbers above are milder only
because the wall here is 10 m thick rather than sub-nuclear.

## The honest part

The simulation is an exact solution of Einstein's equations — but that is a
weaker statement than it sounds. General relativity lets you write down *any*
metric and then read off, from `G_μν = 8πG/c⁴ T_μν`, the stress-energy needed
to hold it up. Alcubierre ran the equations backwards: he chose the geometry he
wanted and computed the matter it demands. The matter it demands does not exist.

Known obstructions, in rough order of severity:

1. **Exotic matter.** The required `T_μν` violates the weak, null, and dominant
   energy conditions. Casimir-type effects give tiny, static, laboratory-scale
   negative energy densities — nothing remotely like a macroscopic shell.
2. **Quantum inequalities.** Pfenning & Ford (1997) showed the wall must be
   thinner than ~10² Planck lengths, which drives the energy requirement back
   up to absurd values (the mission report prints the wall thickness in Planck
   lengths so you can see how far off it is).
3. **The horizon.** Computed above: the bubble cannot be controlled from
   within, so it must be laid down in advance along the entire route — which
   requires something already at the destination.
4. **Causality.** Two superluminal bubbles on different trajectories can be
   combined into a closed timelike curve.
5. **Semiclassical instability.** The wall behaves like a horizon and radiates;
   Finazzi, Liberati & Barceló (2009) found the interior temperature diverges,
   destabilising the bubble.
6. **The bulldozer problem.** The animation shows it directly: particles near
   the axis are captured, since `f = 1` forces them to `dx/dt = v_s`. The
   bubble sweeps up interstellar matter and releases it, extremely blueshifted,
   at the destination (McMonigal, Lewis & O'Byrne 2012).

Modern work moves toward *subluminal* solitons with positive energy —
Bobrick & Martire (2021), Lentz (2021), Fell & Heisenberg (2021),
Schuster et al. (2023) — which are physically far more respectable but no
longer faster than light.

## Roadmap

### 1. Null-geodesic ray tracing — the view from the bridge

Integrate null geodesics of the Alcubierre metric backwards from the ship to
reconstruct what the crew would actually see through a window.

The metric is not static, so the geodesics have to be integrated in the full
4D spacetime rather than reduced to an effective potential. In ADM form with
lapse `α = 1`, shift `β^x = -v_s f(r_s)` and flat spatial metric, the
Hamiltonian for a photon is

```
H = β^i p_i + √(δ^{ij} p_i p_j)
```

so the ray equations are `dx^i/dλ = ∂H/∂p_i`, `dp_i/dλ = -∂H/∂x^i`, with the
shift supplying all the coupling. Practical notes:

- integrate **backwards** in time from the observer, one ray per pixel, and
  map the escaping direction onto a background star field or an equirectangular
  sky texture;
- `integrators.py` already vectorises RK4 over an ensemble, so a batch of
  rays costs the same as one; the wall, however, needs adaptive stepping —
  `f'` is sharply peaked and a fixed step will walk straight through it, so an
  embedded RK45 belongs in the same module;
- carry the photon frequency along each ray to get the **blueshift map** at the
  front wall, which is the observable that makes the bulldozer problem
  quantitative rather than anecdotal;
- for `v_s > c` the horizon computed by `horizon_offset` shows up as a region
  no backwards ray can reach — a black disc ahead of the ship. That is a
  useful correctness check on the integrator: the disc must appear exactly
  where the bisection puts it.

Deliverable: a rendered star field seen from inside the bubble at several
`v_s`, plus the blueshift map, sharing the shape function with the existing
code.

### 2. Van Den Broeck's two-scale bubble

Van Den Broeck (1999) noticed that the energy requirement scales with the
*surface* of the bubble, not with the volume it encloses. Adding a second
shape function `B(r_s)` that inflates the spatial volume inside a microscopic
neck,

```
ds² = -c² dt² + B(r_s)²[(dx - v_s f(r_s) dt)² + dy² + dz²]
```

produces a pocket with a large interior volume (metres across, enough for the
ship) hidden behind an outer surface of nuclear size. Since `E ∝ v_s² R² σ` and
`R` is now the radius of the *neck*, the total exotic energy drops by roughly
**thirty orders of magnitude** — from stellar masses down to gram-scale, a few
solar masses becoming a few milligrams.

Implementation plan:

- add the `B` profile to `shapes.py` and a `BroeckMetric` in `metrics/`
  overriding `conformal_factor`; the interior is no longer conformally flat,
  so `expansion` and `energy_density` must be re-derived rather than inherited
  — `B` and its first two derivatives both enter `T_μν`;
- the generic quadrature in `WarpMetric.total_exotic_energy` already carries
  the `√γ = B³` factor, so the budget needs no new integrator, and the test
  comparing it against the Alcubierre closed form guards the shared path;
- the figures and the driver take a metric, so the two spacetimes can be
  compared side by side on identical axes with no changes to `viz/`;
- add the energy scaling plot that is the whole point: exotic mass vs neck
  radius, showing the collapse from `M_sun` to milligrams;
- the honest counterpart: the quantum inequalities of Pfenning & Ford still
  apply to the neck, the interior volume has to be seeded somehow, and the
  causal and stability problems listed above are untouched. The variant makes
  the drive cheap, not physical.

Reference: C. Van Den Broeck, *A "warp drive" with more reasonable total energy
requirements*, Class. Quantum Grav. **16**, 3973 (1999).

## References

- M. Alcubierre, *The warp drive: hyper-fast travel within general relativity*,
  Class. Quantum Grav. **11**, L73 (1994)
- M. J. Pfenning & L. H. Ford, Class. Quantum Grav. **14**, 1743 (1997)
- C. Van Den Broeck, Class. Quantum Grav. **16**, 3973 (1999)
- S. Finazzi, S. Liberati & C. Barceló, Phys. Rev. D **79**, 124017 (2009)
- B. McMonigal, G. F. Lewis & P. O'Byrne, Phys. Rev. D **85**, 064024 (2012)
- A. Bobrick & G. Martire, Class. Quantum Grav. **38**, 105009 (2021)
