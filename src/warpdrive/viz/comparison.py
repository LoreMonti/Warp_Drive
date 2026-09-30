# ==========================================================
# Comparison figures: Alcubierre against Van Den Broeck
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT, G, L_PLANCK, M_SUN
from .figures import energy_norm, energy_ticks
from .style import (
    ACCENT,
    BG,
    EDGE,
    FG,
    TRACER,
    bubble_colormap,
    dark_axes,
    dark_figure,
    save,
    style_colorbar,
)


def _legend(ax, **kwargs):
    legend = ax.legend(facecolor=BG, edgecolor=EDGE, labelcolor=FG,
                       fontsize=8, **kwargs)
    legend.get_frame().set_alpha(0.85)
    return legend


def plot_metric_comparison(alcubierre, broeck, path, extent=1.3,
                           resolution=500, n_radial=20000):
    """
    The two spacetimes on identical axes, lengths in units of the
    Alcubierre radius.

    Top: energy density in the meridional plane, on one shared symmetric
    logarithmic colour scale. Bottom left: the density along a radius
    perpendicular to the motion, where both walls are crossed and the
    Alcubierre term is largest. Bottom right: the conformal factor and
    the proper radius l(r) = \\int_0^r B dr', which shows the pocket
    holding more space than its coordinate size.
    """

    scale = alcubierre.radius
    span = extent * scale
    grid = np.linspace(-span, span, resolution)
    X, RHO = np.meshgrid(grid, grid)

    maps = [alcubierre.energy_density(X, RHO, 0.0),
            broeck.energy_density(X, RHO, 0.0)]
    norm = energy_norm(np.concatenate([m.ravel() for m in maps]))

    fig, axes = dark_figure(2, 2, figsize=(12.0, 10.5))

    for ax, eps, metric in zip(axes[0], maps, (alcubierre, broeck)):
        mesh = ax.pcolormesh(X / scale, RHO / scale, eps,
                             cmap=bubble_colormap(), norm=norm,
                             shading="auto")
        ax.set_title(metric.name)
        ax.set_xlabel(r"$(x - x_s)/R_A$")
        ax.set_ylabel(r"$\rho / R_A$")
        ax.set_aspect("equal")
        dark_axes(ax)
    style_colorbar(fig.colorbar(mesh, ax=list(axes[0]), pad=0.02,
                                shrink=0.9, ticks=energy_ticks(norm)),
                   r"energy density  [J m$^{-3}$], symmetric log")

    # --- density along the transverse radius
    ax = axes[1, 0]
    r = np.linspace(0.0, span, n_radial)
    for metric, colour, style in ((alcubierre, ACCENT, "-"),
                                  (broeck, TRACER, "--")):
        ax.plot(r / scale, metric.energy_density(0.0, r, 0.0), style,
                color=colour, lw=1.6, label=metric.name)
    ax.set_yscale("symlog", linthresh=norm.linthresh)
    ax.set_ylim(-1.5 * norm.vmax, 1.5 * norm.vmax)
    ax.set_yticks(energy_ticks(norm, step=2))
    ax.axhline(0.0, color=FG, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"$\rho / R_A$   (at $x = x_s$)")
    ax.set_ylabel(r"$\varepsilon$  [J m$^{-3}$], symmetric log")
    ax.set_title("Across both walls")
    dark_axes(ax)
    _legend(ax, loc="upper right")

    # --- conformal factor and proper radius
    ax = axes[1, 1]
    conformal = np.asarray(broeck.conformal_factor(r, 0.0, 0.0), dtype=float)
    proper = np.concatenate([[0.0], np.cumsum(
        0.5 * (conformal[1:] + conformal[:-1]) * np.diff(r))])

    ax.plot(r / scale, proper / scale, color=TRACER, lw=1.8,
            label=rf"{broeck.name}: $\ell(r) = \int_0^r B\,dr'$")
    ax.plot(r / scale, r / scale, color=ACCENT, lw=1.4, ls="-",
            label=rf"{alcubierre.name}: $\ell(r) = r$")
    ax.set_xlabel(r"coordinate radius  $r / R_A$")
    ax.set_ylabel(r"proper radius  $\ell / R_A$")
    ax.set_title("How much space the pocket holds")
    dark_axes(ax)
    _legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.08))

    twin = ax.twinx()
    twin.plot(r / scale, conformal, color=FG, lw=1.0, ls=":",
              label=r"$B(r)$, right axis")
    twin.set_yscale("log")
    twin.set_ylabel(r"$B(r)$", color=FG)
    twin.tick_params(colors=FG, labelsize=9)
    for spine in twin.spines.values():
        spine.set_color(EDGE)
    _legend(twin, loc="center right")

    fig.suptitle("Same shift wall, with and without a pocket", color=FG,
                 fontsize=14)
    return save(fig, path)


def plot_neck_scaling(scaling, path, paper_neck=3.0e-15):
    """
    Exotic and positive mass of Van Den Broeck bubbles against the neck
    radius, from a `NeckScaling`, with the Alcubierre bubble the neck
    replaces as a horizontal line.
    """

    R = scaling.neck_radius
    to_sun = 1.0 / (M_SUN * C_LIGHT ** 2)

    fig, ax = dark_figure(figsize=(8.5, 6.2))

    ax.loglog(R, -scaling.total_negative * to_sun, color=ACCENT, lw=2.6,
              label=r"total negative, $|M_-|$")
    ax.loglog(R, -scaling.wall * to_sun, color=ACCENT, lw=1.2, ls="--",
              label=r"shift wall, $\propto R^2$")
    ax.loglog(R, -scaling.transition_negative * to_sun, color=FG, lw=1.2,
              ls=":", label=r"transition of $B$, negative")
    ax.loglog(R, scaling.transition_positive * to_sun, color=TRACER, lw=2.0,
              label=r"transition of $B$, positive $M_+$")
    ax.axhline(-scaling.alcubierre * to_sun, color="#ff6b6b", lw=1.6,
               label="Alcubierre bubble as large as the pocket")
    ax.axvline(paper_neck, color=FG, lw=0.8, alpha=0.5)
    ax.annotate("1999 paper", (paper_neck, 1.0), (4.0 * paper_neck, 1e4),
                color=FG, fontsize=8,
                arrowprops=dict(color=FG, arrowstyle="->", lw=0.8))

    ax.set_xlabel(r"neck radius $R$  [m]")
    ax.set_ylabel(r"mass  [$M_\odot$]")
    ax.set_title("Shrinking the neck: the wall pays less,\n"
                 "the pocket's transition region sets the floor")
    dark_axes(ax)
    _legend(ax, loc="center right")

    fig.tight_layout()
    return save(fig, path)


def plot_pocket_throat(broeck, path, n_points=40001):
    """
    The throat of the pocket, against the proper radial distance from its
    centre: the areal radius A = B r on top, the radial null contraction
    eps + p_r below, in units of c^4 / 8 pi G per square metre.

    A pocket larger inside than outside forces A to fall and rise again;
    at its minimum, the throat, A is convex in proper distance and the
    null energy condition fails. The region round the maximum of A, where
    it holds, balances the violation exactly in the integral
    int (eps + p_r) A dl = 0.
    """

    inner = broeck.inner_radius
    outer = inner + broeck.thickness
    r = np.linspace(0.8 * inner, outer + 0.25 * broeck.thickness, n_points)
    conformal = broeck.conformal_profile(r)
    proper = np.concatenate([[0.0], np.cumsum(
        0.5 * (conformal[1:] + conformal[:-1]) * np.diff(r))])
    proper += float(broeck.conformal_profile(0.0)) * r[0]
    areal = broeck.areal_radius(r)
    unit = C_LIGHT ** 4 / (8.0 * np.pi * G)
    null = broeck.pocket_null_energy(r) / unit

    radius, throat_areal = broeck.throat()
    throat_proper = float(np.interp(radius, r, proper))
    peak = int(np.argmax(areal * (r <= radius)))

    fig, (top, bottom) = dark_figure(2, 1, figsize=(8.5, 7.0), sharex=True)

    top.plot(proper, areal, color=TRACER, lw=2.0,
             label=r"areal radius $A = B\,r$")
    top.plot(proper, proper, color=FG, lw=0.8, ls="--", alpha=0.5,
             label=r"flat space, $A = \ell$")
    top.plot(throat_proper, throat_areal, "o", color=ACCENT, ms=7,
             label=f"throat: A = {throat_areal:.2f} m")
    top.plot(proper[peak], areal[peak], "s", color="#ff9f43", ms=6,
             label=f"maximum: A = {areal[peak]:.1f} m")
    top.set_ylabel(r"$A$  [m]")
    top.set_title("The pocket is a throat: A falls to a minimum and "
                  "rises again")
    dark_axes(top)
    _legend(top, loc="upper right")

    bottom.plot(proper, null, color=ACCENT, lw=1.6)
    bottom.fill_between(proper, null, 0.0, where=null < 0.0, color=ACCENT,
                        alpha=0.25, label="null energy condition violated")
    bottom.fill_between(proper, null, 0.0, where=null > 0.0, color="#ff9f43",
                        alpha=0.25, label="null energy condition holds")
    bottom.axhline(0.0, color=FG, lw=0.6, alpha=0.4)
    bottom.axvline(throat_proper, color=ACCENT, lw=0.8, ls=":")
    bottom.set_yscale("symlog", linthresh=1e-3)
    bottom.set_xlabel(r"proper distance from the centre  $\ell$  [m]")
    bottom.set_ylabel(r"$(\varepsilon + p_r)\,/\,(c^4/8\pi G)$  [m$^{-2}$]")
    bottom.set_title(r"$\varepsilon + p_r = -\frac{c^4}{8\pi G}\,"
                     r"\frac{2}{A}\,\frac{d^2A}{d\ell^2}$, "
                     "independent of the speed of the bubble")
    dark_axes(bottom)
    _legend(bottom, loc="lower left")

    fig.tight_layout()
    return save(fig, path)


def plot_energy_floor(path, pocket_radius=100.0, outer_radius=20.0,
                      orders=(10, 80), n_inner=40):
    """
    The net E_tot of Van Den Broeck transition regions against the lower
    bounds of section 3b, in solar masses.

    Left: at fixed proper pocket radius P and outer radius b, polynomial
    profiles of several orders against the inner radius a, above the
    Dirichlet bound for each a, whose minimum over a is the floor
    (2 c^4/G)(P - b), reached at a* = b^2/P. Right: the floor against the
    excess P - b, with the default and the 1999 configurations and the
    energy of their actual profiles.
    """

    # local imports keep viz free of a hard dependency on diagnostics
    from ..diagnostics import pocket_energy_floor
    from ..metrics.broeck import BroeckMetric

    to_sun = 1.0 / (M_SUN * C_LIGHT ** 2)
    P, b = pocket_radius, outer_radius
    floor, best = pocket_energy_floor(P, b)

    inner = np.geomspace(0.02 * b, 0.9 * b, n_inner)
    bound = [BroeckMetric(speed=C_LIGHT, inner_radius=a, thickness=b - a,
                          alpha=P / a - 1.0).pocket_energy_bound()
             for a in inner]

    fig, (left, right) = dark_figure(1, 2, figsize=(12.5, 5.0))
    colours = plt.get_cmap("plasma")(np.linspace(0.3, 0.8, len(orders)))
    for order, colour in zip(orders, colours):
        actual = [BroeckMetric(speed=C_LIGHT, inner_radius=a,
                               thickness=b - a, alpha=P / a - 1.0,
                               order=order).transition_energy_budget().net
                  for a in inner]
        left.loglog(inner, np.array(actual) * to_sun, color=colour, lw=1.8,
                    label=f"polynomial profile, n = {order}")
    left.loglog(inner, np.array(bound) * to_sun, color=ACCENT, lw=2.2,
                label="Dirichlet bound at each a")
    left.axhline(floor * to_sun, color=FG, lw=1.0, ls="--",
                 label=r"floor $2c^4(P - b)/G$")
    left.plot(best, floor * to_sun, "o", color=FG, ms=6)
    left.annotate(r"$a^* = b^2/P$", (best, floor * to_sun),
                  (best * 0.3, floor * to_sun * 1.6), color=FG, fontsize=9,
                  arrowprops=dict(color=FG, arrowstyle="->", lw=0.8))
    left.set_xlabel(r"inner radius of the transition  $a$  [m]")
    left.set_ylabel(r"net $E_\mathrm{tot}$  [$M_\odot$]")
    left.set_title(f"P = {P:g} m, b = {b:g} m: no profile goes below "
                   "the bound")
    dark_axes(left)
    _legend(left, loc="upper left")

    excess = np.geomspace(1e-2, 1e4, 200)
    right.loglog(excess, 2.0 * C_LIGHT ** 4 / G * excess * to_sun,
                 color=FG, lw=1.2, ls="--", label="floor")
    for metric, name, colour in (
            (BroeckMetric(speed=C_LIGHT), "default", TRACER),
            (BroeckMetric.from_paper(), "1999 paper", ACCENT)):
        outer = metric.inner_radius + metric.thickness
        gap = metric.pocket_proper_radius() - outer
        actual = metric.transition_energy_budget().net * to_sun
        right.plot(gap, actual, "o", color=colour, ms=7,
                   label=f"{name}: actual profile")
        right.plot(gap, pocket_energy_floor(
            metric.pocket_proper_radius(), outer)[0] * to_sun, "x",
            color=colour, ms=8)
    right.set_xlabel(r"excess of the pocket over its outside  $P - b$  [m]")
    right.set_ylabel(r"net $E_\mathrm{tot}$  [$M_\odot$]")
    right.set_title("The floor grows with how much larger the pocket is "
                    "inside")
    dark_axes(right)
    _legend(right, loc="upper left")

    fig.tight_layout()
    return save(fig, path)


def plot_quantum_inequality(path, pocket_radius=100.0, n_thickness=40,
                            n_points=100000):
    """
    Margin of the Ford-Roman quantum inequality, |peak density| / |limit|,
    for Van Den Broeck transition regions of thickness D~ = R~ enclosing a
    pocket of fixed proper radius P, so alpha = P / D~ - 1.

    The margin grows roughly as D~^2 (the peak goes as 1/D~^2, the limit
    as 1/r_c^4 ~ 1/D~^4), with a slow drift from the position of the peak,
    which moves with alpha. The inequality holds below the thickness where
    the margin crosses 1, and the curvature radius falls below the Planck
    length a little further down: the window between the two is where the
    1999 check sits. Both of its configurations have P = 100 m.

    Left: the margin against D~, with the two configurations of the paper
    and both thresholds. Right: the curvature radius in Planck lengths
    against the same axis.
    """

    # local imports keep viz free of a hard dependency on diagnostics
    from ..diagnostics import quantum_inequality_check
    from ..metrics.broeck import BroeckMetric

    def pocket(thickness):
        return BroeckMetric(speed=C_LIGHT, radius=3.0 * thickness,
                            sigma=1.0e2 / thickness,
                            inner_radius=thickness, thickness=thickness,
                            alpha=pocket_radius / thickness - 1.0)

    thickness = np.geomspace(1.0e-35, 1.0e-12, n_thickness)
    checks = [quantum_inequality_check(pocket(d), n_points=n_points,
                                       n_speeds=1) for d in thickness]
    margin = np.array([c.margin for c in checks])
    curvature = np.array([c.curvature_radius for c in checks]) / L_PLANCK

    # both curves are power laws to high accuracy: interpolate in log-log
    log_d = np.log(thickness)
    holds = float(np.exp(np.interp(0.0, np.log(margin), log_d)))
    planck = float(np.exp(np.interp(0.0, np.log(curvature), log_d)))

    fig, (left, right) = dark_figure(1, 2, figsize=(12.5, 5.0))
    for ax in (left, right):
        ax.axvspan(planck, holds, color=TRACER, alpha=0.15, lw=0,
                   label="inequality holds, $r_c > \\ell_P$")
        ax.axvline(holds, color=TRACER, lw=1.0, ls="--")
        ax.axvline(planck, color=FG, lw=1.0, ls=":")

    left.loglog(thickness, margin, color=ACCENT, lw=2.0,
                label="margin, observers at rest")
    left.axhline(1.0, color=FG, lw=1.0, ls="--")
    for d, name in ((1.0e-15, "eq. (7): energies"),
                    (1.0e-32, "eq. (22): the check")):
        check = quantum_inequality_check(pocket(d), n_points=n_points,
                                         n_speeds=1)
        left.plot(d, check.margin, "o", color=FG, ms=7)
        left.annotate(f"{name}\n{check.margin:.1e}", (d, check.margin),
                      (d * 30.0, check.margin * 1e-4), color=FG,
                      fontsize=9,
                      arrowprops=dict(color=FG, arrowstyle="->", lw=0.8))
    left.set_xlabel(r"thickness of the transition  $\tilde\Delta = \tilde R$  [m]")
    left.set_ylabel(r"$|\rho_\mathrm{peak}|\,/\,|\rho_\mathrm{QI}|$")
    left.set_title(f"P = {pocket_radius:g} m: the inequality holds only "
                   f"below {holds:.1e} m")
    dark_axes(left)
    _legend(left, loc="upper left")

    right.loglog(thickness, curvature, color=ACCENT, lw=2.0,
                 label=r"smallest curvature radius $r_c$")
    right.axhline(1.0, color=FG, lw=1.0, ls=":")
    right.set_xlabel(r"thickness of the transition  $\tilde\Delta = \tilde R$  [m]")
    right.set_ylabel(r"$r_c\,/\,\ell_P$")
    right.set_title(f"...and r_c reaches the Planck length at {planck:.1e} m")
    dark_axes(right)
    _legend(right, loc="upper left")

    fig.tight_layout()
    return save(fig, path)


def plot_curvature_bound(path, pocket_radius=100.0, outer_radius=20.0,
                         orders=(4, 10, 20, 40, 80, 160), n_inner=12,
                         n_points=100000):
    """
    What a bound on the curvature radius costs.

    Left: net E_tot over the floor (2 c^4/G)(P - b) against the smallest
    curvature radius r_c, at fixed proper pocket radius P and outer radius
    b. Polynomial profiles of several orders, each for a range of inner
    radii, against `ConePocket`, which stays on the floor up to a
    correction of order r_c^2: the bound on r_c does not force the
    factor the polynomials pay.

    Right: D~ / r_c of the polynomial against its order n, for the
    alpha = 1e34 of the first arXiv versions of the 1999 paper and the
    alpha = 1e17 of its published eq. (7), with the paper's n = 80.
    """

    # local imports keep viz free of a hard dependency on diagnostics
    from ..diagnostics import (ConePocket, curvature_radius_by_order,
                               pocket_energy_floor)
    from ..metrics.broeck import BroeckMetric

    P, b = pocket_radius, outer_radius
    floor, _ = pocket_energy_floor(P, b)

    fig, (left, right) = dark_figure(1, 2, figsize=(12.5, 5.0))
    colours = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, len(orders)))
    for order, colour in zip(orders, colours):
        radii, energies = [], []
        for a in np.geomspace(0.05 * b, 0.9 * b, n_inner):
            pocket = BroeckMetric(speed=C_LIGHT, inner_radius=a,
                                  thickness=b - a, alpha=P / a - 1.0,
                                  order=order)
            radii.append(pocket.curvature_radius(n_points)[0])
            energies.append(pocket.transition_energy_budget(n_points).net)
        left.loglog(np.array(radii) / b, np.array(energies) / floor, "o-",
                    color=colour, ms=3, lw=1.0,
                    label=f"polynomial, n = {order}")

    r_min = np.geomspace(1e-4, 0.45, 200) * b
    cone = [ConePocket(P, b, r).energy() / floor for r in r_min]
    left.loglog(r_min / b, cone, color=ACCENT, lw=2.4,
                label="rounded cone")
    left.axhline(1.0, color=FG, lw=1.0, ls="--",
                 label=r"floor $2c^4(P - b)/G$")
    left.set_xlabel(r"smallest curvature radius  $r_c / b$")
    left.set_ylabel(r"net $E_\mathrm{tot}$ / floor")
    left.set_title(f"P = {P:g} m, b = {b:g} m: the curvature bound "
                   "costs almost nothing")
    dark_axes(left)
    _legend(left, loc="upper right", ncol=2)

    orders_scan = np.arange(10, 161)
    for alpha, name, colour in ((1.0e34, r"$\alpha = 10^{34}$ (v1-v4)",
                                 ACCENT),
                                (1.0e17, r"$\alpha = 10^{17}$, eq. (7)",
                                 TRACER)):
        ratios = curvature_radius_by_order(alpha, orders_scan, n_points)
        k = int(np.argmin(ratios))
        right.semilogy(orders_scan, ratios, color=colour, lw=2.0,
                       label=f"{name}: best n = {orders_scan[k]}")
        right.plot(orders_scan[k], ratios[k], "o", color=colour, ms=6)
    right.axvline(80, color=FG, lw=1.0, ls=":", label="the paper's n = 80")
    right.set_xlabel("order of the polynomial  $n$")
    right.set_ylabel(r"$\tilde\Delta / r_c$  (smaller is smoother)")
    right.set_title("n = 80 is optimal only for the first configuration")
    dark_axes(right)
    _legend(right, loc="upper right")

    fig.tight_layout()
    return save(fig, path)
