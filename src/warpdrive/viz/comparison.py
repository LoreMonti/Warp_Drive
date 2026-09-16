# ==========================================================
# Comparison figures: Alcubierre against Van Den Broeck
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import numpy as np

# --- Local imports ---
from ..constants import C_LIGHT, M_SUN
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
