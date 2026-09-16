# ==========================================================
# Static figures
#
# Every routine takes a configured WarpMetric and an output path, so the
# same set of plots can be produced for any member of the metric family.
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import os

# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

# --- Local imports ---
from ..shapes import tanh_top_hat, tanh_top_hat_derivative
from .style import (
    ACCENT,
    BG,
    FG,
    bubble_colormap,
    dark_axes,
    dark_figure,
    save,
    style_colorbar,
)


def energy_norm(eps, decades=4):
    """
    Symmetric logarithmic colour scale for an energy density map.

    Linear within `decades` orders of magnitude of the largest |eps| and
    logarithmic beyond, identical for both signs, so a positive region
    many times weaker than the negative one still shows, and two maps
    passed the same norm share one colour scale.
    """

    peak = float(np.max(np.abs(eps)))
    if peak == 0.0:
        peak = 1.0
    return SymLogNorm(linthresh=peak * 10.0 ** (-decades), vmin=-peak,
                      vmax=peak, base=10)


def energy_ticks(norm, step=1):
    """
    Ticks for a symmetric logarithmic energy scale: zero and every
    `step`-th decade above the linear threshold, for both signs. Decades
    inside the linear band are skipped, since they crowd around zero.
    """

    low = int(np.ceil(np.log10(norm.linthresh))) + 1
    high = int(np.floor(np.log10(norm.vmax)))
    decades = [10.0 ** k for k in range(high, low - 1, -step)]
    return sorted([-d for d in decades] + [0.0] + decades)


def energy_title(eps):
    """Describe the sign content of a density map from the data itself."""

    if np.max(eps) <= 0.0:
        return ("Exotic matter: negative energy density everywhere,\n"
                "distributed as a torus around the axis of motion")
    return ("Energy density: negative (blue) and positive (red),\n"
            "the weak energy condition is violated where it is blue")


def plot_shape_function(metric, path, sigmas=(0.05, None, 0.4)):
    """
    The shape function and its derivative, for a few wall thicknesses,
    plus the conformal factor B(r_s) when the metric has a non-trivial
    one.

    Passing None inside `sigmas` substitutes the metric's own value, which
    is drawn solid.
    """

    own = metric.sigma
    sigmas = sorted({own if s is None else s for s in sigmas})
    r = np.linspace(0.0, 2.5 * metric.radius, 2000)

    conformal = np.asarray(metric.conformal_factor(r, 0.0, 0.0), dtype=float)
    inflated = not np.all(conformal == 1.0)

    fig, axes = dark_figure(3 if inflated else 2, 1,
                            figsize=(7.5, 9.0 if inflated else 6.5),
                            sharex=True)
    ax1, ax2 = axes[0], axes[1]

    others = iter(["--", ":", "-."])
    for sigma in sigmas:
        dashes = "-" if sigma == own else next(others)
        ax1.plot(r / metric.radius, tanh_top_hat(r, metric.radius, sigma),
                 dashes, lw=2.0,
                 label=rf"$\sigma R$ = {sigma * metric.radius:.0f}")
        ax2.plot(r / metric.radius,
                 tanh_top_hat_derivative(r, metric.radius, sigma)
                 * metric.radius,
                 dashes, lw=2.0)

    ax1.set_ylabel(r"$f(r_s)$")
    ax2.set_ylabel(r"$R\,\mathrm{d}f/\mathrm{d}r_s$")

    if inflated:
        ax3 = axes[2]
        ax3.plot(r / metric.radius, conformal, color=ACCENT, lw=2.0)
        ax3.set_yscale("log")
        ax3.set_ylabel(r"$B(r_s)$")

    for ax in axes:
        ax.axvline(1.0, color="#ff9f43", lw=1.0, alpha=0.6)
        dark_axes(ax)

    axes[-1].set_xlabel(r"$r_s / R$")
    ax1.set_title("Shape function: flat inside, flat outside,\n"
                  "all the curvature squeezed into the wall")

    legend = ax1.legend(facecolor=BG, edgecolor="#394052", labelcolor=FG,
                        fontsize=9)
    legend.get_frame().set_alpha(0.8)

    fig.tight_layout()
    return save(fig, path)


def plot_expansion_scalar(metric, path, extent=2.0, resolution=220):
    """The iconic Alcubierre surface: contraction ahead, expansion behind."""

    span = extent * metric.radius
    grid = np.linspace(-span, span, resolution)
    X, RHO = np.meshgrid(grid, grid)
    theta = metric.expansion(X, RHO, 0.0)

    fig = plt.figure(figsize=(9.0, 6.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X / metric.radius, RHO / metric.radius, theta,
                    cmap="coolwarm", rstride=2, cstride=2,
                    linewidth=0, antialiased=True, alpha=0.95)

    ax.set_xlabel(r"$(x - x_s)/R$   $\rightarrow$ direction of travel")
    ax.set_ylabel(r"$\rho / R$")
    ax.set_zlabel(r"$\theta$  [s$^{-1}$]", labelpad=8)
    ax.set_title(r"Expansion scalar $\theta = v_s\,(x_s/r_s)\,f'(r_s)$"
                 "\ncontraction ahead (blue), expansion behind (red)")
    ax.view_init(elev=28, azim=-58)
    dark_axes(ax, three_d=True)

    fig.tight_layout()
    return save(fig, path)


def plot_energy_density(metric, path, extent=2.0, resolution=400):
    """
    The energy density in the meridional plane, on a symmetric
    logarithmic colour scale: blue where it is negative, red where it is
    positive, dark where it vanishes.

    For Alcubierre it is negative everywhere it is non-zero and vanishes
    on the axis: a torus wrapped around the direction of motion. A
    conformal factor adds a shell carrying both signs.
    """

    span = extent * metric.radius
    grid = np.linspace(-span, span, resolution)
    X, RHO = np.meshgrid(grid, grid)
    eps = metric.energy_density(X, RHO, 0.0)

    fig, ax = dark_figure(figsize=(7.6, 6.4))
    mesh = ax.pcolormesh(X / metric.radius, RHO / metric.radius, eps,
                         cmap=bubble_colormap(),
                         norm=energy_norm(eps), shading="auto")
    norm = mesh.norm
    style_colorbar(fig.colorbar(mesh, ax=ax, pad=0.02,
                                ticks=energy_ticks(norm)),
                   r"energy density  [J m$^{-3}$], symmetric log")

    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=ACCENT, lw=1.0,
                            ls="--", alpha=0.7))
    ax.plot(0, 0, marker="^", color="#ffffff", ms=9, zorder=5)
    ax.annotate("ship\n(flat space, at rest)", (0, 0), (0.15, 0.55),
                color=FG, fontsize=8,
                arrowprops=dict(color=FG, arrowstyle="->", lw=0.8))

    ax.set_xlabel(r"$(x - x_s)/R$")
    ax.set_ylabel(r"$\rho / R$")
    ax.set_title(energy_title(eps))
    ax.set_aspect("equal")
    dark_axes(ax)

    fig.tight_layout()
    return save(fig, path)


def plot_shell_3d(metric, path, n_samples=400000, seed=42):
    """
    Monte-Carlo rendering of the exotic-matter shell.

    Points are accepted with a probability proportional to the negative
    part of eps, so the visual density of the cloud is the physical
    density of negative energy; positive energy, where a metric has any,
    is not drawn. The IXS-style twin rings are drawn where that matter has to
    be held.
    """

    rng = np.random.default_rng(seed)
    span = 1.6 * metric.radius
    points = rng.uniform(-span, span, size=(n_samples, 3))
    exotic = np.maximum(-metric.energy_density(points[:, 0], points[:, 1],
                                               points[:, 2]), 0.0)
    weight = exotic / max(float(exotic.max()), np.finfo(float).tiny)

    keep = rng.random(len(weight)) < weight
    points, weight = points[keep], weight[keep]

    colours = plt.get_cmap("plasma")(0.25 + 0.75 * weight)
    colours[:, 3] = 0.10 + 0.35 * weight

    fig = plt.figure(figsize=(8.5, 7.8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0] / metric.radius, points[:, 1] / metric.radius,
               points[:, 2] / metric.radius, c=colours, s=5.0, linewidths=0,
               depthshade=False)

    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    for x_ring in (-0.32, 0.32):
        ax.plot(np.full_like(phi, x_ring), 0.97 * np.cos(phi),
                0.97 * np.sin(phi), color=ACCENT, lw=2.6, alpha=0.95,
                zorder=10)
    ax.plot([-0.5, 0.52], [0, 0], [0, 0], color="#f2f5fa", lw=6.0,
            solid_capstyle="round", zorder=11)
    ax.plot([0.52, 0.62], [0, 0], [0, 0], color="#f2f5fa", lw=2.5,
            solid_capstyle="round", zorder=11)

    ax.set_xlabel(r"$x/R$")
    ax.set_ylabel(r"$y/R$")
    ax.set_zlabel(r"$z/R$")
    ax.set_title("Negative-energy shell and an IXS-style twin-ring hull\n"
                 "the rings are drawn where the exotic matter has to live")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-64)
    dark_axes(ax, three_d=True)

    fig.tight_layout()
    return save(fig, path)


def plot_all(metric, outdir):
    """Produce the full static set and return the list of paths."""

    os.makedirs(outdir, exist_ok=True)
    return [
        plot_shape_function(metric, os.path.join(outdir,
                                                 "01_shape_function.png")),
        plot_expansion_scalar(metric, os.path.join(outdir,
                                                   "02_expansion_scalar.png")),
        plot_energy_density(metric, os.path.join(outdir,
                                                 "03_energy_density.png")),
        plot_shell_3d(metric, os.path.join(outdir, "04_shell_3d.png")),
    ]
