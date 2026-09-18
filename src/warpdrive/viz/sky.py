# ==========================================================
# The view from the bridge: the sky seen from the centre of the bubble
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# --- Local imports ---
from ..constants import C_LIGHT
from ..geodesics import sky_map
from .style import ACCENT, BG, EDGE, FG, dark_axes, save, style_colorbar

#: Source angles, in degrees from the direction of travel, drawn as rings.
GRID_DEGREES = (30, 60, 90, 120, 150)


def star_field(n_stars=2500, seed=7):
    """
    Stars spread uniformly over the sphere, with a spread of brightness.

    Returns (polar angle from the direction of travel [rad], azimuth
    [rad], relative brightness in (0, 1]).
    """

    rng = np.random.default_rng(seed)
    polar = np.arccos(rng.uniform(-1.0, 1.0, n_stars))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, n_stars)
    brightness = rng.pareto(2.5, n_stars) + 1.0
    return polar, azimuth, brightness / brightness.max()


def star_sizes(brightness, flux_ratio):
    """
    Marker areas for stars of intrinsic `brightness` seen with a flux
    `flux_ratio` times their flux at rest: the area grows as the apparent
    flux to the power 0.2, so a factor of 1e4 still fits on the page
    without merging neighbouring stars, and is clipped to stay visible
    and bounded.
    """

    apparent = np.asarray(brightness) * np.asarray(flux_ratio)
    return np.clip(4.0 * apparent ** 0.2, 0.2, 14.0)


def _ratio_colours():
    """Red below the rest frequency, blue above, log10 of the ratio."""

    return plt.get_cmap("RdBu"), Normalize(vmin=-1.2, vmax=1.2)


def _fisheye(ax, title):
    ax.set_facecolor(BG)
    ax.set_theta_zero_location("N")
    ax.set_ylim(0.0, 180.0)
    ax.set_yticks([45, 90, 135, 180])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.grid(color=FG, alpha=0.08)
    ax.spines["polar"].set_color(EDGE)
    ax.set_title(title, color=FG, fontsize=10, pad=8)


def plot_sky(metrics, path, n_stars=2500, seed=7, n_rays=721):
    """
    All-sky fisheye views from the centre of the bubble, one per metric,
    after a view of the same stars at rest.

    The centre of each disc is straight ahead, the rim straight behind,
    and the radius is the angle from the direction of travel. Stars are
    coloured by the frequency ratio E_ship / E_far and sized by the flux
    they deliver, R^4 mu times their flux at rest, and the rings mark
    where the true source angles of 30, 60, ... degrees appear. The black
    rim is where the light reaching the ship is redshifted below the
    threshold of `sky_map`: it comes from a vanishing sliver of sky next
    to the visible limit.
    """

    polar, azimuth, brightness = star_field(n_stars, seed)
    cmap, norm = _ratio_colours()

    panels = [("at rest", None)] + [
        (f"{m.name}, $v_s$ = {m.speed / C_LIGHT:g}$c$", sky_map(m, n_rays))
        for m in metrics
    ]

    fig = plt.figure(figsize=(4.2 * len(panels), 4.6))
    fig.patch.set_facecolor(BG)

    for i, (title, sky) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="polar")
        _fisheye(ax, title)

        if sky is None:
            look, ratio = polar, np.ones_like(polar)
            flux = np.ones_like(polar)
            rings = {g: float(g) for g in GRID_DEGREES}
        else:
            look, ratio = sky.apparent(polar)
            flux = sky.flux_ratio(polar)
            rings = {g: float(np.degrees(sky.apparent(np.radians(g))[0]))
                     for g in GRID_DEGREES}
            dark = float(np.degrees(sky.look_angle[-1]))
            if dark < 179.0:
                theta = np.linspace(0.0, 2.0 * np.pi, 361)
                ax.fill_between(theta, dark, 180.0, color="#000000",
                                zorder=0)
                ax.text(np.pi, 0.5 * (dark + 180.0), "redshifted to nothing",
                        color=FG, alpha=0.6, ha="center", va="center",
                        fontsize=7)

        seen = np.isfinite(look)
        ax.scatter(azimuth[seen], np.degrees(look[seen]),
                   s=star_sizes(brightness[seen], flux[seen]),
                   c=np.log10(ratio[seen]), cmap=cmap, norm=norm,
                   linewidths=0, zorder=2)

        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        for degrees, radius in rings.items():
            if np.isfinite(radius):
                ax.plot(theta, np.full_like(theta, radius), color=ACCENT,
                        lw=0.6, alpha=0.35, zorder=1)
                ax.text(np.radians(20.0), radius, f"{degrees}°", color=ACCENT,
                        fontsize=6, alpha=0.8)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=fig.axes, orientation="horizontal",
                            fraction=0.04, pad=0.06, aspect=50)
    style_colorbar(colorbar, r"$\log_{10}\,(E_\mathrm{ship}/E_\mathrm{far})$")
    colorbar.ax.xaxis.set_tick_params(color=FG, labelcolor=FG)
    fig.suptitle("The sky from the centre of the bubble: straight ahead at "
                 "the centre, straight behind at the rim", color=FG,
                 fontsize=12, y=0.97)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.02, right=0.98)
    return save(fig, path)


def plot_sky_mapping(metrics, path, n_rays=721):
    """
    Where a source appears, how blueshifted and how much brighter,
    against its true angle, for several bubbles; dotted lines mark the
    visible limit arccos(-c / v_s).
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 4.6))
    fig.patch.set_facecolor(BG)
    colours = plt.get_cmap("plasma")(np.linspace(0.2, 0.85, len(metrics)))

    source = np.linspace(0.0, np.pi, 721)
    ax1.plot(np.degrees(source), np.degrees(source), color=FG, lw=0.8,
             ls="--", alpha=0.5, label="at rest")

    for metric, colour in zip(metrics, colours):
        sky = sky_map(metric, n_rays)
        look, ratio = sky.apparent(source)
        label = rf"$v_s$ = {metric.speed / C_LIGHT:g}$c$"
        ax1.plot(np.degrees(source), np.degrees(look), color=colour, lw=1.8,
                 label=label)
        ax2.semilogy(np.degrees(source), ratio, color=colour, lw=1.8,
                     label=label)
        ax3.semilogy(np.degrees(source), sky.flux_ratio(source),
                     color=colour, lw=1.8, label=label)
        if sky.visible_limit < np.pi:
            for ax in (ax1, ax2, ax3):
                ax.axvline(np.degrees(sky.visible_limit), color=colour,
                           lw=0.8, ls=":")

    xlabel = "true angle of the source from the direction of travel [°]"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("apparent angle [°]")
    ax1.set_title("Where the stars appear")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"$E_\mathrm{ship}/E_\mathrm{far}$")
    ax2.set_title(r"Blueshift, $1 - (v_s/c)\,n_\xi$")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(r"$F_\mathrm{ship}/F_\mathrm{far}$")
    ax3.set_title(r"Flux of a star, $R^4\,\mu$")
    ax3.axhline(1.0, color=FG, lw=0.8, ls="--", alpha=0.5)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0.0, 180.0)
        dark_axes(ax)
        legend = ax.legend(facecolor=BG, edgecolor=EDGE, labelcolor=FG,
                           fontsize=8)
        legend.get_frame().set_alpha(0.85)

    fig.tight_layout()
    return save(fig, path)
