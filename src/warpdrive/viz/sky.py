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
from ..geodesics import (
    TRAPPED,
    sky_map,
    throat_radius,
    trace_rays_3d,
    visible_cone,
)
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


# --- Observers away from the centre ---
def fisheye_directions(resolution):
    """
    Lines of sight for a square fisheye image: straight ahead (+x) at the
    centre, straight behind on the rim, the angle from the direction of
    travel growing linearly with the distance from the centre, and +y to
    the right. Returns (directions (3, n) inside the disc, mask of the
    pixels they fill).
    """

    axis = np.linspace(-1.0, 1.0, resolution)
    u, v = np.meshgrid(axis, -axis)
    radius = np.hypot(u, v)
    inside = radius <= 1.0
    polar = np.pi * radius[inside]
    azimuth = np.arctan2(v[inside], u[inside])
    directions = np.array([np.cos(polar),
                           np.sin(polar) * np.cos(azimuth),
                           np.sin(polar) * np.sin(azimuth)])
    return directions, inside


def checkerboard(source, cell_degrees=15.0):
    """
    Sky texture over true source directions (3, n): 1 or 0.45 on a
    checkerboard in polar angle from the direction of travel and azimuth
    about it, so any distortion of the sky shows as bent cells.
    """

    polar = np.degrees(np.arccos(np.clip(source[0], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(source[2], source[1])) % 360.0
    cells = (np.floor(polar / cell_degrees)
             + np.floor(azimuth / cell_degrees)) % 2
    return np.where(cells == 0, 1.0, 0.45)


def _pixel_colours(source, ratio, cmap, norm):
    rgb = cmap(norm(np.log10(np.clip(ratio, 1e-12, None))))[:, :3]
    return rgb * checkerboard(source)[:, None]


def offcentre_view(metric, proper_offset, resolution=361, n_rays=721):
    """
    RGB fisheye image from a point `proper_offset` metres from the centre
    of the pocket, displaced along +y, perpendicular to the motion.

    Each pixel shows the checkerboard sky at the true direction of its
    source, tinted by the frequency ratio. Pixels whose light stays in the
    pocket are dark blue-grey; pixels lost at the horizon or redshifted
    below 1e-6 are black. From the centre the axisymmetric fan of
    `sky_map` is used, which the tests show agrees with the
    three-dimensional tracer.

    Returns (image (resolution, resolution, 3), fraction of the sky seen
    through the pocket, cone half-angle [rad]).
    """

    cmap, norm = _ratio_colours()
    directions, inside = fisheye_directions(resolution)
    n = directions.shape[1]
    colours = np.zeros((n, 3))
    image = np.zeros((resolution, resolution, 3))
    image[...] = np.array([0.027, 0.035, 0.059])

    if proper_offset == 0.0:
        sky = sky_map(metric, n_rays)
        look = np.arccos(np.clip(directions[0], -1.0, 1.0))
        seen = look <= sky.look_angle[-1]
        source_polar = np.interp(look, sky.look_angle, sky.source_angle)
        ratio = np.interp(look, sky.look_angle, sky.frequency_ratio)
        azimuth = np.arctan2(directions[2], directions[1])
        source = np.array([np.cos(source_polar),
                           np.sin(source_polar) * np.cos(azimuth),
                           np.sin(source_polar) * np.sin(azimuth)])
        colours[seen] = _pixel_colours(source[:, seen], ratio[seen],
                                       cmap, norm)
        fraction, cone = 1.0, np.pi / 2.0
    else:
        centre = float(metric.conformal_factor(0.0, 0.0, 0.0))
        r0 = proper_offset / centre
        field = trace_rays_3d(metric, [0.0, r0, 0.0], directions)
        seen = field.escaped & (field.frequency_ratio > 1e-6)
        colours[seen] = _pixel_colours(-field.far_direction[:, seen],
                                       field.frequency_ratio[seen],
                                       cmap, norm)
        colours[field.status == TRAPPED] = np.array([0.13, 0.15, 0.2])
        fraction = float(np.mean(field.status != TRAPPED))
        cone = visible_cone(metric, r0)

    image[inside] = colours
    return image, fraction, cone


def window_view(metric, proper_offset, half_width, resolution=121):
    """
    RGB image of the outward window alone, from `proper_offset` metres off
    the centre along +y: an azimuthal equidistant map centred on the
    radial line, +x (the direction of travel) to the left, spanning
    `half_width` radians from it in each direction.

    Returns (image, extent in degrees for imshow).
    """

    cmap, norm = _ratio_colours()
    axis = np.linspace(-half_width, half_width, resolution)
    a, b = np.meshgrid(axis, -axis)
    angle = np.hypot(a, b)
    turn = np.arctan2(b, a)
    # +y at the centre; rightwards is -x, so the motion points left
    look = np.array([-np.sin(angle) * np.cos(turn),
                     np.cos(angle),
                     np.sin(angle) * np.sin(turn)]).reshape(3, -1)

    centre = float(metric.conformal_factor(0.0, 0.0, 0.0))
    field = trace_rays_3d(metric, [0.0, proper_offset / centre, 0.0], look)
    seen = field.escaped & (field.frequency_ratio > 1e-6)

    colours = np.zeros((look.shape[1], 3))
    colours[seen] = _pixel_colours(-field.far_direction[:, seen],
                                   field.frequency_ratio[seen], cmap, norm)
    colours[field.status == TRAPPED] = np.array([0.13, 0.15, 0.2])

    degrees = np.degrees(half_width)
    return (colours.reshape(resolution, resolution, 3),
            (-degrees, degrees, -degrees, degrees))


def plot_offcentre_sky(metric, offsets, path, resolution=361,
                       window_resolution=121):
    """
    Fisheye views from several points of the pocket, displaced sideways
    from its centre, on the same axes as `plot_sky`: straight ahead at the
    centre of each disc, straight behind on the rim, +y to the right.

    From the centre the whole sky is open. Off it the outside shows only
    through two windows on the 90 degree ring, about the radial line
    outwards (right) and through the centre (left), of half-angle
    arcsin(R_throat / l0); every other line of sight ends inside the
    pocket. The second row zooms on the outward window of each off-centre
    view, with the circle of the closed-form cone drawn on top.
    """

    fig, grid = plt.subplots(2, len(offsets),
                             figsize=(4.2 * len(offsets), 8.8))
    fig.patch.set_facecolor(BG)
    axes, zooms = grid[0], grid[1]
    cmap, norm = _ratio_colours()

    for ax, zoom, offset in zip(axes, zooms, offsets):
        image, fraction, cone = offcentre_view(metric, offset, resolution)
        ax.imshow(image, extent=(-1, 1, -1, 1), interpolation="nearest")
        rim = plt.Circle((0, 0), 1.0, fill=False, color=EDGE, lw=0.8)
        ring = plt.Circle((0, 0), 0.5, fill=False, color=ACCENT, lw=0.6,
                          alpha=0.4, ls="--")
        ax.add_patch(rim)
        ax.add_patch(ring)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        if offset == 0.0:
            subtitle = "centre: the whole sky"
        else:
            subtitle = (f"{offset:g} m off centre: windows of "
                        f"{np.degrees(cone):.1f}°")
        ax.set_title(subtitle, color=FG, fontsize=10)

        if offset == 0.0:
            zoom.axis("off")
            zoom.text(0.5, 0.5,
                      "below: the outward window\nof each view, zoomed,\n"
                      "with the Bouguer cone\n"
                      r"$\sin\psi_c = R_\mathrm{throat}/\ell_0$ dashed",
                      color=FG, ha="center", va="center", fontsize=10,
                      transform=zoom.transAxes)
            continue
        window, extent = window_view(metric, offset, 1.3 * cone,
                                     window_resolution)
        zoom.imshow(window, extent=extent, interpolation="nearest")
        circle = plt.Circle((0, 0), np.degrees(cone), fill=False,
                            color=ACCENT, lw=1.0, ls="--")
        zoom.add_patch(circle)
        zoom.set_facecolor(BG)
        zoom.set_xlabel("degrees from the radial line, motion to the left",
                        color=FG, fontsize=8)
        zoom.tick_params(colors=FG, labelsize=7)
        for spine in zoom.spines.values():
            spine.set_color(EDGE)

    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.03, right=0.99,
                        wspace=0.18, hspace=0.2)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=fig.add_axes([0.2, 0.05, 0.6, 0.018]),
                            orientation="horizontal")
    style_colorbar(colorbar, r"$\log_{10}\,(E_\mathrm{ship}/E_\mathrm{far})$"
                   "   (checkerboard: 15° cells of the true sky; grey: "
                   "inside the pocket)")
    colorbar.ax.xaxis.set_tick_params(color=FG, labelcolor=FG)
    fig.suptitle(
        f"{metric.name}, $v_s$ = {metric.speed / C_LIGHT:g}$c$: the sky from "
        f"inside a pocket of {metric.pocket_proper_radius():g} m with a "
        f"throat of {throat_radius(metric, metric.inner_radius):.1f} m",
        color=FG, fontsize=12, y=0.985)
    return save(fig, path)
