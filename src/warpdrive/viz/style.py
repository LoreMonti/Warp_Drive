# ==========================================================
# Shared plotting style
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --- Palette ---
BG     = "#07090f"     # deep space background
FG     = "#dfe6f2"     # foreground text and ticks
ACCENT = "#5ac8fa"     # hull and bubble wall
TRACER = "#ffe066"     # test particles
EDGE   = "#39405280"   # subtle frame colour


def bubble_colormap():
    """
    Diverging map for the expansion scalar: blue where space contracts,
    the background colour at theta = 0, red where it expands. Keeping
    the neutral value dark stops the vacuum from washing out to white.
    """

    return LinearSegmentedColormap.from_list(
        "warp",
        ["#7fd3ff", "#2b6cff", "#101a33", BG, "#331410", "#c9451b", "#ffb469"],
    )


def dark_axes(ax, three_d=False, grid=True):
    """Apply the palette to a 2D or 3D axis."""

    ax.set_facecolor(BG)

    if three_d:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.03, 0.04, 0.06, 1.0))
            axis.label.set_color(FG)
            axis._axinfo["grid"]["color"] = (0.25, 0.28, 0.35, 0.35)
        ax.tick_params(colors=FG, labelsize=8)
    else:
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        if grid:
            ax.grid(alpha=0.15, color=FG, linewidth=0.5)
        else:
            ax.grid(False)

    ax.title.set_color(FG)
    return ax


def dark_figure(*args, **kwargs):
    """`plt.subplots` with the space palette already applied."""

    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig, axes


def style_colorbar(colorbar, label):
    """Match a colorbar to the palette."""

    colorbar.set_label(label, color=FG)
    colorbar.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    colorbar.outline.set_edgecolor(EDGE)
    return colorbar


def annotation_box():
    """Semi-transparent box for in-axes annotations."""

    return dict(facecolor=BG, edgecolor=EDGE, boxstyle="round,pad=0.35",
                alpha=0.85)


def save(fig, path, dpi=160):
    """Save with the correct background and close the figure."""

    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path
