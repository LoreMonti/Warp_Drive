# ==========================================================
# Animation of a bubble sweeping past a field of test particles
#
# Author: Lorenzo Monti
# ==========================================================


# --- Third-party imports ---
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Local imports ---
from ..constants import C_LIGHT
from ..tracers import bubble_centre, integrate_tracers, make_tracer_grid
from .style import (
    ACCENT,
    BG,
    FG,
    TRACER,
    annotation_box,
    bubble_colormap,
    dark_axes,
    dark_figure,
    style_colorbar,
)


def animate_flyby(metric, path, n_frames=150, fps=22, tail=12,
                  rows=(0.0, 0.5, 1.0, 1.5, 2.0)):
    """
    Render the bubble crossing a lattice of initially static particles.

    Two behaviours show up, both physical: particles further out than R
    are dragged and released, while particles near the axis are captured
    and carried along, because f = 1 inside the bubble forces them to
    dx/dt = v_s.

    Returns the path of the written GIF.
    """

    radius = metric.radius
    x_start = -4.0 * radius
    t_grid = np.linspace(0.0, 8.0 * radius / metric.speed, n_frames)

    x0, y0 = make_tracer_grid(metric, rows)
    traj = integrate_tracers(metric, x0, y0, t_grid, x_start)

    gx = np.linspace(-4.6 * radius, 4.6 * radius, 260)
    gy = np.linspace(-2.3 * radius, 2.3 * radius, 130)
    GX, GY = np.meshgrid(gx, gy)

    fig, ax = dark_figure(figsize=(10.4, 5.4))
    dark_axes(ax, grid=False)
    ax.set_xlim(gx[0] / radius, gx[-1] / radius)
    ax.set_ylim(gy[0] / radius, gy[-1] / radius)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / R$")
    ax.set_ylabel(r"$\rho / R$")

    theta0 = metric.expansion(GX - x_start, GY, 0.0)
    vmax = np.abs(theta0).max()
    mesh = ax.pcolormesh(GX / radius, GY / radius, theta0,
                         cmap=bubble_colormap(), vmin=-vmax, vmax=vmax,
                         shading="gouraud")
    style_colorbar(fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.028),
                   r"$\theta$  [s$^{-1}$]")

    dots = ax.scatter(x0 / radius, y0 / radius, s=9, c=TRACER, alpha=0.9,
                      zorder=4, linewidths=0)
    trails = [ax.plot([], [], color=TRACER, lw=0.7, alpha=0.35, zorder=3)[0]
              for _ in range(len(x0))]
    ship, = ax.plot([], [], marker=">", color="w", ms=12, zorder=6)
    wall, = ax.plot([], [], color=ACCENT, lw=1.0, ls="--", alpha=0.45,
                    zorder=5)

    box = annotation_box()
    clock = ax.text(0.014, 0.955, "", transform=ax.transAxes, color=FG,
                    fontsize=8.5, family="monospace", va="top", zorder=7,
                    bbox=box)
    ax.text(0.014, 0.035,
            "on-axis particles are captured by the bubble and carried along",
            transform=ax.transAxes, color=TRACER, fontsize=8, alpha=0.85,
            va="bottom", zorder=7, bbox=box)

    fig.suptitle(
        rf"{metric.name} bubble at $v_s$ = {metric.speed / C_LIGHT:.0f}$c$   "
        r"$-$   space contracts ahead (blue), expands behind (red)",
        color=FG, fontsize=12, y=0.965,
    )

    rate = metric.proper_time_rate()
    phi = np.linspace(0.0, 2.0 * np.pi, 240)

    def update(frame):
        centre = bubble_centre(t_grid[frame], x_start, metric.speed)
        mesh.set_array(metric.expansion(GX - centre, GY, 0.0).ravel())
        dots.set_offsets(np.column_stack([traj[frame] / radius, y0 / radius]))

        lo = max(0, frame - tail)
        for j, line in enumerate(trails):
            line.set_data(traj[lo:frame + 1, j] / radius,
                          np.full(frame + 1 - lo, y0[j] / radius))

        ship.set_data([centre / radius], [0.0])
        wall.set_data((centre + radius * np.cos(phi)) / radius, np.sin(phi))
        clock.set_text(
            f"crew proper time  tau = {t_grid[frame] * rate * 1e6:6.3f} us\n"
            f"coordinate time     t = {t_grid[frame] * 1e6:6.3f} us\n"
            f"ship speed through local space = 0"
        )
        return (mesh, dots, ship, wall, clock, *trails)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False,
                         interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": BG})
    return path
