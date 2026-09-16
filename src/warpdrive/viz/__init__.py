# --- Visualisation ---
from .animation import animate_flyby
from .comparison import plot_metric_comparison, plot_neck_scaling
from .figures import (
    plot_all,
    plot_energy_density,
    plot_expansion_scalar,
    plot_shape_function,
    plot_shell_3d,
)

__all__ = [
    "animate_flyby",
    "plot_all",
    "plot_energy_density",
    "plot_expansion_scalar",
    "plot_shape_function",
    "plot_shell_3d",
    "plot_metric_comparison",
    "plot_neck_scaling",
]
