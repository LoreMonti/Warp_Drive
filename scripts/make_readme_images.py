#!/usr/bin/env python3
# ==========================================================
# Regenerate every image shown in the README
#
# The README images are the only tracked figures. They are produced here,
# with their parameters fixed in code, rather than copied by hand from a
# driver run, so that any of them can be rebuilt and checked.
#
# Usage:
#   python scripts/make_readme_images.py
#   python scripts/make_readme_images.py --no-animation
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import argparse
import os
import sys

# --- Third-party imports ---
import matplotlib

matplotlib.use("Agg")

import numpy as np                                            # noqa: E402

# --- Local imports ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from warpdrive import (                                       # noqa: E402
    AlcubierreMetric,
    BroeckMetric,
    C_LIGHT,
    neck_scaling,
)
from warpdrive.viz import (                                   # noqa: E402
    animate_flyby,
    plot_energy_density,
    plot_energy_floor,
    plot_expansion_scalar,
    plot_metric_comparison,
    plot_neck_scaling,
    plot_offcentre_sky,
    plot_pocket_throat,
    plot_shell_3d,
    plot_sky,
    plot_sky_mapping,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
README_IMAGES = os.path.join(ROOT, "images", "readme")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the images shown in the README."
    )
    parser.add_argument("--no-animation", action="store_true",
                        help="skip the flyby animation, the slow part")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(README_IMAGES, exist_ok=True)

    def target(name):
        return os.path.join(README_IMAGES, name)

    # the Alcubierre bubble of the README's sample output
    alcubierre = AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0,
                                  sigma=0.1)
    paths = [
        plot_expansion_scalar(alcubierre,
                              target("alcubierre_expansion_scalar.png")),
        plot_energy_density(alcubierre,
                            target("alcubierre_energy_density.png")),
        plot_shell_3d(alcubierre, target("alcubierre_shell_3d.png")),
    ]
    if not args.no_animation:
        # 80 frames at 72 dpi keep the GIF near 2 MB
        paths.append(animate_flyby(alcubierre,
                                   target("alcubierre_flyby.gif"),
                                   n_frames=80, fps=16, dpi=72))

    # Van Den Broeck against Alcubierre with the same 1 m wall, at 10c
    broeck = BroeckMetric(speed=10.0 * C_LIGHT)
    same_wall = AlcubierreMetric(speed=broeck.speed, radius=broeck.radius,
                                 sigma=broeck.sigma)
    paths.append(plot_metric_comparison(same_wall, broeck,
                                        target("broeck_comparison.png")))
    paths.append(plot_pocket_throat(broeck, target("broeck_throat.png")))
    paths.append(plot_energy_floor(target("broeck_energy_floor.png")))

    # neck scan at v_s = c, 100 m pocket, wall of 1e2 Planck lengths
    necks = np.logspace(np.log10(3.0e-15), np.log10(100.0), 60)
    paths.append(plot_neck_scaling(neck_scaling(necks, pocket_radius=100.0,
                                                speed=C_LIGHT),
                                   target("broeck_neck_scaling.png")))

    # the sky from the centre, 1 m wall, three speeds
    bubbles = [AlcubierreMetric(speed=s * C_LIGHT, radius=100.0, sigma=1.0)
               for s in (0.5, 2.0, 10.0)]
    paths.append(plot_sky(bubbles, target("sky_fisheye.png")))
    paths.append(plot_sky_mapping(bubbles, target("sky_mapping.png")))

    # the sky from four points of the default pocket, at 10c
    paths.append(plot_offcentre_sky(broeck, [0.0, 30.0, 60.0, 90.0],
                                    target("sky_offcentre.png")))

    for path in paths:
        print(os.path.relpath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
