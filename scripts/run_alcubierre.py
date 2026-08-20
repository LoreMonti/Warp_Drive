#!/usr/bin/env python3
# ==========================================================
# Driver: full Alcubierre bubble study
#
# Usage:
#   python scripts/run_alcubierre.py
#   python scripts/run_alcubierre.py --speed 2 --radius 50 --sigma 0.5
#   python scripts/run_alcubierre.py --no-animation
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

# --- Local imports ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from warpdrive import (                                       # noqa: E402
    AlcubierreMetric,
    C_LIGHT,
    energy_scaling_table,
    format_profile,
    profile_mission,
)
from warpdrive.viz import animate_flyby, plot_all             # noqa: E402


DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "output"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Alcubierre warp bubble: figures, animation and "
                    "mission profile."
    )
    parser.add_argument("--speed", type=float, default=10.0,
                        help="bubble speed in units of c (default: 10)")
    parser.add_argument("--radius", type=float, default=100.0,
                        help="bubble radius in metres (default: 100)")
    parser.add_argument("--sigma", type=float, default=0.10,
                        help="inverse wall thickness in 1/m (default: 0.10)")
    parser.add_argument("--distance", type=float, default=4.2465,
                        help="target distance in light years "
                             "(default: 4.2465, Proxima Centauri)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="output directory (default: ./output)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the static figures")
    parser.add_argument("--no-animation", action="store_true",
                        help="skip the animation, which is the slow part")
    parser.add_argument("--frames", type=int, default=150,
                        help="animation frames (default: 150)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from warpdrive.constants import LY

    metric = AlcubierreMetric(speed=args.speed * C_LIGHT,
                              radius=args.radius,
                              sigma=args.sigma)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"configuration: {metric!r}\n")

    if not args.no_figures:
        print("static figures ...")
        for path in plot_all(metric, args.outdir):
            print("   ", os.path.relpath(path))

    if not args.no_animation:
        print("animation (this is the slow part) ...")
        path = animate_flyby(metric,
                             os.path.join(args.outdir, "05_warp_flyby.gif"),
                             n_frames=args.frames)
        print("   ", os.path.relpath(path))

    profile = profile_mission(metric, distance=args.distance * LY)
    report = format_profile(profile)

    _, scaling = energy_scaling_table(
        lambda speed, radius: AlcubierreMetric(speed=speed, radius=radius,
                                               sigma=args.sigma),
        speeds=[0.5 * C_LIGHT, C_LIGHT, 2.0 * C_LIGHT, 10.0 * C_LIGHT,
                100.0 * C_LIGHT],
        radii=[10.0, 100.0, 1000.0],
    )

    text = report + "\n\n" + scaling + "\n"
    print("\n" + text)

    with open(os.path.join(args.outdir, "mission_report.txt"), "w") as handle:
        handle.write(text)

    print(f"output written to {os.path.relpath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
