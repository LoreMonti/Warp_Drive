#!/usr/bin/env python3
# ==========================================================
# Driver: Van Den Broeck two-scale bubble against Alcubierre
#
# Usage:
#   python scripts/run_broeck.py
#   python scripts/run_broeck.py --speed 1 --pocket 50
#   python scripts/run_broeck.py --no-figures
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
    format_profile,
    neck_scaling,
    profile_mission,
)
from warpdrive.constants import LY, M_SUN                     # noqa: E402
from warpdrive.viz import (                                   # noqa: E402
    plot_all,
    plot_metric_comparison,
    plot_neck_scaling,
)


DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "output", "broeck"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Van Den Broeck warp bubble: comparison with Alcubierre, "
                    "neck scaling and mission profiles."
    )
    parser.add_argument("--speed", type=float, default=10.0,
                        help="bubble speed in units of c (default: 10)")
    parser.add_argument("--pocket", type=float, default=100.0,
                        help="proper pocket radius in metres for the neck "
                             "scan (default: 100)")
    parser.add_argument("--distance", type=float, default=4.2465,
                        help="target distance in light years "
                             "(default: 4.2465, Proxima Centauri)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="output directory (default: ./output/broeck)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip all figures")
    return parser.parse_args(argv)


def format_neck_scaling(scaling):
    """Tabulate a NeckScaling in solar masses."""

    to_sun = 1.0 / (M_SUN * C_LIGHT ** 2)
    lines = [
        "  NECK SCALING   (solar masses, pocket and wall fixed)",
        f"  {'R [m]':>10} | {'wall':>10} | {'B, neg':>10} | {'B, pos':>10}"
        f" | {'total neg':>10}",
        "  " + "-" * 64,
    ]
    for i, radius in enumerate(scaling.neck_radius):
        lines.append(
            f"  {radius:>10.1e} | {scaling.wall[i] * to_sun:>10.2e} | "
            f"{scaling.transition_negative[i] * to_sun:>10.2e} | "
            f"{scaling.transition_positive[i] * to_sun:>10.2e} | "
            f"{scaling.total_negative[i] * to_sun:>10.2e}"
        )
    lines.append(f"  Alcubierre bubble of the pocket's size: "
                 f"{scaling.alcubierre * to_sun:.2e}")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    speed = args.speed * C_LIGHT
    os.makedirs(args.outdir, exist_ok=True)

    # macroscopic configuration: resolvable by every generic tool
    broeck = BroeckMetric(speed=speed)
    alcubierre = AlcubierreMetric(speed=speed, radius=broeck.radius,
                                  sigma=broeck.sigma)
    paper = BroeckMetric.from_paper(speed=speed)
    print(f"configuration: {broeck!r}\n")

    scaling = neck_scaling(np.logspace(np.log10(3.0e-15),
                                       np.log10(args.pocket), 60),
                           pocket_radius=args.pocket, speed=speed)

    if not args.no_figures:
        print("figures ...")
        paths = plot_all(broeck, args.outdir) + [
            plot_metric_comparison(
                alcubierre, broeck,
                os.path.join(args.outdir, "05_comparison.png")),
            plot_neck_scaling(
                scaling, os.path.join(args.outdir, "06_neck_scaling.png")),
        ]
        for path in paths:
            print("   ", os.path.relpath(path))

    distance = args.distance * LY
    table_radii = np.logspace(np.log10(3.0e-15), np.log10(args.pocket), 9)
    table = neck_scaling(table_radii, pocket_radius=args.pocket, speed=speed)

    text = "\n\n".join([
        format_profile(profile_mission(broeck, distance=distance)),
        "  1999 PAPER CONFIGURATION",
        format_profile(profile_mission(paper, distance=distance)),
        format_neck_scaling(table),
    ]) + "\n"
    print("\n" + text)

    with open(os.path.join(args.outdir, "mission_report.txt"), "w") as handle:
        handle.write(text)

    print(f"output written to {os.path.relpath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
