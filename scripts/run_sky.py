#!/usr/bin/env python3
# ==========================================================
# Driver: the sky seen from the centre of a warp bubble
#
# Usage:
#   python scripts/run_sky.py
#   python scripts/run_sky.py --speeds 0.9 1.5 5
#   python scripts/run_sky.py --no-figures
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import argparse
import math
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
    horizon_surface_gravity,
    trace_rays,
)
from warpdrive.geodesics import sky_map                       # noqa: E402
from warpdrive.viz import plot_sky, plot_sky_mapping          # noqa: E402


DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "output", "sky"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Null rays from the centre of a warp bubble: the sky "
                    "the crew sees."
    )
    parser.add_argument("--speeds", type=float, nargs="+",
                        default=[0.5, 2.0, 10.0],
                        help="bubble speeds in units of c "
                             "(default: 0.5 2 10)")
    parser.add_argument("--radius", type=float, default=100.0,
                        help="bubble radius in metres (default: 100)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="inverse wall thickness in 1/m (default: 1)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="output directory (default: ./output/sky)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the figures")
    return parser.parse_args(argv)


def format_sky_table(bubbles):
    """One line per bubble: what the crew sees and what it cannot."""

    lines = [
        "  THE SKY FROM THE CENTRE OF THE BUBBLE",
        f"  {'v_s/c':>6} | {'ahead':>7} | {'visible up to':>13} | "
        f"{'sky hidden':>10} | {'rear horizon':>12} | {'kappa':>9}",
        "  " + "-" * 72,
    ]
    for metric in bubbles:
        ratio = metric.speed / C_LIGHT
        sky = sky_map(metric)
        hidden = 0.5 * (1.0 - 1.0 / ratio) if ratio > 1.0 else 0.0
        offset = metric.horizon_offset()
        kappa = horizon_surface_gravity(metric)
        horizon = (f"{-offset:>10.3f} m" if offset is not None
                   else f"{'none':>12}")
        gravity = f"{kappa:>7.4f}/m" if kappa is not None else f"{'-':>9}"
        lines.append(
            f"  {ratio:>6g} | {1.0 + ratio:>6.2f}x | "
            f"{math.degrees(sky.visible_limit):>11.2f} ° | "
            f"{100.0 * hidden:>8.1f} % | {horizon} | {gravity}"
        )
    lines.append("  ahead: E_ship / E_far for a star straight ahead, "
                 "1 + v_s/c")
    lines.append("  sky hidden: solid-angle fraction beyond arccos(-c/v_s), "
                 "(1 - c/v_s) / 2")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    bubbles = [AlcubierreMetric(speed=s * C_LIGHT, radius=args.radius,
                                sigma=args.sigma) for s in args.speeds]

    fastest = bubbles[int(np.argmax(args.speeds))]
    broeck = BroeckMetric(speed=fastest.speed, radius=fastest.radius,
                          sigma=fastest.sigma)
    angles = np.linspace(0.0, np.pi, 181)
    a, b = trace_rays(fastest, angles), trace_rays(broeck, angles)
    visible = a.escaped & (a.frequency_ratio > 1e-6)
    difference = np.max(np.abs(a.source_angle[visible]
                               - b.source_angle[visible]))

    if not args.no_figures:
        print("figures ...")
        for path in (plot_sky(bubbles, os.path.join(args.outdir, "sky.png")),
                     plot_sky_mapping(bubbles, os.path.join(
                         args.outdir, "sky_mapping.png"))):
            print("   ", os.path.relpath(path))

    text = format_sky_table(bubbles) + (
        f"\n\n  Van Den Broeck with the same wall at "
        f"{fastest.speed / C_LIGHT:g}c: largest difference in source angle "
        f"{difference:.1e} rad\n"
    )
    print("\n" + text)
    with open(os.path.join(args.outdir, "sky_report.txt"), "w") as handle:
        handle.write(text)

    print(f"output written to {os.path.relpath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
