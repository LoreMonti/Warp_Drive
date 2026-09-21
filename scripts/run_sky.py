#!/usr/bin/env python3
# ==========================================================
# Driver: the sky seen from the centre of a warp bubble
#
# Usage:
#   python scripts/run_sky.py
#   python scripts/run_sky.py --speeds 0.9 1.5 5
#   python scripts/run_sky.py --offsets 0 20 50 100
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
    throat_radius,
    trace_rays,
    unlensed_brightness,
    visible_cone,
)
from warpdrive.geodesics import sky_map                       # noqa: E402
from warpdrive.viz import (                                   # noqa: E402
    plot_offcentre_sky,
    plot_sky,
    plot_sky_mapping,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
DEFAULT_FIGURES = os.path.join(ROOT, "images", "local", "sky")
DEFAULT_REPORTS = os.path.join(ROOT, "reports")


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
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[0.0, 30.0, 60.0, 90.0],
                        help="proper distances from the centre of the "
                             "Van Den Broeck pocket for the off-centre "
                             "views, in metres (default: 0 30 60 90)")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES,
                        help="figure directory, not tracked "
                             "(default: ./images/local/sky)")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS,
                        help="report directory, not tracked "
                             "(default: ./reports)")
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
    light = [
        "  BRIGHTNESS",
        f"  {'v_s/c':>6} | {'star ahead':>10} | {'starlight':>9} | "
        f"{'unlensed':>9} | {'lensing':>7}",
        "  " + "-" * 54,
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
        received = sky.sky_brightness()
        unlensed = unlensed_brightness(ratio)
        light.append(
            f"  {ratio:>6g} | {float(sky.flux_ratio(0.0)):>9.4g}x | "
            f"{received:>8.4g}x | {unlensed:>8.4g}x | "
            f"{100.0 * (received / unlensed - 1.0):>+6.1f}%"
        )
    lines.append("  ahead: E_ship / E_far for a star straight ahead, "
                 "1 + v_s/c")
    lines.append("  sky hidden: solid-angle fraction beyond arccos(-c/v_s), "
                 "(1 - c/v_s) / 2")
    light.append("  star ahead: flux R^4 mu of a star straight ahead")
    light.append("  starlight: light received from an isotropic background")
    light.append("  unlensed: the same if the stars stayed in place, "
                 "((1+u)^5 - max(0,1-u)^5) / 10u")
    return "\n".join(lines) + "\n\n" + "\n".join(light)


def format_offcentre_table(broeck, offsets):
    """The two windows out of the pocket, from each offset."""

    throat = throat_radius(broeck, broeck.inner_radius)
    centre = float(broeck.conformal_factor(0.0, 0.0, 0.0))
    lines = [
        f"  AWAY FROM THE CENTRE   (pocket {broeck.pocket_proper_radius():g} "
        f"m, throat {throat:.3f} m)",
        f"  {'offset':>8} | {'window half-angle':>17} | {'sky open':>9}",
        "  " + "-" * 42,
    ]
    for offset in offsets:
        cone = visible_cone(broeck, offset / centre)
        open_fraction = 1.0 - math.cos(cone)
        lines.append(f"  {offset:>6g} m | {math.degrees(cone):>15.2f} ° | "
                     f"{100.0 * open_fraction:>7.2f} %")
    lines.append("  window: sin(psi_c) = R_throat / offset, two windows")
    lines.append("  sky open: fraction of lines of sight that leave the "
                 "pocket, 1 - cos(psi_c)")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

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
        for path in (
            plot_sky(bubbles, os.path.join(args.figures_dir,
                                           "01_fisheye.png")),
            plot_sky_mapping(bubbles, os.path.join(args.figures_dir,
                                                   "02_mapping.png")),
            plot_offcentre_sky(broeck, args.offsets,
                               os.path.join(args.figures_dir,
                                            "03_offcentre.png")),
        ):
            print("   ", os.path.relpath(path))

    text = format_sky_table(bubbles) + (
        f"\n\n  Van Den Broeck with the same wall at "
        f"{fastest.speed / C_LIGHT:g}c: largest difference in source angle "
        f"{difference:.1e} rad\n\n"
        + format_offcentre_table(broeck, args.offsets) + "\n"
    )
    print("\n" + text)
    report_path = os.path.join(args.reports_dir, "sky.txt")
    with open(report_path, "w") as handle:
        handle.write(text)

    print(f"figures written to {os.path.relpath(args.figures_dir)}")
    print(f"report written to {os.path.relpath(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
