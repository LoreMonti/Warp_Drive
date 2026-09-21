# ==========================================================
# Tests for the figures
#
# Smoke tests that every figure renders for both metrics, plus the two
# helpers that decide what an energy density map says about its sign.
#
# Author: Lorenzo Monti
# ==========================================================

import matplotlib

matplotlib.use("Agg")

import numpy as np                                          # noqa: E402
import pytest                                               # noqa: E402

from warpdrive import (                                     # noqa: E402
    AlcubierreMetric,
    BroeckMetric,
    neck_scaling,
)
from warpdrive.constants import C_LIGHT                     # noqa: E402
from warpdrive.viz import (                                 # noqa: E402
    plot_all,
    plot_energy_floor,
    plot_metric_comparison,
    plot_neck_scaling,
    plot_pocket_throat,
)
from warpdrive.viz.sky import (                             # noqa: E402
    plot_sky,
    checkerboard,
    fisheye_directions,
    plot_offcentre_sky,
    plot_sky_mapping,
    star_field,
    star_sizes,
)
from warpdrive.viz.figures import (                         # noqa: E402
    energy_norm,
    energy_ticks,
    energy_title,
)


def test_energy_norm_is_symmetric():
    norm = energy_norm(np.array([-5.0, 0.0, 2.0]))

    assert norm.vmin == -5.0
    assert norm.vmax == 5.0
    assert norm(0.0) == pytest.approx(0.5)
    assert norm(-1.0) == pytest.approx(1.0 - norm(1.0))


def test_energy_ticks_skip_the_linear_band():
    norm = energy_norm(np.array([-3.0e43, 2.0e42]), decades=4)
    ticks = energy_ticks(norm)

    assert 0.0 in ticks
    assert ticks == sorted(ticks)
    assert [t for t in ticks if t > 0] == [-t for t in reversed(ticks)
                                          if t < 0]
    assert min(t for t in ticks if t > 0) > norm.linthresh
    assert max(ticks) <= norm.vmax


def test_energy_title_follows_the_sign_of_the_data():
    assert "negative energy density everywhere" in energy_title(
        np.array([-1.0, 0.0]))
    assert "positive" in energy_title(np.array([-1.0, 1.0e-6]))


@pytest.mark.parametrize("metric", [
    AlcubierreMetric(speed=10.0 * C_LIGHT),
    BroeckMetric(speed=10.0 * C_LIGHT),
], ids=["alcubierre", "broeck"])
def test_static_figures_render(metric, tmp_path):
    paths = plot_all(metric, str(tmp_path))

    assert len(paths) == 4
    for path in paths:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 10_000


def test_comparison_figures_render(tmp_path):
    broeck = BroeckMetric(speed=10.0 * C_LIGHT)
    alcubierre = AlcubierreMetric(speed=broeck.speed, radius=broeck.radius,
                                  sigma=broeck.sigma)

    paths = [
        plot_metric_comparison(alcubierre, broeck,
                               str(tmp_path / "comparison.png")),
        plot_neck_scaling(neck_scaling(np.logspace(-14.5, 2.0, 6)),
                          str(tmp_path / "neck.png")),
    ]
    for path in paths:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 10_000


def test_star_field_is_uniform_on_the_sphere():
    polar, azimuth, brightness = star_field(n_stars=20000, seed=1)

    assert abs(np.cos(polar).mean()) < 0.02
    assert brightness.max() == 1.0 and brightness.min() > 0.0


def test_sky_figures_render(tmp_path):
    bubbles = [AlcubierreMetric(speed=0.5 * C_LIGHT, radius=100.0, sigma=1.0),
               AlcubierreMetric(speed=10.0 * C_LIGHT, radius=100.0,
                                sigma=1.0)]
    paths = [
        plot_sky(bubbles, str(tmp_path / "sky.png"), n_stars=300,
                 n_rays=181),
        plot_sky_mapping(bubbles, str(tmp_path / "mapping.png"), n_rays=181),
    ]
    for path in paths:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 10_000


def test_star_sizes_grow_with_flux_and_stay_bounded():
    flux = np.array([1e-8, 1e-2, 1.0, 1e2, 1e4, 1e8])
    sizes = star_sizes(np.ones_like(flux), flux)

    assert np.all(np.diff(sizes) >= 0.0)
    assert sizes[0] == 0.2 and sizes[-1] == 14.0
    assert sizes[2] == 4.0


def test_fisheye_puts_ahead_at_the_centre_and_behind_on_the_rim():
    directions, inside = fisheye_directions(101)
    grid = np.full(inside.shape + (3,), np.nan)
    grid[inside] = directions.T

    assert np.allclose(grid[50, 50], [1.0, 0.0, 0.0])
    assert np.allclose(grid[50, 100], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(grid[50, 75], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(np.linalg.norm(directions, axis=0), 1.0)


def test_checkerboard_alternates_across_cells():
    polar = np.radians([7.5, 22.5, 7.5])
    azimuth = np.radians([7.5, 7.5, 22.5])
    source = np.array([np.cos(polar), np.sin(polar) * np.cos(azimuth),
                       np.sin(polar) * np.sin(azimuth)])

    assert list(checkerboard(source)) == [1.0, 0.45, 0.45]


def test_offcentre_figure_renders(tmp_path):
    path = plot_offcentre_sky(BroeckMetric(speed=10.0 * C_LIGHT),
                              [0.0, 60.0], str(tmp_path / "offcentre.png"),
                              resolution=61, window_resolution=31)
    assert (tmp_path / "offcentre.png").stat().st_size > 10_000


def test_throat_figure_renders(tmp_path):
    path = plot_pocket_throat(BroeckMetric(speed=C_LIGHT),
                              str(tmp_path / "throat.png"), n_points=2001)
    assert (tmp_path / "throat.png").stat().st_size > 10_000


def test_energy_floor_figure_renders(tmp_path):
    plot_energy_floor(str(tmp_path / "floor.png"), n_inner=6)
    assert (tmp_path / "floor.png").stat().st_size > 10_000
