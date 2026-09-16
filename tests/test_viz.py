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

from warpdrive import AlcubierreMetric, BroeckMetric        # noqa: E402
from warpdrive.constants import C_LIGHT                     # noqa: E402
from warpdrive.viz import plot_all                          # noqa: E402
from warpdrive.viz.figures import energy_norm, energy_title  # noqa: E402


def test_energy_norm_is_symmetric():
    norm = energy_norm(np.array([-5.0, 0.0, 2.0]))

    assert norm.vmin == -5.0
    assert norm.vmax == 5.0
    assert norm(0.0) == pytest.approx(0.5)
    assert norm(-1.0) == pytest.approx(1.0 - norm(1.0))


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
