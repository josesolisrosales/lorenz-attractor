"""Headless smoke tests for plotting and image export."""

import matplotlib

matplotlib.use('Agg')  # must precede pyplot import anywhere downstream

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from lorenz_attractor.core.parameters import InitialConditions, SimulationConfig  # noqa: E402
from lorenz_attractor.core.simulator import Simulator  # noqa: E402
from lorenz_attractor.export.image import ImageExporter  # noqa: E402
from lorenz_attractor.visualization.plotter import LorenzPlotter  # noqa: E402


@pytest.fixture
def result():
    return Simulator().simulate(
        InitialConditions(1.0, 1.0, 1.0), SimulationConfig(dt=0.01, num_steps=200)
    )


def test_plotter_constructs():
    assert LorenzPlotter(style='dark') is not None


def test_plot_3d_trajectory_builds_a_figure(result):
    fig = LorenzPlotter().plot_3d_trajectory(result)
    assert fig is not None
    plt.close(fig)


def test_image_exporter_writes_files(result, tmp_path):
    exporter = ImageExporter(dpi=72)
    p3d = tmp_path / 'traj3d.png'
    p2d = tmp_path / 'proj2d.png'
    exporter.export_3d_plot(result, str(p3d))
    exporter.export_2d_projections(result, str(p2d))
    assert p3d.exists() and p3d.stat().st_size > 0
    assert p2d.exists() and p2d.stat().st_size > 0
