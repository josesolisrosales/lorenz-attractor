"""Construction smoke tests for realtime visualizers with GL/pygame mocked."""

import sys
from unittest import mock

import pytest

from lorenz_attractor.core.simulator import Simulator


@pytest.fixture(autouse=True)
def _mock_graphics_libs():
    mods = {
        name: mock.MagicMock()
        for name in ('pygame', 'OpenGL', 'OpenGL.GL', 'OpenGL.GLU', 'moderngl')
    }
    with mock.patch.dict(sys.modules, mods):
        for cached in ('lorenz_attractor.visualization.realtime',):
            sys.modules.pop(cached, None)
        yield


def test_realtime_visualizer_constructs():
    from lorenz_attractor.visualization.realtime import RealtimeVisualizer

    viz = RealtimeVisualizer(Simulator(), trail_length=100)
    assert isinstance(viz, RealtimeVisualizer)
    assert viz.trail_length == 100


def test_streaming_visualizer_constructs():
    from lorenz_attractor.visualization.realtime import StreamingVisualizer

    viz = StreamingVisualizer(Simulator(), buffer_size=500)
    assert isinstance(viz, StreamingVisualizer)
    assert viz.buffer_size == 500
