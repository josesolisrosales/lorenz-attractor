"""Tests for analysis.sections (extracted from core.lorenz)."""

import numpy as np

from lorenz_attractor.analysis import sections
from lorenz_attractor.core.lorenz import LorenzSystem


def test_no_crossing_returns_empty_with_correct_shape():
    traj = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    out = sections.poincare_section(traj)
    assert out.shape == (0, 3)


def test_two_crossings_detected_at_plane():
    traj = np.array([[0, 0, 26], [1, 1, 28], [2, 2, 26]], dtype=float)
    out = sections.poincare_section(traj, plane_offset=27.0)
    assert len(out) == 2
    assert abs(out[0][2] - 27.0) < 1e-10


def test_lorenzsystem_method_delegates_identically():
    system = LorenzSystem()
    traj = np.array([[0, 0, 26], [1, 1, 28], [2, 2, 26]], dtype=float)
    np.testing.assert_array_equal(
        system.poincare_section(traj), sections.poincare_section(traj)
    )
