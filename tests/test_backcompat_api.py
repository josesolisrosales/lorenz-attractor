"""Lock the public API: old call sites keep working after the analysis extraction."""

import numpy as np

import lorenz_attractor as la
from lorenz_attractor.analysis import (
    bifurcation_analysis,
    equilibrium_points,
    jacobian,
    lyapunov_exponents,
    parameter_sweep,
    poincare_section,
    sensitivity_analysis,
)
from lorenz_attractor.core.lorenz import LorenzSystem


def test_top_level_exports_present():
    for name in ("LorenzSystem", "Simulator", "LorenzPlotter", "DataExporter"):
        assert hasattr(la, name)


def test_analysis_namespace_exposes_all_functions():
    for fn in (
        jacobian, equilibrium_points, lyapunov_exponents, poincare_section,
        parameter_sweep, bifurcation_analysis, sensitivity_analysis,
    ):
        assert callable(fn)


def test_legacy_methods_still_work():
    system = LorenzSystem()
    state = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(system.jacobian(state), jacobian(system, state))
    assert len(system.equilibrium_points()) == 3
