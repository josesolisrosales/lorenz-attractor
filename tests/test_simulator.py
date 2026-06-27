"""Tests for the Simulator orchestration layer."""

import pytest

from lorenz_attractor.core.parameters import InitialConditions, SimulationConfig
from lorenz_attractor.core.simulator import Simulator


@pytest.mark.parametrize('method', ['euler', 'rk4', 'adaptive', 'dopri5'])
def test_all_advertised_methods_run_end_to_end(method):
    sim = Simulator()
    ic = InitialConditions(1.0, 1.0, 1.0)
    config = SimulationConfig(dt=0.01, num_steps=50, integration_method=method)
    result = sim.simulate(ic, config)
    assert result.trajectory.shape[1] == 3
    assert result.trajectory.shape[0] > 0
