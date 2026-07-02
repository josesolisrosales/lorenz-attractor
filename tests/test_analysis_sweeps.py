"""Tests for analysis.sweeps (extracted from core.simulator)."""

import numpy as np

from lorenz_attractor.analysis import sweeps
from lorenz_attractor.core.parameters import InitialConditions, SimulationConfig
from lorenz_attractor.core.simulator import Simulator


def _small_config():
    return SimulationConfig(dt=0.01, num_steps=100)


def test_parameter_sweep_runs_one_sim_per_value_and_restores_params():
    sim = Simulator()
    original_rho = sim.system.parameters.rho
    values = np.array([20.0, 24.0, 28.0])
    results = sweeps.parameter_sweep(
        sim, 'rho', values, InitialConditions(1.0, 1.0, 1.0), _small_config()
    )
    assert len(results) == 3
    # parameters restored after the sweep
    assert sim.system.parameters.rho == original_rho
    # each result used the corresponding rho
    assert [r.parameters.rho for r in results] == [20.0, 24.0, 28.0]


def test_bifurcation_analysis_returns_expected_keys():
    sim = Simulator()
    out = sweeps.bifurcation_analysis(
        sim,
        'rho',
        (20.0, 28.0),
        num_points=3,
        initial_conditions=InitialConditions(1.0, 1.0, 1.0),
        config=_small_config(),
    )
    assert set(out) == {'parameter_values', 'attractors', 'parameter_name'}
    assert len(out['parameter_values']) == 3
    assert len(out['attractors']) == 3


def test_sensitivity_analysis_returns_three_divergences():
    sim = Simulator()
    out = sweeps.sensitivity_analysis(
        sim,
        InitialConditions(1.0, 1.0, 1.0),
        perturbation=1e-8,
        config=_small_config(),
    )
    assert set(out) >= {'original_trajectory', 'divergences', 'time', 'perturbation'}
    assert len(out['divergences']) == 3


def test_simulator_methods_delegate():
    sim = Simulator()
    values = np.array([24.0, 28.0])
    via_method = sim.parameter_sweep(
        'rho', values, InitialConditions(1.0, 1.0, 1.0), _small_config()
    )
    assert len(via_method) == 2
