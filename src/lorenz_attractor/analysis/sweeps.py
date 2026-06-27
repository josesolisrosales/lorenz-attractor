"""Parameter sweeps, bifurcation, and sensitivity analysis.

These operate over a Simulator instance so they remain decoupled from any
specific system and ready to generalize to other dynamical systems later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.parameters import InitialConditions, LorenzParameters, SimulationConfig

if TYPE_CHECKING:
    from ..core.simulator import SimulationResult, Simulator


def parameter_sweep(
    simulator: 'Simulator',
    parameter_name: str,
    parameter_values: np.ndarray,
    initial_conditions: InitialConditions,
    config: SimulationConfig,
) -> 'List[SimulationResult]':
    """Run one simulation per parameter value; restore original params after."""
    original_params = simulator.system.parameters
    results: List['SimulationResult'] = []
    for param_value in parameter_values:
        params_dict = original_params.to_dict()
        params_dict[parameter_name] = param_value
        simulator.system.update_parameters(LorenzParameters.from_dict(params_dict))
        results.append(simulator.simulate(initial_conditions, config))
    simulator.system.update_parameters(original_params)
    return results


def bifurcation_analysis(
    simulator: 'Simulator',
    parameter_name: str,
    parameter_range: Tuple[float, float],
    num_points: int = 100,
    initial_conditions: Optional[InitialConditions] = None,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, Any]:
    """Sweep a parameter and collect the post-transient attractor of each run."""
    if initial_conditions is None:
        initial_conditions = InitialConditions.random()
    if config is None:
        config = SimulationConfig(num_steps=20000)

    param_min, param_max = parameter_range
    param_values = np.linspace(param_min, param_max, num_points)

    results = parameter_sweep(
        simulator, parameter_name, param_values, initial_conditions, config
    )

    attractors = [result.trajectory[-1000:] for result in results]
    return {
        'parameter_values': param_values,
        'attractors': attractors,
        'parameter_name': parameter_name,
    }


def sensitivity_analysis(
    simulator: 'Simulator',
    initial_conditions: InitialConditions,
    perturbation: float = 1e-10,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, Any]:
    """Quantify divergence from perturbing each initial coordinate."""
    if config is None:
        config = SimulationConfig()

    original_result = simulator.simulate(initial_conditions, config)
    perturbed_ics = [
        InitialConditions(
            initial_conditions.x + perturbation,
            initial_conditions.y,
            initial_conditions.z,
        ),
        InitialConditions(
            initial_conditions.x,
            initial_conditions.y + perturbation,
            initial_conditions.z,
        ),
        InitialConditions(
            initial_conditions.x,
            initial_conditions.y,
            initial_conditions.z + perturbation,
        ),
    ]
    perturbed_results = simulator.simulate_multiple(perturbed_ics, config)

    divergences = [
        np.linalg.norm(r.trajectory - original_result.trajectory, axis=1)
        for r in perturbed_results
    ]
    return {
        'original_trajectory': original_result.trajectory,
        'perturbed_trajectories': [r.trajectory for r in perturbed_results],
        'divergences': divergences,
        'time': original_result.time,
        'perturbation': perturbation,
    }
