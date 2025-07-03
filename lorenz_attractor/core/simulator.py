"""Advanced simulator for Lorenz attractor with multiple integration methods."""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .lorenz import LorenzSystem
from .parameters import LorenzParameters, InitialConditions, SimulationConfig
from ..integration.integrators import (
    EulerIntegrator, 
    RungeKutta4Integrator,
    AdaptiveIntegrator,
    DormandPrince54Integrator
)


@dataclass
class SimulationResult:
    """Container for simulation results."""
    
    time: np.ndarray
    trajectory: np.ndarray
    parameters: LorenzParameters
    initial_conditions: InitialConditions
    config: SimulationConfig
    metadata: Dict[str, Any]
    
    @property
    def x(self) -> np.ndarray:
        """X coordinates."""
        return self.trajectory[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Y coordinates."""
        return self.trajectory[:, 1]
    
    @property
    def z(self) -> np.ndarray:
        """Z coordinates."""
        return self.trajectory[:, 2]
    
    def poincare_section(self, plane_normal: np.ndarray = None, 
                        plane_offset: float = 27.0) -> np.ndarray:
        """Compute Poincaré section."""
        if plane_normal is None:
            plane_normal = np.array([0, 0, 1])
        
        system = LorenzSystem(self.parameters)
        return system.poincare_section(self.trajectory, plane_normal, plane_offset)
    
    def save(self, filename: str):
        """Save simulation results to file."""
        np.savez_compressed(
            filename,
            time=self.time,
            trajectory=self.trajectory,
            parameters=self.parameters.to_dict(),
            initial_conditions=self.initial_conditions.to_array(),
            config=self.config.to_dict(),
            metadata=self.metadata
        )
    
    @classmethod
    def load(cls, filename: str) -> 'SimulationResult':
        """Load simulation results from file."""
        data = np.load(filename, allow_pickle=True)
        
        return cls(
            time=data['time'],
            trajectory=data['trajectory'],
            parameters=LorenzParameters.from_dict(data['parameters'].item()),
            initial_conditions=InitialConditions.from_array(data['initial_conditions']),
            config=SimulationConfig.from_dict(data['config'].item()),
            metadata=data['metadata'].item()
        )


class Simulator:
    """Advanced simulator for the Lorenz attractor system."""
    
    def __init__(self, system: Optional[LorenzSystem] = None):
        """
        Initialize the simulator.
        
        Args:
            system: Lorenz system to simulate. If None, uses default parameters.
        """
        self.system = system or LorenzSystem()
        self._integrator_map = {
            'euler': EulerIntegrator,
            'rk4': RungeKutta4Integrator,
            'adaptive': AdaptiveIntegrator,
            'dopri5': DormandPrince54Integrator
        }
    
    def simulate(self, initial_conditions: InitialConditions,
                config: SimulationConfig) -> SimulationResult:
        """
        Run a single simulation.
        
        Args:
            initial_conditions: Initial conditions for the simulation
            config: Simulation configuration
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        
        # Create integrator
        integrator_class = self._integrator_map[config.integration_method]
        integrator = integrator_class(config.dt)
        
        # Define system function for integrator
        def system_func(y, t):
            return self.system.derivative(y)
        
        # Integrate
        t_span = (0, config.total_time)
        time_array, trajectory = integrator.integrate(
            system_func, 
            initial_conditions.to_array(), 
            t_span, 
            config.num_steps
        )
        
        # Apply save interval
        if config.save_interval > 1:
            indices = np.arange(0, len(time_array), config.save_interval)
            time_array = time_array[indices]
            trajectory = trajectory[indices]
        
        # Create metadata
        metadata = {
            'simulation_time': time.time() - start_time,
            'integrator': config.integration_method,
            'effective_dt': config.dt,
            'total_points': len(time_array),
            'lyapunov_time': config.total_time / np.log(2)  # Approximate
        }
        
        return SimulationResult(
            time=time_array,
            trajectory=trajectory,
            parameters=self.system.parameters,
            initial_conditions=initial_conditions,
            config=config,
            metadata=metadata
        )
    
    def simulate_multiple(self, initial_conditions_list: List[InitialConditions],
                         config: SimulationConfig, 
                         parallel: bool = True) -> List[SimulationResult]:
        """
        Run multiple simulations with different initial conditions.
        
        Args:
            initial_conditions_list: List of initial conditions
            config: Simulation configuration
            parallel: Whether to run simulations in parallel
            
        Returns:
            List of simulation results
        """
        if not parallel:
            return [self.simulate(ic, config) for ic in initial_conditions_list]
        
        # Parallel execution
        results = []
        with ThreadPoolExecutor(max_workers=None) as executor:
            future_to_ic = {
                executor.submit(self.simulate, ic, config): ic 
                for ic in initial_conditions_list
            }
            
            for future in as_completed(future_to_ic):
                results.append(future.result())
        
        return results
    
    def parameter_sweep(self, parameter_name: str, 
                       parameter_values: np.ndarray,
                       initial_conditions: InitialConditions,
                       config: SimulationConfig) -> List[SimulationResult]:
        """
        Perform a parameter sweep.
        
        Args:
            parameter_name: Name of parameter to sweep ('sigma', 'rho', or 'beta')
            parameter_values: Array of parameter values
            initial_conditions: Initial conditions
            config: Simulation configuration
            
        Returns:
            List of simulation results for each parameter value
        """
        original_params = self.system.parameters
        results = []
        
        for param_value in parameter_values:
            # Create new parameters
            params_dict = original_params.to_dict()
            params_dict[parameter_name] = param_value
            new_params = LorenzParameters.from_dict(params_dict)
            
            # Update system
            self.system.update_parameters(new_params)
            
            # Simulate
            result = self.simulate(initial_conditions, config)
            results.append(result)
        
        # Restore original parameters
        self.system.update_parameters(original_params)
        
        return results
    
    def bifurcation_analysis(self, parameter_name: str,
                           parameter_range: Tuple[float, float],
                           num_points: int = 100,
                           initial_conditions: InitialConditions = None,
                           config: SimulationConfig = None) -> Dict[str, np.ndarray]:
        """
        Perform bifurcation analysis.
        
        Args:
            parameter_name: Parameter to vary
            parameter_range: (min_value, max_value)
            num_points: Number of parameter values to test
            initial_conditions: Initial conditions (default: random)
            config: Simulation configuration
            
        Returns:
            Dictionary with parameter values and corresponding attractors
        """
        if initial_conditions is None:
            initial_conditions = InitialConditions.random()
        
        if config is None:
            config = SimulationConfig(num_steps=20000)
        
        # Create parameter array
        param_min, param_max = parameter_range
        param_values = np.linspace(param_min, param_max, num_points)
        
        # Run parameter sweep
        results = self.parameter_sweep(parameter_name, param_values, 
                                     initial_conditions, config)
        
        # Extract attractor points (last 1000 points of each simulation)
        attractors = []
        for result in results:
            # Take last points after transient
            attractor_points = result.trajectory[-1000:]
            attractors.append(attractor_points)
        
        return {
            'parameter_values': param_values,
            'attractors': attractors,
            'parameter_name': parameter_name
        }
    
    def sensitivity_analysis(self, initial_conditions: InitialConditions,
                           perturbation: float = 1e-10,
                           config: SimulationConfig = None) -> Dict[str, Any]:
        """
        Analyze sensitivity to initial conditions.
        
        Args:
            initial_conditions: Base initial conditions
            perturbation: Size of perturbation
            config: Simulation configuration
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if config is None:
            config = SimulationConfig()
        
        # Original trajectory
        original_result = self.simulate(initial_conditions, config)
        
        # Perturbed trajectories
        perturbed_ics = [
            InitialConditions(
                initial_conditions.x + perturbation, 
                initial_conditions.y, 
                initial_conditions.z
            ),
            InitialConditions(
                initial_conditions.x, 
                initial_conditions.y + perturbation, 
                initial_conditions.z
            ),
            InitialConditions(
                initial_conditions.x, 
                initial_conditions.y, 
                initial_conditions.z + perturbation
            )
        ]
        
        perturbed_results = self.simulate_multiple(perturbed_ics, config)
        
        # Calculate divergence
        divergences = []
        for result in perturbed_results:
            diff = np.linalg.norm(result.trajectory - original_result.trajectory, axis=1)
            divergences.append(diff)
        
        return {
            'original_trajectory': original_result.trajectory,
            'perturbed_trajectories': [r.trajectory for r in perturbed_results],
            'divergences': divergences,
            'time': original_result.time,
            'perturbation': perturbation
        }
    
    def estimate_lyapunov_exponent(self, initial_conditions: InitialConditions,
                                  config: SimulationConfig = None) -> float:
        """
        Estimate the largest Lyapunov exponent.
        
        Args:
            initial_conditions: Initial conditions
            config: Simulation configuration
            
        Returns:
            Estimate of largest Lyapunov exponent
        """
        if config is None:
            config = SimulationConfig(num_steps=50000)
        
        # Use system's built-in method
        return self.system.lyapunov_exponents(initial_conditions, 
                                            config.dt, 
                                            config.num_steps)[0]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Simulator(system={self.system})"