#!/usr/bin/env python3
"""
Basic usage examples for the Lorenz Attractor Professional Suite.

This script demonstrates the fundamental capabilities of the simulation package.
"""

import numpy as np
import matplotlib.pyplot as plt

from lorenz_attractor import (
    LorenzSystem, Simulator, LorenzPlotter, 
    LorenzParameters, InitialConditions, SimulationConfig
)


def example_basic_simulation():
    """Example 1: Basic simulation with default parameters."""
    print("Example 1: Basic Simulation")
    print("-" * 40)
    
    # Create Lorenz system with classical parameters
    system = LorenzSystem()
    simulator = Simulator(system)
    
    # Set up simulation
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=10000)
    
    # Run simulation
    result = simulator.simulate(initial_conditions, config)
    
    print(f"Simulation completed in {result.metadata['simulation_time']:.3f} seconds")
    print(f"Generated {len(result.trajectory)} points")
    
    # Create basic plot
    plotter = LorenzPlotter(style='dark')
    fig = plotter.plot_3d_trajectory(result, title="Basic Lorenz Attractor")
    plt.show()


def example_parameter_exploration():
    """Example 2: Exploring different parameter sets."""
    print("\nExample 2: Parameter Exploration")
    print("-" * 40)
    
    # Define different parameter sets
    parameter_sets = [
        LorenzParameters.classical(),      # Chaotic
        LorenzParameters.periodic(),       # Periodic
        LorenzParameters(sigma=10, rho=24.5, beta=8/3),  # Transition
    ]
    
    labels = ["Chaotic (ρ=28)", "Periodic (ρ=24)", "Transition (ρ=24.5)"]
    
    # Run simulations
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=15000)
    
    results = []
    for params in parameter_sets:
        system = LorenzSystem(params)
        simulator = Simulator(system)
        result = simulator.simulate(initial_conditions, config)
        results.append(result)
    
    # Create comparison plot
    fig = plt.figure(figsize=(15, 5))
    
    for i, (result, label) in enumerate(zip(results, labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot(result.x, result.y, result.z, linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


def example_sensitivity_analysis():
    """Example 3: Sensitivity to initial conditions."""
    print("\nExample 3: Sensitivity Analysis")
    print("-" * 40)
    
    # Create system and simulator
    system = LorenzSystem()
    simulator = Simulator(system)
    
    # Define initial conditions with small perturbations
    base_ic = InitialConditions(x=1.0, y=1.0, z=1.0)
    perturbation = 1e-8
    
    initial_conditions = [
        base_ic,
        InitialConditions(x=base_ic.x + perturbation, y=base_ic.y, z=base_ic.z),
        InitialConditions(x=base_ic.x, y=base_ic.y + perturbation, z=base_ic.z),
    ]
    
    config = SimulationConfig(dt=0.01, num_steps=8000)
    
    # Run simulations
    results = simulator.simulate_multiple(initial_conditions, config)
    
    # Calculate divergence
    base_trajectory = results[0].trajectory
    divergences = []
    
    for result in results[1:]:
        diff = np.linalg.norm(result.trajectory - base_trajectory, axis=1)
        divergences.append(diff)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 3D trajectories
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    colors = ['blue', 'red', 'green']
    labels = ['Original', 'X perturbed', 'Y perturbed']
    
    for result, color, label in zip(results, colors, labels):
        ax1.plot(result.x, result.y, result.z, color=color, linewidth=0.5, 
                label=label, alpha=0.8)
    
    ax1.set_title('Sensitive Trajectories')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Divergence plot
    ax2 = fig.add_subplot(1, 2, 2)
    time = results[0].time
    
    for i, divergence in enumerate(divergences):
        ax2.semilogy(time, divergence, label=f'{labels[i+1]}')
    
    ax2.set_title('Trajectory Divergence')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distance from Original')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate approximate Lyapunov exponent
    # Use the time when divergence reaches e times initial perturbation
    for i, divergence in enumerate(divergences):
        # Find when divergence exceeds e * perturbation
        exceed_indices = np.where(divergence > np.e * perturbation)[0]
        if len(exceed_indices) > 0:
            lyapunov_time = time[exceed_indices[0]]
            lyapunov_exponent = 1.0 / lyapunov_time
            print(f"Approximate Lyapunov exponent ({labels[i+1]}): {lyapunov_exponent:.3f}")


def example_different_integrators():
    """Example 4: Comparing different integration methods."""
    print("\nExample 4: Integration Methods Comparison")
    print("-" * 40)
    
    # Test different integration methods
    methods = ['euler', 'rk4', 'adaptive', 'dopri5']
    
    # Create system
    system = LorenzSystem()
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    
    results = []
    timings = []
    
    for method in methods:
        config = SimulationConfig(dt=0.01, num_steps=5000, integration_method=method)
        simulator = Simulator(system)
        
        result = simulator.simulate(initial_conditions, config)
        results.append(result)
        timings.append(result.metadata['simulation_time'])
        
        print(f"{method:10}: {result.metadata['simulation_time']:.4f} seconds")
    
    # Create comparison plot
    fig = plt.figure(figsize=(16, 12))
    
    for i, (result, method) in enumerate(zip(results, methods)):
        # 3D plot
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot(result.x, result.y, result.z, linewidth=0.5)
        ax.set_title(f'{method.upper()} Method\n'
                    f'Time: {result.metadata["simulation_time"]:.4f}s')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


def example_advanced_analysis():
    """Example 5: Advanced analysis features."""
    print("\nExample 5: Advanced Analysis")
    print("-" * 40)
    
    # Create system and run simulation
    system = LorenzSystem()
    simulator = Simulator(system)
    
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=20000)
    
    result = simulator.simulate(initial_conditions, config)
    
    # Calculate equilibrium points
    equilibria = system.equilibrium_points()
    print(f"Found {len(equilibria)} equilibrium points:")
    for i, eq in enumerate(equilibria):
        print(f"  Equilibrium {i+1}: ({eq[0]:.3f}, {eq[1]:.3f}, {eq[2]:.3f})")
    
    # Compute Poincaré section
    poincare_points = result.poincare_section()
    print(f"Poincaré section contains {len(poincare_points)} points")
    
    # Create advanced visualization
    plotter = LorenzPlotter(style='scientific')
    fig = plotter.plot_phase_space(result, 
                                  poincare_plane={'normal': np.array([0, 0, 1]), 
                                                'offset': 27.0})
    plt.show()


def example_real_time_visualization():
    """Example 6: Real-time visualization (requires user interaction)."""
    print("\nExample 6: Real-time Visualization")
    print("-" * 40)
    print("This will open a real-time visualization window.")
    print("Close the window or press ESC to continue.")
    
    try:
        from lorenz_attractor.visualization.realtime import RealtimeVisualizer
        
        # Create system and visualizer
        system = LorenzSystem()
        simulator = Simulator(system)
        visualizer = RealtimeVisualizer(simulator, trail_length=1000)
        
        # Setup simulation
        initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
        config = SimulationConfig(dt=0.01, num_steps=100000)
        
        # Start real-time animation (matplotlib version)
        anim = visualizer.start_matplotlib_animation(initial_conditions, config, 
                                                    update_interval=50)
        plt.show()
        
    except ImportError as e:
        print(f"Real-time visualization requires additional dependencies: {e}")
    except Exception as e:
        print(f"Error in real-time visualization: {e}")


def main():
    """Run all examples."""
    print("Lorenz Attractor Professional Suite - Basic Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_simulation()
    example_parameter_exploration()
    example_sensitivity_analysis()
    example_different_integrators()
    example_advanced_analysis()
    
    # Optional real-time example
    response = input("\nWould you like to run the real-time visualization example? (y/n): ")
    if response.lower().startswith('y'):
        example_real_time_visualization()
    
    print("\nAll examples completed!")
    print("Explore the advanced features using the CLI or web interface.")


if __name__ == '__main__':
    main()