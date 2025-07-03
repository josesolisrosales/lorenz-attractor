#!/usr/bin/env python3
"""
Advanced features demonstration for the Lorenz Attractor Professional Suite.

This script showcases the sophisticated capabilities including parameter sweeps,
bifurcation analysis, video export, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lorenz_attractor import (
    LorenzSystem, Simulator, LorenzPlotter, VideoExporter, DataExporter,
    LorenzParameters, InitialConditions, SimulationConfig
)


def example_parameter_sweep():
    """Demonstrate parameter sweep analysis."""
    print("Advanced Example 1: Parameter Sweep")
    print("-" * 50)
    
    # Create simulator
    system = LorenzSystem()
    simulator = Simulator(system)
    
    # Define sweep parameters
    rho_values = np.linspace(20, 30, 20)
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=8000, integration_method='rk4')
    
    print(f"Running parameter sweep: rho from {rho_values[0]} to {rho_values[-1]}")
    print(f"Number of simulations: {len(rho_values)}")
    
    # Run parameter sweep
    results = simulator.parameter_sweep('rho', rho_values, initial_conditions, config)
    
    # Create visualization
    plotter = LorenzPlotter(style='scientific')
    fig = plotter.plot_parameter_sweep(results, 'rho', rho_values)
    
    # Save plot
    output_file = "parameter_sweep_rho.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Parameter sweep plot saved to {output_file}")
    
    plt.show()
    
    return results, rho_values


def example_bifurcation_analysis():
    """Demonstrate bifurcation analysis."""
    print("\nAdvanced Example 2: Bifurcation Analysis")
    print("-" * 50)
    
    # Create simulator
    system = LorenzSystem()
    simulator = Simulator(system)
    
    # Run bifurcation analysis
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=15000)
    
    print("Running bifurcation analysis for rho parameter...")
    bifurcation_data = simulator.bifurcation_analysis(
        'rho', (0.5, 50), num_points=200, 
        initial_conditions=initial_conditions, config=config
    )
    
    # Create bifurcation diagram
    plotter = LorenzPlotter(style='dark')
    fig = plotter.plot_bifurcation_diagram(bifurcation_data)
    
    # Save plot
    output_file = "bifurcation_diagram.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bifurcation diagram saved to {output_file}")
    
    plt.show()
    
    return bifurcation_data


def example_video_export():
    """Demonstrate video export capabilities."""
    print("\nAdvanced Example 3: Video Export")
    print("-" * 50)
    
    # Create high-quality simulation for video
    system = LorenzSystem()
    simulator = Simulator(system)
    
    initial_conditions = InitialConditions.true_random(scale=2.0)
    config = SimulationConfig(dt=0.005, num_steps=15000, integration_method='rk4')
    
    print("Running high-resolution simulation for video...")
    result = simulator.simulate(initial_conditions, config)
    
    # Create video exporter
    video_exporter = VideoExporter(fps=30, dpi=150)
    
    # Export trajectory animation
    print("Exporting trajectory animation...")
    video_file = "lorenz_trajectory.mp4"
    video_exporter.export_trajectory_animation(
        result, video_file, trail_length=800, quality='high'
    )
    print(f"Trajectory video saved to {video_file}")
    
    # Export rotating view
    print("Exporting rotating view...")
    rotating_file = "lorenz_rotating.mp4"
    video_exporter.export_rotating_view_video(
        result, rotating_file, rotation_speed=2.0, quality='medium'
    )
    print(f"Rotating view video saved to {rotating_file}")
    
    return result


def example_multiple_trajectories():
    """Demonstrate multiple trajectory comparison."""
    print("\nAdvanced Example 4: Multiple Trajectory Comparison")
    print("-" * 50)
    
    # Create system
    system = LorenzSystem()
    simulator = Simulator(system)
    
    # Generate multiple initial conditions
    base_ic = InitialConditions(x=1.0, y=1.0, z=1.0)
    perturbations = [0, 1e-10, 1e-8, 1e-6, 1e-4]
    
    initial_conditions = []
    for pert in perturbations:
        ic = InitialConditions(
            x=base_ic.x + pert,
            y=base_ic.y + pert,
            z=base_ic.z + pert
        )
        initial_conditions.append(ic)
    
    # Run simulations
    config = SimulationConfig(dt=0.01, num_steps=12000)
    results = simulator.simulate_multiple(initial_conditions, config, parallel=True)
    
    # Create comparison plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (result, color, pert) in enumerate(zip(results, colors, perturbations)):
        alpha = 0.8 if i == 0 else 0.6
        linewidth = 1.0 if i == 0 else 0.7
        label = f'Original' if i == 0 else f'Pert: {pert:.0e}'
        
        ax.plot(result.x, result.y, result.z, color=color, linewidth=linewidth,
               alpha=alpha, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Butterfly Effect: Sensitivity to Initial Conditions')
    ax.legend()
    
    output_file = "butterfly_effect.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Butterfly effect visualization saved to {output_file}")
    
    plt.show()
    
    # Export as video
    video_exporter = VideoExporter()
    labels = [f'Pert: {p:.0e}' if p > 0 else 'Original' for p in perturbations]
    
    print("Exporting multi-trajectory video...")
    video_file = "butterfly_effect.mp4"
    video_exporter.export_multi_trajectory_video(results, video_file, labels=labels)
    print(f"Multi-trajectory video saved to {video_file}")
    
    return results


def example_lyapunov_analysis():
    """Demonstrate Lyapunov exponent calculation."""
    print("\nAdvanced Example 5: Lyapunov Exponent Analysis")
    print("-" * 50)
    
    # Test different parameter values
    rho_values = [20, 24, 28, 32, 40]
    lyapunov_exponents = []
    
    for rho in rho_values:
        print(f"Calculating Lyapunov exponent for rho = {rho}...")
        
        # Create system
        params = LorenzParameters(sigma=10, rho=rho, beta=8/3)
        system = LorenzSystem(params)
        
        # Calculate Lyapunov exponent
        initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
        lyap = system.lyapunov_exponents(initial_conditions, dt=0.01, num_steps=50000)
        lyapunov_exponents.append(lyap[0])  # Largest exponent
        
        print(f"  Largest Lyapunov exponent: {lyap[0]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, lyapunov_exponents, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Chaos threshold')
    plt.xlabel('Rho Parameter')
    plt.ylabel('Largest Lyapunov Exponent')
    plt.title('Lyapunov Exponent vs Rho Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = "lyapunov_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lyapunov analysis plot saved to {output_file}")
    
    plt.show()
    
    return rho_values, lyapunov_exponents


def example_data_export():
    """Demonstrate comprehensive data export."""
    print("\nAdvanced Example 6: Comprehensive Data Export")
    print("-" * 50)
    
    # Create high-quality simulation
    system = LorenzSystem()
    simulator = Simulator(system)
    
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=20000, integration_method='dopri5')
    
    result = simulator.simulate(initial_conditions, config)
    
    # Create data exporter
    data_exporter = DataExporter()
    
    # Export in multiple formats
    print("Exporting data in multiple formats...")
    
    formats = {
        'CSV': 'lorenz_data.csv',
        'JSON': 'lorenz_data.json',
        'HDF5': 'lorenz_data.h5',
        'NumPy': 'lorenz_data.npz',
        'MATLAB': 'lorenz_data.mat'
    }
    
    for format_name, filename in formats.items():
        try:
            if format_name == 'CSV':
                data_exporter.export_csv(result, filename)
            elif format_name == 'JSON':
                data_exporter.export_json(result, filename)
            elif format_name == 'HDF5':
                data_exporter.export_hdf5(result, filename)
            elif format_name == 'NumPy':
                data_exporter.export_numpy(result, filename)
            elif format_name == 'MATLAB':
                data_exporter.export_matlab(result, filename)
            
            print(f"  {format_name} format: {filename}")
        except Exception as e:
            print(f"  {format_name} format: Failed ({e})")
    
    # Create comprehensive data package
    print("\nCreating comprehensive data package...")
    package_path = data_exporter.create_data_package([result], "lorenz_data_package")
    print(f"Data package created at: {package_path}")
    
    return result


def example_custom_analysis():
    """Demonstrate custom analysis workflow."""
    print("\nAdvanced Example 7: Custom Analysis Workflow")
    print("-" * 50)
    
    # Create multiple systems with different parameters
    parameter_sets = [
        LorenzParameters(sigma=10, rho=28, beta=8/3),    # Classical
        LorenzParameters(sigma=10, rho=99.96, beta=8/3), # Hyperchaotic
        LorenzParameters(sigma=10, rho=145, beta=8/3),   # Different regime
    ]
    
    labels = ["Classical", "Hyperchaotic", "High Rho"]
    
    # Run comprehensive analysis
    initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
    config = SimulationConfig(dt=0.01, num_steps=25000, integration_method='rk4')
    
    all_results = []
    
    for params, label in zip(parameter_sets, labels):
        print(f"Analyzing {label} regime...")
        
        system = LorenzSystem(params)
        simulator = Simulator(system)
        
        # Run simulation
        result = simulator.simulate(initial_conditions, config)
        all_results.append(result)
        
        # Calculate basic statistics
        print(f"  Max Z: {np.max(result.z):.2f}")
        print(f"  Min Z: {np.min(result.z):.2f}")
        print(f"  Z Range: {np.ptp(result.z):.2f}")
        
        # Calculate Poincaré section
        poincare = result.poincare_section()
        print(f"  Poincaré points: {len(poincare)}")
    
    # Create comprehensive comparison
    plotter = LorenzPlotter(style='scientific')
    
    # 3D comparison
    fig = plt.figure(figsize=(18, 6))
    
    for i, (result, label) in enumerate(zip(all_results, labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(result.time)))
        for j in range(len(result.trajectory) - 1):
            ax.plot(result.trajectory[j:j+2, 0], 
                   result.trajectory[j:j+2, 1], 
                   result.trajectory[j:j+2, 2], 
                   color=colors[j], linewidth=0.3, alpha=0.7)
        
        ax.set_title(f'{label}\nρ={result.parameters.rho}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    output_file = "regime_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Regime comparison saved to {output_file}")
    
    plt.show()
    
    return all_results


def main():
    """Run all advanced examples."""
    print("Lorenz Attractor Professional Suite - Advanced Features")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("advanced_examples_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run advanced examples
        print("Running advanced feature demonstrations...")
        
        # Parameter sweep
        sweep_results, rho_values = example_parameter_sweep()
        
        # Bifurcation analysis  
        bifurcation_data = example_bifurcation_analysis()
        
        # Video export (may take some time)
        response = input("\nRun video export examples? This may take several minutes (y/n): ")
        if response.lower().startswith('y'):
            video_result = example_video_export()
        
        # Multiple trajectories
        multi_results = example_multiple_trajectories()
        
        # Lyapunov analysis
        lyap_rho, lyap_values = example_lyapunov_analysis()
        
        # Data export
        export_result = example_data_export()
        
        # Custom analysis
        custom_results = example_custom_analysis()
        
        print("\n" + "=" * 70)
        print("All advanced examples completed successfully!")
        print("\nGenerated files:")
        print("- parameter_sweep_rho.png")
        print("- bifurcation_diagram.png")
        print("- butterfly_effect.png")
        print("- lyapunov_analysis.png")
        print("- regime_comparison.png")
        print("- Various data export files")
        if 'video_result' in locals():
            print("- lorenz_trajectory.mp4")
            print("- lorenz_rotating.mp4")
            print("- butterfly_effect.mp4")
        
        print("\nExplore the CLI for more advanced features:")
        print("  lorenz-sim --help")
        print("  lorenz-sim web  # Launch web interface")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError in advanced examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()