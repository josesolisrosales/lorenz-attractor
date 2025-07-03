"""Command-line interface for the Lorenz attractor simulation suite."""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

from .core.lorenz import LorenzSystem
from .core.simulator import Simulator
from .core.parameters import LorenzParameters, InitialConditions, SimulationConfig
from .visualization.plotter import LorenzPlotter
from .visualization.realtime import RealtimeVisualizer
from .export.video import VideoExporter
from .export.data import DataExporter


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Professional Lorenz Attractor Simulation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  lorenz-sim simulate --sigma 10 --rho 28 --beta 2.667

  # High-resolution simulation with video export
  lorenz-sim simulate --steps 50000 --dt 0.005 --export-video output.mp4

  # Parameter sweep
  lorenz-sim sweep --parameter rho --range 20 30 --steps 100

  # Real-time visualization
  lorenz-sim realtime --method pygame

  # Bifurcation analysis
  lorenz-sim bifurcation --parameter rho --range 0 50 --points 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run a single simulation')
    sim_parser.add_argument('--sigma', type=float, default=10.0, help='Sigma parameter')
    sim_parser.add_argument('--rho', type=float, default=28.0, help='Rho parameter')
    sim_parser.add_argument('--beta', type=float, default=8/3, help='Beta parameter')
    sim_parser.add_argument('--x0', type=float, default=1.0, help='Initial X coordinate')
    sim_parser.add_argument('--y0', type=float, default=1.0, help='Initial Y coordinate')
    sim_parser.add_argument('--z0', type=float, default=1.0, help='Initial Z coordinate')
    sim_parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    sim_parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    sim_parser.add_argument('--method', choices=['euler', 'rk4', 'adaptive', 'dopri5'], 
                           default='rk4', help='Integration method')
    sim_parser.add_argument('--output', type=str, help='Output file for results')
    sim_parser.add_argument('--plot', action='store_true', help='Show 3D plot')
    sim_parser.add_argument('--export-video', type=str, help='Export video file')
    sim_parser.add_argument('--export-data', type=str, help='Export data file')
    sim_parser.add_argument('--style', choices=['dark', 'light', 'scientific'], 
                           default='dark', help='Plot style')
    sim_parser.add_argument('--color-by-time', action='store_true',
                           help='Color trajectory by time (slower but prettier)')
    sim_parser.add_argument('--high-quality', action='store_true',
                           help='High quality plots (300 DPI, slower)')
    sim_parser.add_argument('--fast-plot', action='store_true',
                           help='Fast plotting mode (decimation for long trajectories)')
    sim_parser.add_argument('--random-ic', action='store_true', 
                           help='Use random initial conditions')
    sim_parser.add_argument('--true-random', action='store_true',
                           help='Use true random initial conditions')
    
    # Real-time command
    realtime_parser = subparsers.add_parser('realtime', help='Real-time visualization')
    realtime_parser.add_argument('--method', choices=['matplotlib', 'pygame', 'moderngl'],
                                default='matplotlib', help='Visualization method')
    realtime_parser.add_argument('--sigma', type=float, default=10.0, help='Sigma parameter')
    realtime_parser.add_argument('--rho', type=float, default=28.0, help='Rho parameter')
    realtime_parser.add_argument('--beta', type=float, default=8/3, help='Beta parameter')
    realtime_parser.add_argument('--x0', type=float, default=1.0, help='Initial X coordinate')
    realtime_parser.add_argument('--y0', type=float, default=1.0, help='Initial Y coordinate')
    realtime_parser.add_argument('--z0', type=float, default=1.0, help='Initial Z coordinate')
    realtime_parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    realtime_parser.add_argument('--trail', type=int, default=2000, help='Trail length')
    
    # Parameter sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Parameter sweep analysis')
    sweep_parser.add_argument('--parameter', choices=['sigma', 'rho', 'beta'],
                             required=True, help='Parameter to sweep')
    sweep_parser.add_argument('--range', nargs=2, type=float, required=True,
                             help='Parameter range (min max)')
    sweep_parser.add_argument('--steps', type=int, default=50, help='Number of parameter values')
    sweep_parser.add_argument('--x0', type=float, default=1.0, help='Initial X coordinate')
    sweep_parser.add_argument('--y0', type=float, default=1.0, help='Initial Y coordinate')
    sweep_parser.add_argument('--z0', type=float, default=1.0, help='Initial Z coordinate')
    sweep_parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    sweep_parser.add_argument('--sim-steps', type=int, default=10000, help='Simulation steps')
    sweep_parser.add_argument('--method', choices=['euler', 'rk4', 'adaptive', 'dopri5'],
                             default='rk4', help='Integration method')
    sweep_parser.add_argument('--output', type=str, help='Output plot file')
    sweep_parser.add_argument('--export-video', type=str, help='Export sweep video')
    sweep_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    # Bifurcation command
    bifurcation_parser = subparsers.add_parser('bifurcation', help='Bifurcation analysis')
    bifurcation_parser.add_argument('--parameter', choices=['sigma', 'rho', 'beta'],
                                   required=True, help='Parameter to analyze')
    bifurcation_parser.add_argument('--range', nargs=2, type=float, required=True,
                                   help='Parameter range (min max)')
    bifurcation_parser.add_argument('--points', type=int, default=200, help='Number of points')
    bifurcation_parser.add_argument('--x0', type=float, default=1.0, help='Initial X coordinate')
    bifurcation_parser.add_argument('--y0', type=float, default=1.0, help='Initial Y coordinate')
    bifurcation_parser.add_argument('--z0', type=float, default=1.0, help='Initial Z coordinate')
    bifurcation_parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    bifurcation_parser.add_argument('--sim-steps', type=int, default=20000, help='Simulation steps')
    bifurcation_parser.add_argument('--output', type=str, help='Output plot file')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Advanced analysis')
    analysis_parser.add_argument('--type', choices=['lyapunov', 'poincare', 'sensitivity'],
                                required=True, help='Analysis type')
    analysis_parser.add_argument('--sigma', type=float, default=10.0, help='Sigma parameter')
    analysis_parser.add_argument('--rho', type=float, default=28.0, help='Rho parameter')
    analysis_parser.add_argument('--beta', type=float, default=8/3, help='Beta parameter')
    analysis_parser.add_argument('--x0', type=float, default=1.0, help='Initial X coordinate')
    analysis_parser.add_argument('--y0', type=float, default=1.0, help='Initial Y coordinate')
    analysis_parser.add_argument('--z0', type=float, default=1.0, help='Initial Z coordinate')
    analysis_parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    analysis_parser.add_argument('--steps', type=int, default=50000, help='Number of steps')
    analysis_parser.add_argument('--output', type=str, help='Output file')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--host', type=str, default='localhost', help='Host address')
    web_parser.add_argument('--port', type=int, default=8050, help='Port number')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser


def cmd_simulate(args) -> None:
    """Handle simulate command."""
    # Create parameters
    params = LorenzParameters(sigma=args.sigma, rho=args.rho, beta=args.beta)
    
    # Create initial conditions
    if args.random_ic:
        initial_conditions = InitialConditions.random(scale=2.0)
    elif args.true_random:
        initial_conditions = InitialConditions.true_random(scale=2.0)
    else:
        initial_conditions = InitialConditions(x=args.x0, y=args.y0, z=args.z0)
    
    # Create simulation config
    config = SimulationConfig(
        dt=args.dt,
        num_steps=args.steps,
        integration_method=args.method
    )
    
    # Run simulation
    print(f"Running simulation with parameters: σ={params.sigma}, ρ={params.rho}, β={params.beta}")
    print(f"Initial conditions: ({initial_conditions.x:.3f}, {initial_conditions.y:.3f}, {initial_conditions.z:.3f})")
    print(f"Integration method: {config.integration_method}")
    
    system = LorenzSystem(params)
    simulator = Simulator(system)
    result = simulator.simulate(initial_conditions, config)
    
    print(f"Simulation completed in {result.metadata['simulation_time']:.3f} seconds")
    
    # Save results
    if args.output:
        result.save(args.output)
        print(f"Results saved to {args.output}")
    
    # Export data
    if args.export_data:
        exporter = DataExporter()
        exporter.export_csv(result, args.export_data)
        print(f"Data exported to {args.export_data}")
    
    # Create visualizations
    if args.plot or args.export_video:
        # Set DPI based on quality setting
        dpi = 300 if args.high_quality else 150
        plotter = LorenzPlotter(style=args.style, dpi=dpi)
        
        if args.plot:
            fig = plotter.plot_3d_trajectory(
                result, 
                color_by_time=args.color_by_time,
                fast_mode=not args.high_quality or args.fast_plot
            )
            import matplotlib.pyplot as plt
            plt.show()
        
        if args.export_video:
            video_exporter = VideoExporter()
            video_exporter.export_trajectory_animation(result, args.export_video)
            print(f"Video exported to {args.export_video}")


def cmd_realtime(args) -> None:
    """Handle realtime command."""
    # Create parameters
    params = LorenzParameters(sigma=args.sigma, rho=args.rho, beta=args.beta)
    initial_conditions = InitialConditions(x=args.x0, y=args.y0, z=args.z0)
    config = SimulationConfig(dt=args.dt, num_steps=100000)
    
    # Create simulator and visualizer
    system = LorenzSystem(params)
    simulator = Simulator(system)
    visualizer = RealtimeVisualizer(simulator, trail_length=args.trail)
    
    print(f"Starting real-time visualization using {args.method}")
    print("Press ESC or close window to exit")
    
    try:
        if args.method == 'matplotlib':
            anim = visualizer.start_matplotlib_animation(initial_conditions, config)
            import matplotlib.pyplot as plt
            plt.show()
        elif args.method == 'pygame':
            visualizer.start_pygame_visualization(initial_conditions, config)
        elif args.method == 'moderngl':
            visualizer.start_moderngl_visualization(initial_conditions, config)
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error in real-time visualization: {e}")


def cmd_sweep(args) -> None:
    """Handle parameter sweep command."""
    # Create base parameters
    base_params = LorenzParameters()
    initial_conditions = InitialConditions(x=args.x0, y=args.y0, z=args.z0)
    config = SimulationConfig(
        dt=args.dt,
        num_steps=args.sim_steps,
        integration_method=args.method
    )
    
    # Create parameter values
    param_values = np.linspace(args.range[0], args.range[1], args.steps)
    
    # Create simulator
    system = LorenzSystem(base_params)
    simulator = Simulator(system)
    
    print(f"Running parameter sweep: {args.parameter} from {args.range[0]} to {args.range[1]}")
    print(f"Number of parameter values: {args.steps}")
    
    # Run sweep
    results = simulator.parameter_sweep(
        args.parameter, param_values, initial_conditions, config
    )
    
    print(f"Parameter sweep completed")
    
    # Create plots
    plotter = LorenzPlotter()
    fig = plotter.plot_parameter_sweep(results, args.parameter, param_values)
    
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    
    # Export video if requested
    if args.export_video:
        video_exporter = VideoExporter()
        video_exporter.export_parameter_sweep_video(
            results, args.parameter, param_values, args.export_video
        )
        print(f"Video exported to {args.export_video}")


def cmd_bifurcation(args) -> None:
    """Handle bifurcation analysis command."""
    # Create parameters
    base_params = LorenzParameters()
    initial_conditions = InitialConditions(x=args.x0, y=args.y0, z=args.z0)
    config = SimulationConfig(dt=args.dt, num_steps=args.sim_steps)
    
    # Create simulator
    system = LorenzSystem(base_params)
    simulator = Simulator(system)
    
    print(f"Running bifurcation analysis: {args.parameter} from {args.range[0]} to {args.range[1]}")
    print(f"Number of points: {args.points}")
    
    # Run bifurcation analysis
    bifurcation_data = simulator.bifurcation_analysis(
        args.parameter, tuple(args.range), args.points, initial_conditions, config
    )
    
    print("Bifurcation analysis completed")
    
    # Create plot
    plotter = LorenzPlotter()
    fig = plotter.plot_bifurcation_diagram(bifurcation_data)
    
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Bifurcation diagram saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_analysis(args) -> None:
    """Handle advanced analysis command."""
    # Create parameters
    params = LorenzParameters(sigma=args.sigma, rho=args.rho, beta=args.beta)
    initial_conditions = InitialConditions(x=args.x0, y=args.y0, z=args.z0)
    config = SimulationConfig(dt=args.dt, num_steps=args.steps)
    
    # Create simulator
    system = LorenzSystem(params)
    simulator = Simulator(system)
    
    if args.type == 'lyapunov':
        print("Calculating Lyapunov exponents...")
        lyapunov_exponents = simulator.estimate_lyapunov_exponent(initial_conditions, config)
        print(f"Largest Lyapunov exponent: {lyapunov_exponents:.6f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"Lyapunov exponent: {lyapunov_exponents:.6f}\n")
                f.write(f"Parameters: σ={params.sigma}, ρ={params.rho}, β={params.beta}\n")
                f.write(f"Initial conditions: ({initial_conditions.x}, {initial_conditions.y}, {initial_conditions.z})\n")
    
    elif args.type == 'poincare':
        print("Computing Poincaré section...")
        result = simulator.simulate(initial_conditions, config)
        poincare_points = result.poincare_section()
        
        print(f"Found {len(poincare_points)} Poincaré section points")
        
        if args.output:
            np.savetxt(args.output, poincare_points, header="X Y Z")
            print(f"Poincaré section saved to {args.output}")
    
    elif args.type == 'sensitivity':
        print("Running sensitivity analysis...")
        sensitivity_data = simulator.sensitivity_analysis(initial_conditions, config=config)
        
        print("Sensitivity analysis completed")
        
        # Create plot
        plotter = LorenzPlotter()
        fig = plotter.plot_sensitivity_analysis(sensitivity_data)
        
        if args.output:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Sensitivity analysis plot saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()


def cmd_web(args) -> None:
    """Handle web interface command."""
    print(f"Starting web interface at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    from .web.app import create_app
    app = create_app()
    app.run_server(debug=args.debug, host=args.host, port=args.port)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'simulate':
            cmd_simulate(args)
        elif args.command == 'realtime':
            cmd_realtime(args)
        elif args.command == 'sweep':
            cmd_sweep(args)
        elif args.command == 'bifurcation':
            cmd_bifurcation(args)
        elif args.command == 'analysis':
            cmd_analysis(args)
        elif args.command == 'web':
            cmd_web(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()