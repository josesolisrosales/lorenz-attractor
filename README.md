# Lorenz Attractor Professional Suite

A comprehensive, professional-grade simulation and visualization toolkit for the Lorenz attractor and related chaotic dynamical systems. This project transforms a basic college simulation into a production-quality scientific computing package.

## 🌟 Features

### Core Simulation Engine
- **Advanced Numerical Integration**: Multiple methods including Euler, RK4, adaptive RK, and Dormand-Prince 5(4)
- **High-Performance Computing**: Numba-compiled critical functions for optimal performance
- **Parameter Management**: Type-safe parameter handling with validation
- **True Random Initial Conditions**: Integration with system entropy for reproducible randomness

### Visualization & Graphics
- **Real-time 3D Visualization**: OpenGL, ModernGL, and Pygame backends
- **Professional Plotting**: Publication-quality matplotlib and interactive Plotly visualizations
- **Multiple View Modes**: 3D trajectories, 2D projections, phase space analysis
- **Video Export**: High-quality MP4 and GIF animation export with customizable quality settings

### Advanced Analysis
- **Parameter Sweeps**: Automated exploration of parameter space
- **Bifurcation Analysis**: Comprehensive bifurcation diagram generation
- **Lyapunov Exponents**: Accurate calculation of system stability measures  
- **Poincaré Sections**: Phase space cross-sections for detailed analysis
- **Sensitivity Analysis**: Butterfly effect demonstration and quantification

### Data Management
- **Multiple Export Formats**: CSV, JSON, HDF5, NumPy, MATLAB, and Pickle
- **Comprehensive Data Packages**: Bundled exports with metadata and documentation
- **Simulation Metadata**: Detailed tracking of simulation parameters and performance

### User Interfaces
- **Command Line Interface**: Full-featured CLI with comprehensive options
- **Web Application**: Interactive Dash-based web interface for exploration
- **Python API**: Clean, documented API for programmatic use

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/josesolisrosales/lorenz-attractor.git
cd lorenz-attractor

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from lorenz_attractor import LorenzSystem, Simulator, LorenzPlotter
from lorenz_attractor.core.parameters import InitialConditions, SimulationConfig

# Create and run simulation
system = LorenzSystem()
simulator = Simulator(system)

initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)
config = SimulationConfig(dt=0.01, num_steps=10000)

result = simulator.simulate(initial_conditions, config)

# Visualize results
plotter = LorenzPlotter(style='dark')
fig = plotter.plot_3d_trajectory(result)
```

### Command Line Interface

```bash
# Basic simulation
lorenz-sim simulate --sigma 10 --rho 28 --beta 2.667

# Parameter sweep
lorenz-sim sweep --parameter rho --range 20 30 --steps 100

# Real-time visualization
lorenz-sim realtime --method pygame

# Launch web interface
lorenz-sim web
```

### Web Interface

```bash
# Launch interactive web application
lorenz-sim web --host 0.0.0.0 --port 8050
```

Then open http://localhost:8050 in your browser.

## 📊 Examples

### Parameter Sensitivity Analysis
```python
# Demonstrate butterfly effect
base_ic = InitialConditions(x=1.0, y=1.0, z=1.0)
perturbed_ic = InitialConditions(x=1.0 + 1e-10, y=1.0, z=1.0)

results = simulator.simulate_multiple([base_ic, perturbed_ic], config)
# Results show exponential divergence characteristic of chaos
```

### Bifurcation Analysis
```python
# Analyze system behavior across parameter range
bifurcation_data = simulator.bifurcation_analysis(
    'rho', (0.5, 50), num_points=200, 
    initial_conditions=base_ic, config=config
)

plotter = LorenzPlotter()
fig = plotter.plot_bifurcation_diagram(bifurcation_data)
```

### Video Export
```python
# Create high-quality animation
video_exporter = VideoExporter(fps=60, dpi=200)
video_exporter.export_trajectory_animation(
    result, "lorenz_evolution.mp4", 
    quality='ultra', trail_length=1000
)
```

## 🏗️ Architecture

The package is organized into modular components:

```
lorenz_attractor/
├── core/           # Core simulation engine
│   ├── lorenz.py   # Lorenz system implementation
│   ├── simulator.py # Simulation orchestration
│   └── parameters.py # Parameter management
├── integration/    # Numerical integration methods
├── visualization/  # Plotting and real-time graphics
├── export/         # Data and video export
├── analysis/       # Advanced analysis tools
├── web/           # Web interface
└── utils/         # Utility functions
```

## 🎯 Advanced Features

### Performance Optimization
- Numba JIT compilation for critical loops
- Parallel simulation execution
- Memory-efficient data structures
- Optimized integration algorithms

### Scientific Computing
- IEEE 754 compliant floating-point arithmetic
- Adaptive step size control
- Error estimation and monitoring
- Numerical stability analysis

### Extensibility
- Plugin architecture for custom integrators
- Configurable visualization backends
- Extensible export formats
- Modular analysis components

## 📈 Performance

Typical performance on modern hardware:
- **Basic simulation** (10k points): ~0.1 seconds
- **Parameter sweep** (100 simulations): ~10 seconds  
- **Bifurcation analysis** (1000 points): ~2 minutes
- **Video export** (1 minute HD): ~30 seconds

## 🔬 Scientific Applications

This toolkit is suitable for:
- **Chaos Theory Research**: Detailed analysis of chaotic dynamics
- **Educational Demonstrations**: Interactive exploration of nonlinear systems
- **Numerical Methods Development**: Testing integration algorithms
- **Visualization Research**: Advanced scientific visualization techniques

## 📚 Documentation

- [API Reference](docs/api.md)
- [User Guide](docs/guide.md)
- [Examples](examples/)
- [Developer Documentation](docs/development.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Edward Lorenz** for discovering the Lorenz attractor
- **Scientific Python Community** for the excellent ecosystem
- **NumPy, SciPy, Matplotlib** for foundational tools
- **Plotly, Dash** for modern web visualization

## 📞 Citation

If you use this software in academic work, please cite:

```bibtex
@software{lorenz_attractor_pro,
  title={Lorenz Attractor Professional Suite},
  author={Jose Solis Rosales},
  year={2024},
  url={https://github.com/josesolisrosales/lorenz-attractor},
  version={2.0.0}
}
```

---

**Transform your understanding of chaos with professional-grade simulation tools.** 🦋
