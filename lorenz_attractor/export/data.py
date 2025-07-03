"""Data export capabilities for Lorenz attractor simulations."""

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from datetime import datetime

# Optional imports
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

from ..core.simulator import SimulationResult


class DataExporter:
    """Export simulation data in various formats."""
    
    def __init__(self):
        """Initialize data exporter."""
        pass
    
    def export_csv(self, result: SimulationResult, filename: str,
                   include_metadata: bool = True) -> str:
        """
        Export simulation result to CSV format.
        
        Args:
            result: Simulation result
            filename: Output filename
            include_metadata: Whether to include metadata in comments
            
        Returns:
            Path to exported file
        """
        # Create DataFrame
        df = pd.DataFrame({
            'time': result.time,
            'x': result.x,
            'y': result.y,
            'z': result.z
        })
        
        # Add metadata as comments if requested
        if include_metadata:
            metadata_lines = [
                f"# Lorenz Attractor Simulation Data",
                f"# Generated on: {datetime.now().isoformat()}",
                f"# Parameters: σ={result.parameters.sigma}, ρ={result.parameters.rho}, β={result.parameters.beta}",
                f"# Initial conditions: x0={result.initial_conditions.x}, y0={result.initial_conditions.y}, z0={result.initial_conditions.z}",
                f"# Integration method: {result.config.integration_method}",
                f"# Time step: {result.config.dt}",
                f"# Number of steps: {result.config.num_steps}",
                f"# Simulation time: {result.metadata.get('simulation_time', 'N/A')} seconds",
            ]
            
            # Write metadata and data
            with open(filename, 'w') as f:
                for line in metadata_lines:
                    f.write(line + '\n')
                f.write('\n')
                df.to_csv(f, index=False)
        else:
            df.to_csv(filename, index=False)
        
        return filename
    
    def export_json(self, result: SimulationResult, filename: str,
                   compact: bool = False) -> str:
        """
        Export simulation result to JSON format.
        
        Args:
            result: Simulation result
            filename: Output filename
            compact: Whether to use compact format
            
        Returns:
            Path to exported file
        """
        data = {
            'metadata': {
                'title': 'Lorenz Attractor Simulation Data',
                'generated_on': datetime.now().isoformat(),
                'version': '2.0.0'
            },
            'parameters': result.parameters.to_dict(),
            'initial_conditions': {
                'x': result.initial_conditions.x,
                'y': result.initial_conditions.y,
                'z': result.initial_conditions.z
            },
            'config': result.config.to_dict(),
            'simulation_metadata': result.metadata,
            'data': {
                'time': result.time.tolist(),
                'trajectory': result.trajectory.tolist()
            }
        }
        
        indent = None if compact else 2
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=indent)
        
        return filename
    
    def export_hdf5(self, result: SimulationResult, filename: str) -> str:
        """
        Export simulation result to HDF5 format.
        
        Args:
            result: Simulation result
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")
            
        with h5py.File(filename, 'w') as f:
            # Create groups
            data_group = f.create_group('data')
            params_group = f.create_group('parameters')
            config_group = f.create_group('config')
            metadata_group = f.create_group('metadata')
            
            # Store data
            data_group.create_dataset('time', data=result.time)
            data_group.create_dataset('trajectory', data=result.trajectory)
            data_group.create_dataset('x', data=result.x)
            data_group.create_dataset('y', data=result.y)
            data_group.create_dataset('z', data=result.z)
            
            # Store parameters
            params_group.attrs['sigma'] = result.parameters.sigma
            params_group.attrs['rho'] = result.parameters.rho
            params_group.attrs['beta'] = result.parameters.beta
            
            # Store initial conditions
            ic_group = params_group.create_group('initial_conditions')
            ic_group.attrs['x'] = result.initial_conditions.x
            ic_group.attrs['y'] = result.initial_conditions.y
            ic_group.attrs['z'] = result.initial_conditions.z
            
            # Store configuration
            config_group.attrs['dt'] = result.config.dt
            config_group.attrs['num_steps'] = result.config.num_steps
            config_group.attrs['integration_method'] = result.config.integration_method
            config_group.attrs['save_interval'] = result.config.save_interval
            
            # Store metadata
            for key, value in result.metadata.items():
                if isinstance(value, (int, float, str)):
                    metadata_group.attrs[key] = value
            
            # Add global attributes
            f.attrs['title'] = 'Lorenz Attractor Simulation Data'
            f.attrs['version'] = '2.0.0'
            f.attrs['generated_on'] = datetime.now().isoformat()
        
        return filename
    
    def export_numpy(self, result: SimulationResult, filename: str) -> str:
        """
        Export simulation result to NumPy format.
        
        Args:
            result: Simulation result
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        # Create structured array
        data = {
            'time': result.time,
            'trajectory': result.trajectory,
            'parameters': result.parameters.to_dict(),
            'initial_conditions': result.initial_conditions.to_array(),
            'config': result.config.to_dict(),
            'metadata': result.metadata
        }
        
        np.savez_compressed(filename, **data)
        return filename
    
    def export_matlab(self, result: SimulationResult, filename: str) -> str:
        """
        Export simulation result to MATLAB format.
        
        Args:
            result: Simulation result
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("scipy is required for MATLAB export")
        
        data = {
            'time': result.time,
            'x': result.x,
            'y': result.y,
            'z': result.z,
            'trajectory': result.trajectory,
            'sigma': result.parameters.sigma,
            'rho': result.parameters.rho,
            'beta': result.parameters.beta,
            'x0': result.initial_conditions.x,
            'y0': result.initial_conditions.y,
            'z0': result.initial_conditions.z,
            'dt': result.config.dt,
            'num_steps': result.config.num_steps,
            'integration_method': result.config.integration_method
        }
        
        savemat(filename, data)
        return filename
    
    def export_pickle(self, result: SimulationResult, filename: str) -> str:
        """
        Export simulation result to pickle format.
        
        Args:
            result: Simulation result
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        with open(filename, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filename
    
    def export_multiple_results(self, results: List[SimulationResult],
                               base_filename: str, format: str = 'csv') -> List[str]:
        """
        Export multiple simulation results.
        
        Args:
            results: List of simulation results
            base_filename: Base filename (will be numbered)
            format: Export format
            
        Returns:
            List of exported filenames
        """
        exported_files = []
        base_path = Path(base_filename)
        
        for i, result in enumerate(results):
            filename = f"{base_path.stem}_{i:03d}{base_path.suffix}"
            
            if format == 'csv':
                self.export_csv(result, filename)
            elif format == 'json':
                self.export_json(result, filename)
            elif format == 'hdf5':
                if HAS_HDF5:
                    self.export_hdf5(result, filename)
                else:
                    print(f"Warning: HDF5 export not available, skipping {filename}")
                    continue
            elif format == 'numpy':
                self.export_numpy(result, filename)
            elif format == 'matlab':
                self.export_matlab(result, filename)
            elif format == 'pickle':
                self.export_pickle(result, filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            exported_files.append(filename)
        
        return exported_files
    
    def export_parameter_sweep_summary(self, results: List[SimulationResult],
                                     parameter_name: str,
                                     parameter_values: np.ndarray,
                                     filename: str) -> str:
        """
        Export parameter sweep summary statistics.
        
        Args:
            results: List of simulation results
            parameter_name: Name of swept parameter
            parameter_values: Array of parameter values
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        summary_data = []
        
        for param_val, result in zip(parameter_values, results):
            # Calculate statistics
            stats = {
                parameter_name: param_val,
                'max_x': np.max(result.x),
                'min_x': np.min(result.x),
                'mean_x': np.mean(result.x),
                'std_x': np.std(result.x),
                'max_y': np.max(result.y),
                'min_y': np.min(result.y),
                'mean_y': np.mean(result.y),
                'std_y': np.std(result.y),
                'max_z': np.max(result.z),
                'min_z': np.min(result.z),
                'mean_z': np.mean(result.z),
                'std_z': np.std(result.z),
                'simulation_time': result.metadata.get('simulation_time', 0),
                'total_points': result.metadata.get('total_points', 0)
            }
            summary_data.append(stats)
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def export_bifurcation_data(self, bifurcation_data: Dict[str, Any],
                               filename: str) -> str:
        """
        Export bifurcation analysis data.
        
        Args:
            bifurcation_data: Dictionary with bifurcation analysis results
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        param_values = bifurcation_data['parameter_values']
        attractors = bifurcation_data['attractors']
        param_name = bifurcation_data['parameter_name']
        
        # Flatten attractor data
        all_data = []
        
        for param_val, attractor in zip(param_values, attractors):
            for point in attractor:
                all_data.append({
                    param_name: param_val,
                    'x': point[0],
                    'y': point[1],
                    'z': point[2]
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def create_data_package(self, results: List[SimulationResult],
                           package_name: str, include_plots: bool = True) -> str:
        """
        Create a comprehensive data package with multiple formats.
        
        Args:
            results: List of simulation results
            package_name: Package directory name
            include_plots: Whether to include plots
            
        Returns:
            Path to package directory
        """
        package_path = Path(package_name)
        package_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (package_path / 'data').mkdir(exist_ok=True)
        (package_path / 'plots').mkdir(exist_ok=True)
        (package_path / 'metadata').mkdir(exist_ok=True)
        
        exported_files = []
        
        for i, result in enumerate(results):
            base_name = f"simulation_{i:03d}"
            
            # Export in multiple formats
            csv_file = self.export_csv(result, package_path / 'data' / f"{base_name}.csv")
            json_file = self.export_json(result, package_path / 'data' / f"{base_name}.json")
            numpy_file = self.export_numpy(result, package_path / 'data' / f"{base_name}.npz")
            
            exported_files.extend([csv_file, json_file, numpy_file])
            
            # Create plots if requested
            if include_plots:
                from ..visualization.plotter import LorenzPlotter
                plotter = LorenzPlotter()
                
                # 3D plot
                fig = plotter.plot_3d_trajectory(result, title=f"Simulation {i}")
                fig.savefig(package_path / 'plots' / f"{base_name}_3d.png", 
                           dpi=300, bbox_inches='tight')
                
                # 2D projections
                fig = plotter.plot_2d_projections(result, title=f"Simulation {i} - Projections")
                fig.savefig(package_path / 'plots' / f"{base_name}_2d.png",
                           dpi=300, bbox_inches='tight')
                
                import matplotlib.pyplot as plt
                plt.close('all')
        
        # Create package manifest
        manifest = {
            'package_name': package_name,
            'created_on': datetime.now().isoformat(),
            'version': '2.0.0',
            'num_simulations': len(results),
            'files': [str(f) for f in exported_files],
            'description': 'Lorenz Attractor Simulation Data Package'
        }
        
        with open(package_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create README
        readme_content = f"""# Lorenz Attractor Simulation Data Package

Generated on: {datetime.now().isoformat()}
Number of simulations: {len(results)}

## Directory Structure

- `data/`: Simulation data in multiple formats (CSV, JSON, NumPy)
- `plots/`: Visualization plots (PNG format)
- `metadata/`: Additional metadata files
- `manifest.json`: Package manifest with file listing

## Data Formats

- **CSV**: Human-readable format with comments
- **JSON**: Structured format with full metadata
- **NumPy**: Binary format for Python analysis

## Usage

```python
import numpy as np
import pandas as pd

# Load CSV data
data = pd.read_csv('data/simulation_000.csv', comment='#')

# Load NumPy data
data = np.load('data/simulation_000.npz', allow_pickle=True)
```
"""
        
        with open(package_path / 'README.md', 'w') as f:
            f.write(readme_content)
        
        return str(package_path)