"""Advanced plotting capabilities for Lorenz attractor visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any, Tuple

from ..core.simulator import SimulationResult

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class LorenzPlotter:
    """Advanced plotting class for Lorenz attractor visualizations."""
    
    def __init__(self, style: str = 'dark', dpi: int = 150):
        """
        Initialize the plotter.
        
        Args:
            style: Plotting style ('dark', 'light', 'scientific')
            dpi: DPI for plots (150 for fast rendering, 300 for high quality)
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Setup plotting style."""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1f1f1f'
            self.text_color = 'white'
            self.accent_color = '#00d4ff'
        elif self.style == 'light':
            if HAS_SEABORN:
                plt.style.use('seaborn-v0_8-whitegrid')
            else:
                plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.accent_color = '#ff6b6b'
        else:  # scientific
            if HAS_SEABORN:
                plt.style.use('seaborn-v0_8-paper')
            else:
                plt.style.use('classic')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.accent_color = '#2196f3'
    
    def plot_3d_trajectory(self, result: SimulationResult, 
                          title: str = "Lorenz Attractor",
                          color_by_time: bool = False,
                          line_width: float = 0.5,
                          show_axes: bool = True,
                          save_path: Optional[str] = None,
                          fast_mode: bool = True) -> plt.Figure:
        """
        Plot 3D trajectory of the Lorenz attractor.
        
        Args:
            result: Simulation result
            title: Plot title
            color_by_time: Whether to color trajectory by time
            line_width: Line width for trajectory
            show_axes: Whether to show axes
            save_path: Path to save the plot
            fast_mode: Use fast rendering (decimation for long trajectories)
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Decimate data for performance if trajectory is very long
        trajectory = result.trajectory
        if fast_mode and len(trajectory) > 5000:
            step = len(trajectory) // 5000
            trajectory = trajectory[::step]
        
        if color_by_time and len(trajectory) > 1:
            # Use Line3DCollection for efficient colored line rendering
            points = trajectory.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
            
            # Create collection
            lc = Line3DCollection(segments, colors=colors, linewidths=line_width)
            ax.add_collection3d(lc)
        else:
            # Simple single-color plot for best performance
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=self.accent_color, linewidth=line_width)
        
        # Mark initial condition
        ax.scatter([result.trajectory[0, 0]], 
                  [result.trajectory[0, 1]], 
                  [result.trajectory[0, 2]], 
                  color='red', s=50, label='Initial Condition')
        
        ax.set_xlabel('X', color=self.text_color)
        ax.set_ylabel('Y', color=self.text_color)
        ax.set_zlabel('Z', color=self.text_color)
        ax.set_title(title, color=self.text_color, fontsize=16)
        
        # Set appropriate axis limits for Lorenz attractor
        ax.set_xlim([-25, 25])
        ax.set_ylim([-35, 35])
        ax.set_zlim([0, 50])
        
        # Set optimal viewing angle
        ax.view_init(elev=15, azim=45)
        
        if not show_axes:
            ax.set_axis_off()
        
        # Add parameter information
        params = result.parameters
        info_text = f'σ={params.sigma:.2f}, ρ={params.rho:.2f}, β={params.beta:.2f}'
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 color=self.text_color, fontsize=10, va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_2d_projections(self, result: SimulationResult,
                           title: str = "Lorenz Attractor Projections",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D projections of the trajectory.
        
        Args:
            result: Simulation result
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # XY projection
        axes[0, 0].plot(result.x, result.y, color=self.accent_color, linewidth=0.5)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('XY Projection')
        axes[0, 0].grid(True, alpha=0.3)
        
        # XZ projection
        axes[0, 1].plot(result.x, result.z, color=self.accent_color, linewidth=0.5)
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        axes[0, 1].set_title('XZ Projection')
        axes[0, 1].grid(True, alpha=0.3)
        
        # YZ projection
        axes[1, 0].plot(result.y, result.z, color=self.accent_color, linewidth=0.5)
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].set_title('YZ Projection')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series
        axes[1, 1].plot(result.time, result.z, color=self.accent_color, linewidth=0.5)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Z')
        axes[1, 1].set_title('Z vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_phase_space(self, result: SimulationResult,
                        poincare_plane: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot phase space analysis including Poincaré sections.
        
        Args:
            result: Simulation result
            poincare_plane: Dictionary with 'normal' and 'offset' keys
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(result.x, result.y, result.z, color=self.accent_color, linewidth=0.5)
        ax1.set_title('3D Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Poincaré section
        if poincare_plane:
            poincare_points = result.poincare_section(
                poincare_plane.get('normal', np.array([0, 0, 1])),
                poincare_plane.get('offset', 27.0)
            )
            
            ax2 = fig.add_subplot(2, 2, 2)
            if len(poincare_points) > 0:
                ax2.scatter(poincare_points[:, 0], poincare_points[:, 1], 
                           s=1, alpha=0.6, color=self.accent_color)
            ax2.set_title('Poincaré Section')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.grid(True, alpha=0.3)
        
        # Return map (if Poincaré section exists)
        if poincare_plane and len(poincare_points) > 1:
            ax3 = fig.add_subplot(2, 2, 3)
            z_values = poincare_points[:, 2]
            ax3.scatter(z_values[:-1], z_values[1:], s=1, alpha=0.6, color=self.accent_color)
            ax3.set_title('Return Map')
            ax3.set_xlabel('Z_n')
            ax3.set_ylabel('Z_{n+1}')
            ax3.grid(True, alpha=0.3)
        
        # Power spectrum
        ax4 = fig.add_subplot(2, 2, 4)
        freqs = np.fft.fftfreq(len(result.z), result.time[1] - result.time[0])
        power = np.abs(np.fft.fft(result.z))**2
        ax4.loglog(freqs[1:len(freqs)//2], power[1:len(freqs)//2], 
                   color=self.accent_color, linewidth=0.5)
        ax4.set_title('Power Spectrum')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Power')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_3d(self, result: SimulationResult,
                           title: str = "Interactive Lorenz Attractor") -> go.Figure:
        """
        Create interactive 3D plot using Plotly.
        
        Args:
            result: Simulation result
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Color by time
        colors = np.linspace(0, 1, len(result.time))
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=result.x,
                y=result.y,
                z=result.z,
                mode='lines',
                line=dict(
                    color=colors,
                    colorscale='Viridis',
                    width=3
                ),
                name='Trajectory'
            ),
            go.Scatter3d(
                x=[result.x[0]],
                y=[result.y[0]],
                z=[result.z[0]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=8
                ),
                name='Initial Condition'
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_parameter_sweep(self, results: List[SimulationResult],
                           parameter_name: str,
                           parameter_values: np.ndarray,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot results from parameter sweep.
        
        Args:
            results: List of simulation results
            parameter_name: Name of swept parameter
            parameter_values: Array of parameter values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # Max Z vs parameter
        max_z = [np.max(result.z) for result in results]
        axes[0, 0].plot(parameter_values, max_z, 'o-', color=self.accent_color)
        axes[0, 0].set_xlabel(parameter_name)
        axes[0, 0].set_ylabel('Max Z')
        axes[0, 0].set_title(f'Max Z vs {parameter_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Min Z vs parameter
        min_z = [np.min(result.z) for result in results]
        axes[0, 1].plot(parameter_values, min_z, 'o-', color=self.accent_color)
        axes[0, 1].set_xlabel(parameter_name)
        axes[0, 1].set_ylabel('Min Z')
        axes[0, 1].set_title(f'Min Z vs {parameter_name}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Standard deviation vs parameter
        std_z = [np.std(result.z) for result in results]
        axes[1, 0].plot(parameter_values, std_z, 'o-', color=self.accent_color)
        axes[1, 0].set_xlabel(parameter_name)
        axes[1, 0].set_ylabel('Std Z')
        axes[1, 0].set_title(f'Z Standard Deviation vs {parameter_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample trajectories
        for i, result in enumerate(results[::len(results)//5]):  # Sample every 5th result
            axes[1, 1].plot(result.time, result.z, alpha=0.7, linewidth=0.5,
                           label=f'{parameter_name}={parameter_values[i*len(results)//5]:.2f}')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Z')
        axes[1, 1].set_title('Sample Trajectories')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter Sweep: {parameter_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_bifurcation_diagram(self, bifurcation_data: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot bifurcation diagram.
        
        Args:
            bifurcation_data: Dictionary with bifurcation analysis results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        param_values = bifurcation_data['parameter_values']
        attractors = bifurcation_data['attractors']
        param_name = bifurcation_data['parameter_name']
        
        # Plot attractor points
        for i, (param_val, attractor) in enumerate(zip(param_values, attractors)):
            # Use Z coordinate for bifurcation diagram
            z_values = attractor[:, 2]
            param_array = np.full_like(z_values, param_val)
            
            ax.scatter(param_array, z_values, s=0.1, alpha=0.5, 
                      color=self.accent_color, rasterized=True)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Z')
        ax.set_title(f'Bifurcation Diagram: {param_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_data: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_data: Dictionary with sensitivity analysis results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        time = sensitivity_data['time']
        divergences = sensitivity_data['divergences']
        
        # Plot divergences
        labels = ['X perturbation', 'Y perturbation', 'Z perturbation']
        for i, (div, label) in enumerate(zip(divergences, labels)):
            axes[0, 0].semilogy(time, div, label=label, linewidth=1)
        
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Divergence')
        axes[0, 0].set_title('Trajectory Divergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Lyapunov exponent estimation
        # Simple estimation: log(divergence) / time
        for i, (div, label) in enumerate(zip(divergences, labels)):
            # Avoid log(0) by adding small epsilon
            log_div = np.log(div + 1e-15)
            lyap_estimate = log_div / time[1:]  # Skip first point
            axes[0, 1].plot(time[1:], lyap_estimate, label=label, linewidth=1)
        
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Lyapunov Exponent Estimate')
        axes[0, 1].set_title('Local Lyapunov Exponent')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot original vs perturbed trajectory (3D)
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')
        original_traj = sensitivity_data['original_trajectory']
        perturbed_traj = sensitivity_data['perturbed_trajectories'][0]  # X perturbation
        
        ax3d.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2],
                 color='blue', linewidth=0.5, label='Original')
        ax3d.plot(perturbed_traj[:, 0], perturbed_traj[:, 1], perturbed_traj[:, 2],
                 color='red', linewidth=0.5, label='Perturbed')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('Original vs Perturbed Trajectory')
        ax3d.legend()
        
        # Plot distance evolution
        distance = np.linalg.norm(original_traj - perturbed_traj, axis=1)
        axes[1, 1].plot(time, distance, color=self.accent_color, linewidth=1)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].set_title('Distance Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig