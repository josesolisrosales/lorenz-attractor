"""Video export capabilities for Lorenz attractor simulations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import cv2
from typing import Optional, Tuple, List, Dict, Any
import os
import tempfile
from datetime import datetime

from ..core.simulator import SimulationResult
from ..visualization.plotter import LorenzPlotter


class VideoExporter:
    """Export Lorenz attractor simulations as videos."""
    
    def __init__(self, fps: int = 30, dpi: int = 100):
        """
        Initialize video exporter.
        
        Args:
            fps: Frames per second for video output
            dpi: DPI for video frames
        """
        self.fps = fps
        self.dpi = dpi
        self.plotter = LorenzPlotter(dpi=dpi)
    
    def export_trajectory_animation(self, result: SimulationResult,
                                   filename: str,
                                   trail_length: int = 500,
                                   video_duration: Optional[float] = None,
                                   view_angle: Tuple[float, float] = (30, 45),
                                   zoom_factor: float = 1.0,
                                   quality: str = 'high') -> str:
        """
        Export trajectory animation as video.
        
        Args:
            result: Simulation result
            filename: Output filename
            trail_length: Length of trajectory trail
            video_duration: Duration of video in seconds (if None, uses full trajectory)
            view_angle: Camera view angle (elevation, azimuth)
            zoom_factor: Zoom factor for the view
            quality: Video quality ('low', 'medium', 'high', 'ultra')
            
        Returns:
            Path to exported video file
        """
        # Setup video parameters based on quality
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15, 'dpi': 72},
            'medium': {'bitrate': 2000, 'fps': 24, 'dpi': 100},
            'high': {'bitrate': 4000, 'fps': 30, 'dpi': 150},
            'ultra': {'bitrate': 8000, 'fps': 60, 'dpi': 200}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        # Determine frame indices
        total_frames = len(result.trajectory)
        if video_duration:
            num_frames = int(video_duration * settings['fps'])
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            frame_indices = np.arange(0, total_frames, max(1, total_frames // (settings['fps'] * 10)))
        
        # Setup figure
        fig = plt.figure(figsize=(12, 9), dpi=settings['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Set view
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set limits with zoom
        x_range = np.ptp(result.x)
        y_range = np.ptp(result.y) 
        z_range = np.ptp(result.z)
        max_range = max(x_range, y_range, z_range) / zoom_factor
        
        x_center = (np.max(result.x) + np.min(result.x)) / 2
        y_center = (np.max(result.y) + np.min(result.y)) / 2
        z_center = (np.max(result.z) + np.min(result.z)) / 2
        
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz Attractor Evolution')
        
        # Initialize plot elements
        line, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.7)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        
        def animate(frame_idx):
            """Animation function."""
            current_idx = frame_indices[frame_idx]
            
            # Calculate trail start index
            start_idx = max(0, current_idx - trail_length)
            
            # Get trajectory segment
            trail_x = result.x[start_idx:current_idx+1]
            trail_y = result.y[start_idx:current_idx+1]
            trail_z = result.z[start_idx:current_idx+1]
            
            # Update trail
            line.set_data(trail_x, trail_y)
            line.set_3d_properties(trail_z)
            
            # Update current point
            point.set_data([result.x[current_idx]], [result.y[current_idx]])
            point.set_3d_properties([result.z[current_idx]])
            
            # Update title with time
            ax.set_title(f'Lorenz Attractor - t = {result.time[current_idx]:.2f}')
            
            return line, point
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(frame_indices), 
                           interval=1000/settings['fps'], blit=True)
        
        # Save video
        if filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        elif filename.endswith('.gif'):
            writer = PillowWriter(fps=settings['fps'])
        else:
            # Default to mp4
            filename = filename + '.mp4'
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        
        print(f"Exporting video to {filename}...")
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        return filename
    
    def export_parameter_sweep_video(self, results: List[SimulationResult],
                                    parameter_name: str,
                                    parameter_values: np.ndarray,
                                    filename: str,
                                    quality: str = 'high') -> str:
        """
        Export parameter sweep as video.
        
        Args:
            results: List of simulation results
            parameter_name: Name of swept parameter
            parameter_values: Array of parameter values
            filename: Output filename
            quality: Video quality
            
        Returns:
            Path to exported video file
        """
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15, 'dpi': 72},
            'medium': {'bitrate': 2000, 'fps': 24, 'dpi': 100},
            'high': {'bitrate': 4000, 'fps': 30, 'dpi': 150},
            'ultra': {'bitrate': 8000, 'fps': 60, 'dpi': 200}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        # Setup figure
        fig = plt.figure(figsize=(15, 10), dpi=settings['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Set consistent view limits
        all_x = np.concatenate([r.x for r in results])
        all_y = np.concatenate([r.y for r in results])
        all_z = np.concatenate([r.z for r in results])
        
        ax.set_xlim(np.min(all_x), np.max(all_x))
        ax.set_ylim(np.min(all_y), np.max(all_y))
        ax.set_zlim(np.min(all_z), np.max(all_z))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Initialize plot elements
        line, = ax.plot([], [], [], 'b-', linewidth=0.5, alpha=0.8)
        
        def animate(frame_idx):
            """Animation function."""
            if frame_idx < len(results):
                result = results[frame_idx]
                param_value = parameter_values[frame_idx]
                
                # Update trajectory
                line.set_data(result.x, result.y)
                line.set_3d_properties(result.z)
                
                # Update title
                ax.set_title(f'Parameter Sweep: {parameter_name} = {param_value:.3f}')
            
            return line,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(results), 
                           interval=1000/settings['fps'], blit=True)
        
        # Save video
        if filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        elif filename.endswith('.gif'):
            writer = PillowWriter(fps=settings['fps'])
        else:
            filename = filename + '.mp4'
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        
        print(f"Exporting parameter sweep video to {filename}...")
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        return filename
    
    def export_multi_trajectory_video(self, results: List[SimulationResult],
                                     filename: str,
                                     colors: Optional[List[str]] = None,
                                     quality: str = 'high') -> str:
        """
        Export multiple trajectories as video.
        
        Args:
            results: List of simulation results
            filename: Output filename
            colors: List of colors for each trajectory
            quality: Video quality
            
        Returns:
            Path to exported video file
        """
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15, 'dpi': 72},
            'medium': {'bitrate': 2000, 'fps': 24, 'dpi': 100},
            'high': {'bitrate': 4000, 'fps': 30, 'dpi': 150},
            'ultra': {'bitrate': 8000, 'fps': 60, 'dpi': 200}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        if colors is None:
            colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        # Setup figure
        fig = plt.figure(figsize=(15, 10), dpi=settings['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Set consistent view limits
        all_x = np.concatenate([r.x for r in results])
        all_y = np.concatenate([r.y for r in results])
        all_z = np.concatenate([r.z for r in results])
        
        ax.set_xlim(np.min(all_x), np.max(all_x))
        ax.set_ylim(np.min(all_y), np.max(all_y))
        ax.set_zlim(np.min(all_z), np.max(all_z))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Multiple Lorenz Trajectories')
        
        # Initialize plot elements
        lines = []
        points = []
        for i, color in enumerate(colors):
            line, = ax.plot([], [], [], color=color, linewidth=0.5, alpha=0.7)
            point, = ax.plot([], [], [], 'o', color=color, markersize=6)
            lines.append(line)
            points.append(point)
        
        # Find maximum length
        max_length = max(len(result.trajectory) for result in results)
        
        def animate(frame_idx):
            """Animation function."""
            for i, (result, line, point) in enumerate(zip(results, lines, points)):
                if frame_idx < len(result.trajectory):
                    # Update trajectory up to current frame
                    line.set_data(result.x[:frame_idx+1], result.y[:frame_idx+1])
                    line.set_3d_properties(result.z[:frame_idx+1])
                    
                    # Update current point
                    point.set_data([result.x[frame_idx]], [result.y[frame_idx]])
                    point.set_3d_properties([result.z[frame_idx]])
                
            return lines + points
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=max_length, 
                           interval=1000/settings['fps'], blit=True)
        
        # Save video
        if filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        elif filename.endswith('.gif'):
            writer = PillowWriter(fps=settings['fps'])
        else:
            filename = filename + '.mp4'
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        
        print(f"Exporting multi-trajectory video to {filename}...")
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        return filename
    
    def export_rotating_view_video(self, result: SimulationResult,
                                  filename: str,
                                  rotation_speed: float = 1.0,
                                  quality: str = 'high') -> str:
        """
        Export video with rotating view around the attractor.
        
        Args:
            result: Simulation result
            filename: Output filename
            rotation_speed: Rotation speed (degrees per frame)
            quality: Video quality
            
        Returns:
            Path to exported video file
        """
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15, 'dpi': 72},
            'medium': {'bitrate': 2000, 'fps': 24, 'dpi': 100},
            'high': {'bitrate': 4000, 'fps': 30, 'dpi': 150},
            'ultra': {'bitrate': 8000, 'fps': 60, 'dpi': 200}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        # Setup figure
        fig = plt.figure(figsize=(12, 9), dpi=settings['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot full trajectory
        ax.plot(result.x, result.y, result.z, 'b-', linewidth=0.5, alpha=0.7)
        
        # Set limits
        ax.set_xlim(np.min(result.x), np.max(result.x))
        ax.set_ylim(np.min(result.y), np.max(result.y))
        ax.set_zlim(np.min(result.z), np.max(result.z))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz Attractor - Rotating View')
        
        # Calculate number of frames for full rotation
        num_frames = int(360 / rotation_speed)
        
        def animate(frame_idx):
            """Animation function."""
            azimuth = frame_idx * rotation_speed
            ax.view_init(elev=30, azim=azimuth)
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=num_frames, 
                           interval=1000/settings['fps'], blit=False)
        
        # Save video
        if filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        elif filename.endswith('.gif'):
            writer = PillowWriter(fps=settings['fps'])
        else:
            filename = filename + '.mp4'
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        
        print(f"Exporting rotating view video to {filename}...")
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        return filename
    
    def export_comparison_video(self, results: List[SimulationResult],
                               labels: List[str],
                               filename: str,
                               quality: str = 'high') -> str:
        """
        Export comparison video showing multiple results side by side.
        
        Args:
            results: List of simulation results to compare
            labels: Labels for each result
            filename: Output filename
            quality: Video quality
            
        Returns:
            Path to exported video file
        """
        quality_settings = {
            'low': {'bitrate': 1000, 'fps': 15, 'dpi': 72},
            'medium': {'bitrate': 2000, 'fps': 24, 'dpi': 100},
            'high': {'bitrate': 4000, 'fps': 30, 'dpi': 150},
            'ultra': {'bitrate': 8000, 'fps': 60, 'dpi': 200}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        # Setup figure with subplots
        n_results = len(results)
        cols = int(np.ceil(np.sqrt(n_results)))
        rows = int(np.ceil(n_results / cols))
        
        fig = plt.figure(figsize=(5*cols, 4*rows), dpi=settings['dpi'])
        
        axes = []
        lines = []
        points = []
        
        for i, (result, label) in enumerate(zip(results, labels)):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            axes.append(ax)
            
            # Set limits
            ax.set_xlim(np.min(result.x), np.max(result.x))
            ax.set_ylim(np.min(result.y), np.max(result.y))
            ax.set_zlim(np.min(result.z), np.max(result.z))
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(label)
            
            # Initialize plot elements
            line, = ax.plot([], [], [], 'b-', linewidth=0.5, alpha=0.7)
            point, = ax.plot([], [], [], 'ro', markersize=4)
            lines.append(line)
            points.append(point)
        
        # Find maximum length
        max_length = max(len(result.trajectory) for result in results)
        
        def animate(frame_idx):
            """Animation function."""
            for i, (result, line, point) in enumerate(zip(results, lines, points)):
                if frame_idx < len(result.trajectory):
                    # Update trajectory up to current frame
                    line.set_data(result.x[:frame_idx+1], result.y[:frame_idx+1])
                    line.set_3d_properties(result.z[:frame_idx+1])
                    
                    # Update current point
                    point.set_data([result.x[frame_idx]], [result.y[frame_idx]])
                    point.set_3d_properties([result.z[frame_idx]])
            
            return lines + points
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=max_length, 
                           interval=1000/settings['fps'], blit=True)
        
        # Save video
        if filename.endswith('.mp4'):
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        elif filename.endswith('.gif'):
            writer = PillowWriter(fps=settings['fps'])
        else:
            filename = filename + '.mp4'
            writer = FFMpegWriter(fps=settings['fps'], bitrate=settings['bitrate'])
        
        print(f"Exporting comparison video to {filename}...")
        anim.save(filename, writer=writer)
        plt.close(fig)
        
        return filename