"""Image export capabilities."""

import matplotlib.pyplot as plt
from typing import List, Optional
from ..core.simulator import SimulationResult
from ..visualization.plotter import LorenzPlotter


class ImageExporter:
    """Export high-quality images of simulations."""
    
    def __init__(self, dpi: int = 300):
        """Initialize image exporter."""
        self.dpi = dpi
        self.plotter = LorenzPlotter(dpi=dpi)
    
    def export_3d_plot(self, result: SimulationResult, filename: str) -> str:
        """Export 3D trajectory plot."""
        fig = self.plotter.plot_3d_trajectory(result)
        fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return filename
    
    def export_2d_projections(self, result: SimulationResult, filename: str) -> str:
        """Export 2D projections plot."""
        fig = self.plotter.plot_2d_projections(result)
        fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return filename