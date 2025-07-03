"""Interactive visualization components."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ..core.simulator import Simulator
from ..core.lorenz import LorenzSystem
from ..core.parameters import LorenzParameters


class InteractiveVisualizer:
    """Interactive 3D visualization with parameter controls."""
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.simulator = None
        self.fig = None
        self.ax = None
        self.line = None
        
    def create_interactive_plot(self, steps=10000, dt=0.01):
        """Create interactive plot with parameter sliders."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial simulation
        params = LorenzParameters(sigma=self.sigma, rho=self.rho, beta=self.beta)
        system = LorenzSystem(params)
        self.simulator = Simulator(system)
        
        from ..core.parameters import InitialConditions, SimulationConfig
        initial_conditions = InitialConditions(1.0, 1.0, 1.0)
        config = SimulationConfig(dt=dt, num_steps=steps)
        
        result = self.simulator.simulate(initial_conditions, config)
        trajectory = result.trajectory
        
        self.line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Interactive Lorenz Attractor')
        
        # Add sliders
        plt.subplots_adjust(bottom=0.25)
        
        ax_sigma = plt.axes([0.1, 0.1, 0.3, 0.03])
        ax_rho = plt.axes([0.1, 0.05, 0.3, 0.03])
        ax_beta = plt.axes([0.1, 0.0, 0.3, 0.03])
        
        slider_sigma = Slider(ax_sigma, 'Sigma', 1.0, 50.0, valinit=self.sigma)
        slider_rho = Slider(ax_rho, 'Rho', 1.0, 50.0, valinit=self.rho)
        slider_beta = Slider(ax_beta, 'Beta', 0.1, 5.0, valinit=self.beta)
        
        def update(val):
            self.sigma = slider_sigma.val
            self.rho = slider_rho.val
            self.beta = slider_beta.val
            
            params = LorenzParameters(sigma=self.sigma, rho=self.rho, beta=self.beta)
            system = LorenzSystem(params)
            self.simulator = Simulator(system)
            
            result = self.simulator.simulate(initial_conditions, config)
            new_trajectory = result.trajectory
            
            self.line.set_data_3d(new_trajectory[:, 0], new_trajectory[:, 1], new_trajectory[:, 2])
            self.fig.canvas.draw()
        
        slider_sigma.on_changed(update)
        slider_rho.on_changed(update)
        slider_beta.on_changed(update)
        
        plt.show()
        
    def show(self):
        """Show the interactive visualization."""
        self.create_interactive_plot()