"""Parameter management for Lorenz system simulations."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class LorenzParameters:
    """Parameters for the Lorenz system."""
    
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0/3.0
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
    
    @classmethod
    def classical(cls) -> 'LorenzParameters':
        """Returns classical Lorenz parameters (chaotic regime)."""
        return cls(sigma=10.0, rho=28.0, beta=8.0/3.0)
    
    @classmethod
    def periodic(cls) -> 'LorenzParameters':
        """Returns parameters for periodic behavior."""
        return cls(sigma=10.0, rho=24.0, beta=8.0/3.0)
    
    @classmethod
    def fixed_point(cls) -> 'LorenzParameters':
        """Returns parameters for fixed point behavior."""
        return cls(sigma=10.0, rho=0.5, beta=8.0/3.0)
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            'sigma': self.sigma,
            'rho': self.rho,
            'beta': self.beta
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LorenzParameters':
        """Create parameters from dictionary."""
        return cls(**data)


@dataclass
class InitialConditions:
    """Initial conditions for the Lorenz system."""
    
    x: float = 1.0
    y: float = 1.0
    z: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'InitialConditions':
        """Create from numpy array."""
        return cls(x=arr[0], y=arr[1], z=arr[2])
    
    @classmethod
    def random(cls, scale: float = 1.0, seed: Optional[int] = None) -> 'InitialConditions':
        """Generate random initial conditions."""
        if seed is not None:
            np.random.seed(seed)
        
        x, y, z = np.random.normal(0, scale, 3)
        return cls(x=x, y=y, z=z)
    
    @classmethod
    def true_random(cls, scale: float = 1.0) -> 'InitialConditions':
        """Generate truly random initial conditions using system entropy."""
        import os
        
        # Use system entropy for true randomness
        entropy = os.urandom(12)
        seed = int.from_bytes(entropy, 'big')
        
        np.random.seed(seed)
        x, y, z = np.random.normal(0, scale, 3)
        return cls(x=x, y=y, z=z)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    
    dt: float = 0.01
    num_steps: int = 10000
    integration_method: str = "rk4"
    save_interval: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        
        valid_methods = ["euler", "rk4", "rk45", "dopri5"]
        if self.integration_method not in valid_methods:
            raise ValueError(f"integration_method must be one of {valid_methods}")
    
    @property
    def total_time(self) -> float:
        """Total simulation time."""
        return self.dt * self.num_steps
    
    @property
    def time_array(self) -> np.ndarray:
        """Time array for the simulation."""
        return np.arange(0, self.total_time, self.dt)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'dt': self.dt,
            'num_steps': self.num_steps,
            'integration_method': self.integration_method,
            'save_interval': self.save_interval
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationConfig':
        """Create configuration from dictionary."""
        return cls(**data)