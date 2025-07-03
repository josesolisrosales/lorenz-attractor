"""Core Lorenz system implementation with advanced features."""

import numpy as np
from numba import jit
from typing import Tuple, Optional, Union
from .parameters import LorenzParameters, InitialConditions


class LorenzSystem:
    """
    Professional implementation of the Lorenz dynamical system.
    
    The Lorenz system is a system of ordinary differential equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    
    Where σ, ρ, and β are system parameters.
    """
    
    def __init__(self, parameters: Optional[LorenzParameters] = None):
        """
        Initialize the Lorenz system.
        
        Args:
            parameters: System parameters. If None, uses classical values.
        """
        self.parameters = parameters or LorenzParameters.classical()
        self._compiled_derivative = self._compile_derivative()
    
    def _compile_derivative(self):
        """Compile the derivative function for performance."""
        sigma, rho, beta = self.parameters.sigma, self.parameters.rho, self.parameters.beta
        
        @jit(nopython=True)
        def derivative(state: np.ndarray) -> np.ndarray:
            """Compute derivatives of the Lorenz system."""
            x, y, z = state
            
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z
            
            return np.array([dx_dt, dy_dt, dz_dt])
        
        return derivative
    
    def derivative(self, state: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute the derivative of the system state.
        
        Args:
            state: Current state [x, y, z] or (x, y, z)
            
        Returns:
            Derivative [dx/dt, dy/dt, dz/dt]
        """
        if isinstance(state, tuple):
            state = np.array(state)
        
        return self._compiled_derivative(state)
    
    def jacobian(self, state: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute the Jacobian matrix of the system at a given state.
        
        Args:
            state: Current state [x, y, z] or (x, y, z)
            
        Returns:
            3x3 Jacobian matrix
        """
        if isinstance(state, tuple):
            state = np.array(state)
        
        x, y, z = state
        sigma, rho, beta = self.parameters.sigma, self.parameters.rho, self.parameters.beta
        
        J = np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
        
        return J
    
    def equilibrium_points(self) -> list:
        """
        Calculate equilibrium points of the system.
        
        Returns:
            List of equilibrium points as numpy arrays
        """
        sigma, rho, beta = self.parameters.sigma, self.parameters.rho, self.parameters.beta
        
        # Origin is always an equilibrium point
        equilibria = [np.array([0.0, 0.0, 0.0])]
        
        # For rho > 1, there are two additional equilibria
        if rho > 1:
            sqrt_term = np.sqrt(beta * (rho - 1))
            
            # C+ equilibrium
            equilibria.append(np.array([sqrt_term, sqrt_term, rho - 1]))
            
            # C- equilibrium  
            equilibria.append(np.array([-sqrt_term, -sqrt_term, rho - 1]))
        
        return equilibria
    
    def lyapunov_exponents(self, initial_conditions: InitialConditions, 
                          dt: float = 0.01, num_steps: int = 100000) -> np.ndarray:
        """
        Estimate Lyapunov exponents using the method of Benettin et al.
        
        Args:
            initial_conditions: Initial conditions for the trajectory
            dt: Time step
            num_steps: Number of integration steps
            
        Returns:
            Array of three Lyapunov exponents
        """
        # This is a simplified implementation
        # For production use, consider using more sophisticated methods
        
        from scipy.integrate import solve_ivp
        
        def extended_system(t, y):
            """Extended system for Lyapunov exponent calculation."""
            state = y[:3]
            tangent_vectors = y[3:].reshape(3, 3)
            
            # System derivative
            f = self.derivative(state)
            
            # Jacobian
            J = self.jacobian(state)
            
            # Tangent vector derivatives
            tangent_derivatives = J @ tangent_vectors
            
            return np.concatenate([f, tangent_derivatives.flatten()])
        
        # Initial conditions: state + identity matrix for tangent vectors
        y0 = np.concatenate([
            initial_conditions.to_array(),
            np.eye(3).flatten()
        ])
        
        # Integration
        t_span = (0, num_steps * dt)
        t_eval = np.arange(0, num_steps * dt, dt)
        
        sol = solve_ivp(extended_system, t_span, y0, t_eval=t_eval, 
                       method='RK45', rtol=1e-8)
        
        # Extract and orthogonalize tangent vectors
        lyap_sums = np.zeros(3)
        
        for i in range(1, len(sol.t)):
            tangent_matrix = sol.y[3:, i].reshape(3, 3)
            
            # QR decomposition for orthogonalization
            Q, R = np.linalg.qr(tangent_matrix)
            
            # Accumulate logarithms of diagonal elements
            lyap_sums += np.log(np.abs(np.diag(R)))
        
        # Calculate average growth rates
        lyapunov_exponents = lyap_sums / (sol.t[-1])
        
        return np.sort(lyapunov_exponents)[::-1]  # Sort in descending order
    
    def poincare_section(self, trajectory: np.ndarray, 
                        plane_normal: np.ndarray = np.array([0, 0, 1]),
                        plane_offset: float = 27.0) -> np.ndarray:
        """
        Compute Poincaré section of the trajectory.
        
        Args:
            trajectory: Trajectory array of shape (n_points, 3)
            plane_normal: Normal vector to the Poincaré plane
            plane_offset: Offset of the plane from origin
            
        Returns:
            Array of intersection points
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        intersections = []
        
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            
            # Check if trajectory crosses the plane
            d1 = np.dot(p1, plane_normal) - plane_offset
            d2 = np.dot(p2, plane_normal) - plane_offset
            
            if d1 * d2 < 0:  # Sign change indicates crossing
                # Linear interpolation to find intersection
                t = -d1 / (d2 - d1)
                intersection = p1 + t * (p2 - p1)
                intersections.append(intersection)
        
        return np.array(intersections) if intersections else np.array([]).reshape(0, 3)
    
    def update_parameters(self, new_parameters: LorenzParameters):
        """
        Update system parameters and recompile derivative function.
        
        Args:
            new_parameters: New system parameters
        """
        self.parameters = new_parameters
        self._compiled_derivative = self._compile_derivative()
    
    def __repr__(self) -> str:
        """String representation of the system."""
        return (f"LorenzSystem(σ={self.parameters.sigma:.3f}, "
                f"ρ={self.parameters.rho:.3f}, β={self.parameters.beta:.3f})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()