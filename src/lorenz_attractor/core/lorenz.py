"""Core Lorenz system implementation with advanced features."""

from typing import Optional, Tuple, Union

import numpy as np
from numba import jit

from .parameters import InitialConditions, LorenzParameters


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
        sigma, rho, beta = (
            self.parameters.sigma,
            self.parameters.rho,
            self.parameters.beta,
        )

        @jit(nopython=True)
        def derivative(state: np.ndarray) -> np.ndarray:
            """Compute derivatives of the Lorenz system."""
            x, y, z = state

            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z

            return np.array([dx_dt, dy_dt, dz_dt])

        return derivative

    def derivative(
        self, state: Union[np.ndarray, Tuple[float, float, float]]
    ) -> np.ndarray:
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

    def jacobian(
        self, state: Union[np.ndarray, Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix of the system at a given state.

        Args:
            state: Current state [x, y, z] or (x, y, z)

        Returns:
            3x3 Jacobian matrix
        """
        from ..analysis import stability

        return stability.jacobian(self, state)

    def equilibrium_points(self) -> list:
        """
        Calculate equilibrium points of the system.

        Returns:
            List of equilibrium points as numpy arrays
        """
        from ..analysis import stability

        return stability.equilibrium_points(self)

    def lyapunov_exponents(
        self,
        initial_conditions: InitialConditions,
        dt: float = 0.01,
        num_steps: int = 100000,
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents using the method of Benettin et al.

        Args:
            initial_conditions: Initial conditions for the trajectory
            dt: Time step
            num_steps: Number of integration steps

        Returns:
            Array of three Lyapunov exponents
        """
        from ..analysis import stability

        return stability.lyapunov_exponents(self, initial_conditions, dt, num_steps)

    def poincare_section(
        self,
        trajectory: np.ndarray,
        plane_normal: np.ndarray = np.array([0, 0, 1]),
        plane_offset: float = 27.0,
    ) -> np.ndarray:
        """
        Compute Poincaré section of the trajectory.

        Args:
            trajectory: Trajectory array of shape (n_points, 3)
            plane_normal: Normal vector to the Poincaré plane
            plane_offset: Offset of the plane from origin

        Returns:
            Array of intersection points
        """
        from ..analysis import sections

        return sections.poincare_section(trajectory, plane_normal, plane_offset)

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
        return (
            f"LorenzSystem(σ={self.parameters.sigma:.3f}, "
            f"ρ={self.parameters.rho:.3f}, β={self.parameters.beta:.3f})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
