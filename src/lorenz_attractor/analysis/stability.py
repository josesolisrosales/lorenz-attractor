"""Stability analysis for the Lorenz system: Jacobian, equilibria, Lyapunov spectrum.

References:
- Benettin, G. et al. (1980). "Lyapunov characteristic exponents for smooth
  dynamical systems." Meccanica 15, 9-30.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

from ..core.parameters import InitialConditions

if TYPE_CHECKING:
    from ..core.lorenz import LorenzSystem


def jacobian(
    system: 'LorenzSystem',
    state: Union[np.ndarray, Tuple[float, float, float]],
) -> np.ndarray:
    """Compute the 3x3 Jacobian of the Lorenz system at ``state``."""
    if isinstance(state, tuple):
        state = np.array(state)
    x, y, z = state
    sigma, rho, beta = (
        system.parameters.sigma,
        system.parameters.rho,
        system.parameters.beta,
    )
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


def equilibrium_points(system: 'LorenzSystem') -> List[np.ndarray]:
    """Return the equilibrium points: origin always, plus C+/C- when rho > 1."""
    rho, beta = system.parameters.rho, system.parameters.beta
    equilibria: List[np.ndarray] = [np.array([0.0, 0.0, 0.0])]
    if rho > 1:
        sqrt_term = np.sqrt(beta * (rho - 1))
        equilibria.append(np.array([sqrt_term, sqrt_term, rho - 1]))
        equilibria.append(np.array([-sqrt_term, -sqrt_term, rho - 1]))
    return equilibria


def lyapunov_exponents(
    system: 'LorenzSystem',
    initial_conditions: InitialConditions,
    dt: float = 0.01,
    num_steps: int = 100000,
) -> np.ndarray:
    """Estimate the Lyapunov spectrum via the Benettin tangent-space method."""
    from scipy.integrate import solve_ivp

    def extended_system(t: float, y: np.ndarray) -> np.ndarray:
        state = y[:3]
        tangent_vectors = y[3:].reshape(3, 3)
        f = system.derivative(state)
        J = system.jacobian(state)
        tangent_derivatives = J @ tangent_vectors
        return np.concatenate([f, tangent_derivatives.flatten()])

    y0 = np.concatenate([initial_conditions.to_array(), np.eye(3).flatten()])
    t_span = (0, num_steps * dt)
    t_eval = np.arange(0, num_steps * dt, dt)

    sol = solve_ivp(
        extended_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
    )

    lyap_sums = np.zeros(3)
    for i in range(1, len(sol.t)):
        tangent_matrix = sol.y[3:, i].reshape(3, 3)
        Q, R = np.linalg.qr(tangent_matrix)
        lyap_sums += np.log(np.abs(np.diag(R)))

    exponents = lyap_sums / (sol.t[-1])
    return np.sort(exponents)[::-1]
