"""Tests for analysis.stability (extracted from core.lorenz)."""

import numpy as np
import pytest

from lorenz_attractor.analysis import stability
from lorenz_attractor.core.lorenz import LorenzSystem
from lorenz_attractor.core.parameters import InitialConditions, LorenzParameters


def test_jacobian_matches_known_matrix():
    system = LorenzSystem()
    J = stability.jacobian(system, np.array([1.0, 1.0, 1.0]))
    expected = np.array(
        [[-10.0, 10.0, 0.0], [27.0, -1.0, -1.0], [1.0, 1.0, -8.0 / 3.0]]
    )
    np.testing.assert_allclose(J, expected, rtol=1e-10)


def test_equilibria_supercritical_has_three_points():
    system = LorenzSystem()  # rho = 28 > 1
    eq = stability.equilibrium_points(system)
    assert len(eq) == 3
    np.testing.assert_allclose(eq[0], [0.0, 0.0, 0.0])


def test_equilibria_subcritical_has_origin_only():
    system = LorenzSystem(LorenzParameters(sigma=10.0, rho=0.5, beta=8.0 / 3.0))
    assert len(stability.equilibrium_points(system)) == 1


def test_lorenzsystem_methods_delegate_identically():
    system = LorenzSystem()
    state = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        system.jacobian(state), stability.jacobian(system, state)
    )
    eqs_method = system.equilibrium_points()
    eqs_func = stability.equilibrium_points(system)
    for a, b in zip(eqs_method, eqs_func):
        np.testing.assert_array_equal(a, b)
    # Verify lyapunov_exponents delegates identically
    initial_conditions = InitialConditions(1.0, 1.0, 1.0)
    lyap_method = system.lyapunov_exponents(initial_conditions, dt=0.01, num_steps=200)
    lyap_func = stability.lyapunov_exponents(
        system, initial_conditions, dt=0.01, num_steps=200
    )
    np.testing.assert_allclose(lyap_method, lyap_func, rtol=1e-10)


@pytest.mark.slow
def test_lyapunov_largest_exponent_positive_for_chaos():
    system = LorenzSystem()
    exps = stability.lyapunov_exponents(
        system, InitialConditions(1.0, 1.0, 1.0), dt=0.01, num_steps=1000
    )
    assert len(exps) == 3
    assert exps[0] > 0
    assert exps[2] < 0
    assert np.sum(exps) < 0
