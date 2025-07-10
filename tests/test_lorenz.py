"""Tests for the core Lorenz system implementation."""

import numpy as np
import pytest

from lorenz_attractor.core.lorenz import LorenzSystem
from lorenz_attractor.core.parameters import InitialConditions, LorenzParameters


class TestLorenzSystem:
    """Test suite for LorenzSystem class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        system = LorenzSystem()
        assert system.parameters.sigma == 10.0
        assert system.parameters.rho == 28.0
        assert system.parameters.beta == 8.0 / 3.0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        params = LorenzParameters(sigma=5.0, rho=15.0, beta=2.0)
        system = LorenzSystem(params)
        assert system.parameters.sigma == 5.0
        assert system.parameters.rho == 15.0
        assert system.parameters.beta == 2.0

    def test_derivative_computation(self):
        """Test derivative computation."""
        system = LorenzSystem()
        state = np.array([1.0, 1.0, 1.0])

        derivative = system.derivative(state)

        # Expected derivatives for classical parameters at [1, 1, 1]
        expected_dx = 10.0 * (1.0 - 1.0)  # 0.0
        expected_dy = 1.0 * (28.0 - 1.0) - 1.0  # 26.0
        expected_dz = 1.0 * 1.0 - (8.0 / 3.0) * 1.0  # 1 - 8/3 = -5/3

        assert abs(derivative[0] - expected_dx) < 1e-10
        assert abs(derivative[1] - expected_dy) < 1e-10
        assert abs(derivative[2] - expected_dz) < 1e-10

    def test_derivative_with_tuple_input(self):
        """Test derivative computation with tuple input."""
        system = LorenzSystem()
        state = (1.0, 1.0, 1.0)

        derivative = system.derivative(state)
        assert isinstance(derivative, np.ndarray)
        assert len(derivative) == 3

    def test_jacobian_computation(self):
        """Test Jacobian matrix computation."""
        system = LorenzSystem()
        state = np.array([1.0, 1.0, 1.0])

        jacobian = system.jacobian(state)

        # Expected Jacobian for classical parameters at [1, 1, 1]
        expected = np.array(
            [[-10.0, 10.0, 0.0], [27.0, -1.0, -1.0], [1.0, 1.0, -8.0 / 3.0]]
        )

        np.testing.assert_allclose(jacobian, expected, rtol=1e-10)

    def test_equilibrium_points_subcritical(self):
        """Test equilibrium points for subcritical case (rho < 1)."""
        params = LorenzParameters(sigma=10.0, rho=0.5, beta=8.0 / 3.0)
        system = LorenzSystem(params)

        equilibria = system.equilibrium_points()

        # Only origin should exist for rho < 1
        assert len(equilibria) == 1
        np.testing.assert_allclose(equilibria[0], [0.0, 0.0, 0.0])

    def test_equilibrium_points_supercritical(self):
        """Test equilibrium points for supercritical case (rho > 1)."""
        system = LorenzSystem()  # Classical parameters with rho = 28

        equilibria = system.equilibrium_points()

        # Should have three equilibria: origin and two symmetric points
        assert len(equilibria) == 3

        # Check origin
        np.testing.assert_allclose(equilibria[0], [0.0, 0.0, 0.0])

        # Check symmetric equilibria
        beta, rho = 8.0 / 3.0, 28.0
        sqrt_term = np.sqrt(beta * (rho - 1))

        expected_positive = np.array([sqrt_term, sqrt_term, rho - 1])
        expected_negative = np.array([-sqrt_term, -sqrt_term, rho - 1])

        # Find which equilibrium corresponds to which
        for eq in equilibria[1:]:
            if eq[0] > 0:
                np.testing.assert_allclose(eq, expected_positive, rtol=1e-10)
            else:
                np.testing.assert_allclose(eq, expected_negative, rtol=1e-10)

    def test_poincare_section_empty(self):
        """Test Poincaré section with trajectory that doesn't cross plane."""
        system = LorenzSystem()

        # Simple trajectory that doesn't cross z=27 plane
        trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        intersections = system.poincare_section(trajectory)

        assert len(intersections) == 0
        assert intersections.shape == (0, 3)

    def test_poincare_section_with_crossing(self):
        """Test Poincaré section with trajectory that crosses plane."""
        system = LorenzSystem()

        # Trajectory that crosses z=27 plane twice
        trajectory = np.array([[0, 0, 26], [1, 1, 28], [2, 2, 26]])

        intersections = system.poincare_section(trajectory, plane_offset=27.0)

        # Should have two intersections (going up and coming down)
        assert len(intersections) == 2
        # Both intersections should be approximately at z=27
        assert abs(intersections[0][2] - 27.0) < 1e-10
        assert abs(intersections[1][2] - 27.0) < 1e-10

    def test_update_parameters(self):
        """Test parameter updating."""
        system = LorenzSystem()
        original_params = system.parameters

        new_params = LorenzParameters(sigma=5.0, rho=15.0, beta=2.0)
        system.update_parameters(new_params)

        assert system.parameters != original_params
        assert system.parameters.sigma == 5.0
        assert system.parameters.rho == 15.0
        assert system.parameters.beta == 2.0

    def test_string_representation(self):
        """Test string representations."""
        system = LorenzSystem()

        str_repr = str(system)
        repr_str = repr(system)

        assert "LorenzSystem" in str_repr
        assert "σ=10.000" in str_repr
        assert "ρ=28.000" in str_repr
        assert "β=2.667" in str_repr

        assert str_repr == repr_str

    @pytest.mark.slow
    def test_lyapunov_exponents(self):
        """Test Lyapunov exponent calculation (slow test)."""
        system = LorenzSystem()
        initial_conditions = InitialConditions(x=1.0, y=1.0, z=1.0)

        # Use small number of steps for testing
        exponents = system.lyapunov_exponents(
            initial_conditions, dt=0.01, num_steps=1000
        )

        # Should return 3 exponents
        assert len(exponents) == 3

        # For chaotic Lorenz system, expect one positive, one zero, one negative
        assert exponents[0] > 0  # Largest should be positive
        assert exponents[2] < 0  # Smallest should be negative

        # Sum should be negative (dissipative system)
        assert np.sum(exponents) < 0
