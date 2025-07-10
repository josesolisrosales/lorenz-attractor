"""Tests for parameter management classes."""

import numpy as np
import pytest

from lorenz_attractor.core.parameters import (
    InitialConditions,
    LorenzParameters,
    SimulationConfig,
)


class TestLorenzParameters:
    """Test suite for LorenzParameters class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = LorenzParameters()
        assert params.sigma == 10.0
        assert params.rho == 28.0
        assert params.beta == 8.0 / 3.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = LorenzParameters(sigma=5.0, rho=15.0, beta=2.0)
        assert params.sigma == 5.0
        assert params.rho == 15.0
        assert params.beta == 2.0

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            LorenzParameters(sigma=-1.0)

        with pytest.raises(ValueError, match="rho must be positive"):
            LorenzParameters(rho=-1.0)

        with pytest.raises(ValueError, match="beta must be positive"):
            LorenzParameters(beta=-1.0)

    def test_classical_parameters(self):
        """Test classical parameter factory method."""
        params = LorenzParameters.classical()
        assert params.sigma == 10.0
        assert params.rho == 28.0
        assert params.beta == 8.0 / 3.0

    def test_periodic_parameters(self):
        """Test periodic parameter factory method."""
        params = LorenzParameters.periodic()
        assert params.sigma == 10.0
        assert params.rho == 24.0
        assert params.beta == 8.0 / 3.0

    def test_fixed_point_parameters(self):
        """Test fixed point parameter factory method."""
        params = LorenzParameters.fixed_point()
        assert params.sigma == 10.0
        assert params.rho == 0.5
        assert params.beta == 8.0 / 3.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = LorenzParameters(sigma=5.0, rho=15.0, beta=2.0)
        data = params.to_dict()

        expected = {'sigma': 5.0, 'rho': 15.0, 'beta': 2.0}
        assert data == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {'sigma': 5.0, 'rho': 15.0, 'beta': 2.0}
        params = LorenzParameters.from_dict(data)

        assert params.sigma == 5.0
        assert params.rho == 15.0
        assert params.beta == 2.0


class TestInitialConditions:
    """Test suite for InitialConditions class."""

    def test_default_conditions(self):
        """Test default initial conditions."""
        ic = InitialConditions()
        assert ic.x == 1.0
        assert ic.y == 1.0
        assert ic.z == 1.0

    def test_custom_conditions(self):
        """Test custom initial conditions."""
        ic = InitialConditions(x=2.0, y=3.0, z=4.0)
        assert ic.x == 2.0
        assert ic.y == 3.0
        assert ic.z == 4.0

    def test_to_array(self):
        """Test conversion to numpy array."""
        ic = InitialConditions(x=2.0, y=3.0, z=4.0)
        arr = ic.to_array()

        np.testing.assert_array_equal(arr, [2.0, 3.0, 4.0])
        assert isinstance(arr, np.ndarray)

    def test_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([2.0, 3.0, 4.0])
        ic = InitialConditions.from_array(arr)

        assert ic.x == 2.0
        assert ic.y == 3.0
        assert ic.z == 4.0

    def test_random_with_seed(self):
        """Test random initial conditions with seed."""
        ic1 = InitialConditions.random(scale=1.0, seed=42)
        ic2 = InitialConditions.random(scale=1.0, seed=42)

        # Same seed should produce same results
        assert ic1.x == ic2.x
        assert ic1.y == ic2.y
        assert ic1.z == ic2.z

    def test_random_different_seeds(self):
        """Test random initial conditions with different seeds."""
        ic1 = InitialConditions.random(scale=1.0, seed=42)
        ic2 = InitialConditions.random(scale=1.0, seed=123)

        # Different seeds should produce different results
        assert not (ic1.x == ic2.x and ic1.y == ic2.y and ic1.z == ic2.z)

    def test_random_scale(self):
        """Test random initial conditions with different scales."""
        # Generate many samples to test scaling
        samples = [InitialConditions.random(scale=2.0, seed=i) for i in range(100)]
        values = np.array([[ic.x, ic.y, ic.z] for ic in samples])

        # Standard deviation should be approximately equal to scale
        std_devs = np.std(values, axis=0)
        np.testing.assert_allclose(std_devs, [2.0, 2.0, 2.0], rtol=0.2)

    def test_true_random(self):
        """Test truly random initial conditions."""
        ic1 = InitialConditions.true_random(scale=1.0)
        ic2 = InitialConditions.true_random(scale=1.0)

        # Should be different (very high probability)
        assert not (ic1.x == ic2.x and ic1.y == ic2.y and ic1.z == ic2.z)


class TestSimulationConfig:
    """Test suite for SimulationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.dt == 0.01
        assert config.num_steps == 10000
        assert config.integration_method == "rk4"
        assert config.save_interval == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SimulationConfig(
            dt=0.001, num_steps=50000, integration_method="rk45", save_interval=10
        )
        assert config.dt == 0.001
        assert config.num_steps == 50000
        assert config.integration_method == "rk45"
        assert config.save_interval == 10

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="dt must be positive"):
            SimulationConfig(dt=-0.01)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            SimulationConfig(num_steps=-100)

        with pytest.raises(ValueError, match="save_interval must be positive"):
            SimulationConfig(save_interval=-1)

        with pytest.raises(ValueError, match="integration_method must be one of"):
            SimulationConfig(integration_method="invalid_method")

    def test_total_time_property(self):
        """Test total time calculation."""
        config = SimulationConfig(dt=0.01, num_steps=1000)
        assert config.total_time == 10.0

    def test_time_array_property(self):
        """Test time array generation."""
        config = SimulationConfig(dt=0.1, num_steps=10)
        time_array = config.time_array

        expected = np.arange(0, 1.0, 0.1)
        np.testing.assert_array_almost_equal(time_array, expected)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig(
            dt=0.001, num_steps=50000, integration_method="rk45", save_interval=10
        )
        data = config.to_dict()

        expected = {
            'dt': 0.001,
            'num_steps': 50000,
            'integration_method': 'rk45',
            'save_interval': 10,
        }
        assert data == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'dt': 0.001,
            'num_steps': 50000,
            'integration_method': 'rk45',
            'save_interval': 10,
        }
        config = SimulationConfig.from_dict(data)

        assert config.dt == 0.001
        assert config.num_steps == 50000
        assert config.integration_method == "rk45"
        assert config.save_interval == 10
