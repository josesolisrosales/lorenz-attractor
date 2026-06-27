"""Numerical sanity tests for the integrators on dy/dt = -y (y(t)=e^-t)."""

import numpy as np
import pytest

from lorenz_attractor.integration.integrators import (
    AdaptiveIntegrator,
    DormandPrince54Integrator,
    EulerIntegrator,
    RungeKutta4Integrator,
)


def _decay(y, t):
    return -y


def test_rk4_more_accurate_than_euler():
    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    _, y_euler = EulerIntegrator(0.1).integrate(_decay, y0, t_span, num_steps=100)
    _, y_rk4 = RungeKutta4Integrator(0.1).integrate(_decay, y0, t_span, num_steps=100)
    exact = np.exp(-1.0)
    err_euler = abs(y_euler[-1, 0] - exact)
    err_rk4 = abs(y_rk4[-1, 0] - exact)
    assert err_rk4 < err_euler


def test_rk4_error_shrinks_with_smaller_dt():
    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    exact = np.exp(-1.0)
    _, y_coarse = RungeKutta4Integrator(0.1).integrate(_decay, y0, t_span, num_steps=10)
    _, y_fine = RungeKutta4Integrator(0.1).integrate(_decay, y0, t_span, num_steps=100)
    assert abs(y_fine[-1, 0] - exact) < abs(y_coarse[-1, 0] - exact)


@pytest.mark.parametrize(
    'integrator', [RungeKutta4Integrator, AdaptiveIntegrator, DormandPrince54Integrator]
)
def test_integrators_agree_on_decay(integrator):
    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    _, y = integrator(0.05).integrate(_decay, y0, t_span, num_steps=200)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-2


def test_integrate_returns_consistent_shapes():
    y0 = np.array([1.0, 2.0, 3.0])
    t, y = RungeKutta4Integrator(0.01).integrate(_decay, y0, (0.0, 1.0), num_steps=50)
    assert t.shape[0] == 51
    assert y.shape == (51, 3)
