"""Advanced numerical integration methods for dynamical systems."""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np
from numba import jit

# Type alias for the integrator function signature: f(y, t) -> dy/dt
_IntegratorFunc = Callable[[np.ndarray, float], np.ndarray]


class BaseIntegrator(ABC):
    """Base class for numerical integrators."""

    def __init__(self, dt: float):
        """
        Initialize integrator.

        Args:
            dt: Time step
        """
        self.dt = dt

    @abstractmethod
    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """
        Perform one integration step.

        Args:
            f: Function to integrate (dy/dt = f(y, t))
            y: Current state
            t: Current time

        Returns:
            Next state
        """
        pass

    def integrate(
        self,
        f: _IntegratorFunc,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        num_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system over a time span.

        Args:
            f: Function to integrate
            y0: Initial state
            t_span: Time span (t_start, t_end)
            num_steps: Number of steps (if None, uses dt)

        Returns:
            Tuple of (time_array, state_array)
        """
        t_start, t_end = t_span

        if num_steps is None:
            num_steps = int((t_end - t_start) / self.dt)
        else:
            self.dt = (t_end - t_start) / num_steps

        # Initialize arrays
        t = np.linspace(t_start, t_end, num_steps + 1)
        y = np.zeros((num_steps + 1, len(y0)))
        y[0] = y0

        # Integration loop
        for i in range(num_steps):
            y[i + 1] = self.step(f, y[i], t[i])

        return t, y


class EulerIntegrator(BaseIntegrator):
    """Forward Euler integration method."""

    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """Euler integration step."""
        result: np.ndarray = y + self.dt * f(y, t)
        return result


class RungeKutta4Integrator(BaseIntegrator):
    """Fourth-order Runge-Kutta integration method."""

    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """RK4 integration step."""
        dt = self.dt

        k1 = f(y, t)
        k2 = f(y + dt / 2 * k1, t + dt / 2)
        k3 = f(y + dt / 2 * k2, t + dt / 2)
        k4 = f(y + dt * k3, t + dt)

        result: np.ndarray = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return result


class AdaptiveIntegrator(BaseIntegrator):
    """Adaptive step size integrator using embedded Runge-Kutta methods."""

    def __init__(self, dt: float, rtol: float = 1e-6, atol: float = 1e-9):
        """
        Initialize adaptive integrator.

        Args:
            dt: Initial time step
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        super().__init__(dt)
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt / 1000
        self.dt_max = dt * 10
        self._h_internal = dt

    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """Advance by exactly self.dt using error-controlled adaptive sub-steps."""
        macro_dt = self.dt
        if macro_dt <= 0:
            return y

        t_local = 0.0
        h = min(self._h_internal, macro_dt)
        y_cur = np.asarray(y, dtype=float)

        while t_local < macro_dt - 1e-15:
            h = min(h, macro_dt - t_local)

            # Step-doubling error estimate: one full step vs two half steps.
            y_full = self._rk4_step(f, y_cur, t + t_local, h)
            y_half = self._rk4_step(f, y_cur, t + t_local, h / 2)
            y_half = self._rk4_step(f, y_half, t + t_local + h / 2, h / 2)

            error = np.abs(y_full - y_half)
            tolerance = self.atol + self.rtol * np.abs(y_half)

            if np.all(error <= tolerance) or h <= self.dt_min:
                y_cur = y_half
                t_local += h
                if np.all(error <= tolerance):
                    h = min(h * 1.5, macro_dt)  # grow, but never exceed the macro step
            else:
                h = max(h * 0.5, self.dt_min)

        self._h_internal = h
        return y_cur

    def _rk4_step(
        self, f: _IntegratorFunc, y: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Single RK4 step."""
        k1 = f(y, t)
        k2 = f(y + dt / 2 * k1, t + dt / 2)
        k3 = f(y + dt / 2 * k2, t + dt / 2)
        k4 = f(y + dt * k3, t + dt)

        result: np.ndarray = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return result


class DormandPrince54Integrator(BaseIntegrator):
    """Dormand-Prince 5(4) embedded Runge-Kutta method."""

    def __init__(self, dt: float, rtol: float = 1e-6, atol: float = 1e-9):
        """
        Initialize Dormand-Prince integrator.

        Args:
            dt: Initial time step
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        super().__init__(dt)
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt / 1000
        self.dt_max = dt * 10

        # Dormand-Prince coefficients
        self.a = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            ]
        )

        self.b = np.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        self.b_star = np.array(
            [
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ]
        )
        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
        self._h_internal = dt

    def _dopri_substep(
        self, f: _IntegratorFunc, y: np.ndarray, t: float, h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One Dormand-Prince 5(4) step of size h. Returns (5th-order y, error estimate)."""
        k = np.zeros((7, len(y)))
        k[0] = f(y, t)
        for i in range(1, 7):
            y_temp = y + h * np.sum(self.a[i, :i] * k[:i].T, axis=1)
            k[i] = f(y_temp, t + self.c[i] * h)
        y_new: np.ndarray = y + h * np.sum(self.b * k.T, axis=1)
        y_new_star: np.ndarray = y + h * np.sum(self.b_star * k.T, axis=1)
        error: np.ndarray = np.abs(y_new - y_new_star)
        return y_new, error

    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """Advance by exactly self.dt using error-controlled Dormand-Prince sub-steps."""
        macro_dt = self.dt
        if macro_dt <= 0:
            return y

        t_local = 0.0
        h = min(self._h_internal, macro_dt)
        y_cur = np.asarray(y, dtype=float)

        while t_local < macro_dt - 1e-15:
            h = min(h, macro_dt - t_local)
            y_new, error = self._dopri_substep(f, y_cur, t + t_local, h)
            tolerance = self.atol + self.rtol * np.maximum(np.abs(y_cur), np.abs(y_new))

            if np.all(error <= tolerance) or h <= self.dt_min:
                y_cur = y_new
                t_local += h
                if np.all(error <= tolerance):
                    factor = 0.9 * np.power(np.max(tolerance / (error + 1e-14)), 1 / 5)
                    h = min(
                        h * min(factor, 5.0), macro_dt
                    )  # grow, capped, never exceed macro step
            else:
                h = max(h * 0.5, self.dt_min)

        self._h_internal = h
        return y_cur


@jit(nopython=True)  # type: ignore[untyped-decorator]  # numba ships no stubs
def _rk4_step_numba(f_values: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    """Numba-compiled RK4 step for performance."""
    k1 = f_values[0]
    k2 = f_values[1]
    k3 = f_values[2]
    k4 = f_values[3]

    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type: ignore[no-any-return]


class HighPerformanceRK4Integrator(BaseIntegrator):
    """High-performance RK4 integrator using Numba compilation."""

    def __init__(self, dt: float):
        """Initialize high-performance RK4 integrator."""
        super().__init__(dt)
        self._compiled_step = self._compile_step()

    def _compile_step(self) -> Callable[[_IntegratorFunc, np.ndarray, float], np.ndarray]:
        """Compile the integration step for performance."""
        dt = self.dt

        @jit(nopython=True)  # type: ignore[untyped-decorator]  # numba ships no stubs
        def step_function(f_func: Callable[..., np.ndarray], y: np.ndarray, t: float) -> np.ndarray:  # noqa: E501
            """Compiled RK4 step."""
            k1 = f_func(y)
            k2 = f_func(y + dt / 2 * k1)
            k3 = f_func(y + dt / 2 * k2)
            k4 = f_func(y + dt * k3)

            return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type: ignore[no-any-return]

        return step_function  # type: ignore[no-any-return]

    def step(self, f: _IntegratorFunc, y: np.ndarray, t: float) -> np.ndarray:
        """High-performance RK4 step."""
        # Note: This assumes f doesn't depend on t explicitly
        result: np.ndarray = self._compiled_step(f, y, t)
        return result
