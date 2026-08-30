"""One-dimensional force-driven mass-spring-damper interaction model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class InteractionParameters:
    """Ground-truth parameters for ``F = m*x_ddot + c*x_dot + k*x``."""

    stiffness_n_per_m: float
    damping_n_s_per_m: float
    effective_mass_kg: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.stiffness_n_per_m,
                self.damping_n_s_per_m,
                self.effective_mass_kg,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("interaction parameters must be finite")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("stiffness_n_per_m must be strictly positive")
        if self.damping_n_s_per_m < 0.0:
            raise ValueError("damping_n_s_per_m must be non-negative")
        if self.effective_mass_kg <= 0.0:
            raise ValueError("effective_mass_kg must be strictly positive")


@dataclass(frozen=True)
class InteractionSimulation:
    """Time history returned by the numerical or analytical simulator."""

    time_s: NDArray[np.float64]
    applied_force_n: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]
    acceleration_m_per_s2: NDArray[np.float64]
    kinetic_energy_j: NDArray[np.float64]
    elastic_energy_j: NDArray[np.float64]

    @property
    def mechanical_energy_j(self) -> NDArray[np.float64]:
        return self.kinetic_energy_j + self.elastic_energy_j


class MassSpringDamperModel:
    """Integrate a linear one-dimensional target driven by applied force."""

    def __init__(self, parameters: InteractionParameters) -> None:
        self.parameters = parameters

    def acceleration(
        self, displacement_m: ArrayLike, velocity_m_per_s: ArrayLike, force_n: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate acceleration from the governing equation."""

        displacement, velocity, force = np.broadcast_arrays(
            np.asarray(displacement_m, dtype=float),
            np.asarray(velocity_m_per_s, dtype=float),
            np.asarray(force_n, dtype=float),
        )
        if not all(np.all(np.isfinite(value)) for value in (displacement, velocity, force)):
            raise ValueError("state and force samples must be finite")
        p = self.parameters
        return np.asarray(
            (force - p.damping_n_s_per_m * velocity - p.stiffness_n_per_m * displacement)
            / p.effective_mass_kg,
            dtype=float,
        )

    @staticmethod
    def _validated_history(
        time_s: ArrayLike, force_n: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        time = np.asarray(time_s, dtype=float)
        force = np.asarray(force_n, dtype=float)
        if time.ndim != 1 or force.ndim != 1 or time.shape != force.shape:
            raise ValueError("time_s and force_n must be one-dimensional with equal shape")
        if time.size < 2 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
            raise ValueError("time_s must contain at least two finite increasing samples")
        if not np.all(np.isfinite(force)):
            raise ValueError("force_n samples must be finite")
        return time, force

    def _make_result(
        self,
        time: NDArray[np.float64],
        force: NDArray[np.float64],
        displacement: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> InteractionSimulation:
        p = self.parameters
        acceleration = self.acceleration(displacement, velocity, force)
        return InteractionSimulation(
            time_s=time,
            applied_force_n=force,
            displacement_m=displacement,
            velocity_m_per_s=velocity,
            acceleration_m_per_s2=acceleration,
            kinetic_energy_j=0.5 * p.effective_mass_kg * velocity**2,
            elastic_energy_j=0.5 * p.stiffness_n_per_m * displacement**2,
        )

    def simulate(
        self,
        time_s: ArrayLike,
        force_n: ArrayLike,
        *,
        initial_displacement_m: float = 0.0,
        initial_velocity_m_per_s: float = 0.0,
    ) -> InteractionSimulation:
        """Integrate the target with fourth-order Runge-Kutta.

        Force is linearly interpolated between samples, including at each RK4
        midpoint. This makes the input convention explicit and gives fourth-order
        state convergence for smooth sampled probes.
        """

        time, force = self._validated_history(time_s, force_n)
        initial = np.asarray(
            [initial_displacement_m, initial_velocity_m_per_s], dtype=float
        )
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial state must be finite")

        displacement = np.empty_like(time)
        velocity = np.empty_like(time)
        displacement[0], velocity[0] = initial

        def derivative(state: NDArray[np.float64], input_force: float) -> NDArray[np.float64]:
            return np.asarray(
                [state[1], float(self.acceleration(state[0], state[1], input_force))],
                dtype=float,
            )

        for index in range(time.size - 1):
            step = time[index + 1] - time[index]
            force_start = float(force[index])
            force_end = float(force[index + 1])
            force_mid = 0.5 * (force_start + force_end)
            state = np.asarray([displacement[index], velocity[index]], dtype=float)
            k1 = derivative(state, force_start)
            k2 = derivative(state + 0.5 * step * k1, force_mid)
            k3 = derivative(state + 0.5 * step * k2, force_mid)
            k4 = derivative(state + step * k3, force_end)
            next_state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            displacement[index + 1], velocity[index + 1] = next_state

        return self._make_result(time, force, displacement, velocity)

    def analytical_free_response(
        self,
        time_s: ArrayLike,
        *,
        initial_displacement_m: float,
        initial_velocity_m_per_s: float,
    ) -> InteractionSimulation:
        """Return the exact unforced response for all linear damping regimes."""

        time = np.asarray(time_s, dtype=float)
        if time.ndim != 1 or time.size < 1 or not np.all(np.isfinite(time)):
            raise ValueError("time_s must be a non-empty finite one-dimensional array")
        if not np.all(time >= 0.0) or not np.all(np.diff(time) > 0.0):
            raise ValueError("time_s must be non-negative and strictly increasing")
        x0 = float(initial_displacement_m)
        v0 = float(initial_velocity_m_per_s)
        if not np.all(np.isfinite([x0, v0])):
            raise ValueError("initial state must be finite")

        p = self.parameters
        natural_frequency = np.sqrt(p.stiffness_n_per_m / p.effective_mass_kg)
        decay = p.damping_n_s_per_m / (2.0 * p.effective_mass_kg)
        discriminant = decay**2 - natural_frequency**2
        tolerance = 64.0 * np.finfo(float).eps * max(
            decay**2, natural_frequency**2, 1.0
        )

        if discriminant < -tolerance:
            damped_frequency = np.sqrt(-discriminant)
            sine_coefficient = (v0 + decay * x0) / damped_frequency
            exponential = np.exp(-decay * time)
            cosine = np.cos(damped_frequency * time)
            sine = np.sin(damped_frequency * time)
            displacement = exponential * (x0 * cosine + sine_coefficient * sine)
            velocity = exponential * (
                -decay * (x0 * cosine + sine_coefficient * sine)
                - x0 * damped_frequency * sine
                + sine_coefficient * damped_frequency * cosine
            )
        elif discriminant > tolerance:
            root_offset = np.sqrt(discriminant)
            root_one = -decay + root_offset
            root_two = -decay - root_offset
            coefficient_one = (v0 - root_two * x0) / (root_one - root_two)
            coefficient_two = (root_one * x0 - v0) / (root_one - root_two)
            term_one = coefficient_one * np.exp(root_one * time)
            term_two = coefficient_two * np.exp(root_two * time)
            displacement = term_one + term_two
            velocity = root_one * term_one + root_two * term_two
        else:
            coefficient = v0 + decay * x0
            exponential = np.exp(-decay * time)
            displacement = (x0 + coefficient * time) * exponential
            velocity = (coefficient - decay * (x0 + coefficient * time)) * exponential

        force = np.zeros_like(time)
        return self._make_result(time, force, displacement, velocity)

    def analytical_step_response(
        self,
        time_s: ArrayLike,
        *,
        force_n: float,
        initial_displacement_m: float = 0.0,
        initial_velocity_m_per_s: float = 0.0,
    ) -> InteractionSimulation:
        """Return the exact response to a constant force applied at ``t=0``."""

        if not np.isfinite(force_n):
            raise ValueError("force_n must be finite")
        equilibrium = force_n / self.parameters.stiffness_n_per_m
        shifted = self.analytical_free_response(
            time_s,
            initial_displacement_m=initial_displacement_m - equilibrium,
            initial_velocity_m_per_s=initial_velocity_m_per_s,
        )
        displacement = shifted.displacement_m + equilibrium
        force = np.full_like(shifted.time_s, force_n)
        return self._make_result(
            shifted.time_s, force, displacement, shifted.velocity_m_per_s
        )
