"""Stage 1/2 mass-spring-damper target constrained to the contact normal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TargetParameters:
    stiffness_n_per_m: float
    damping_n_s_per_m: float
    effective_mass_kg: float

    def __post_init__(self) -> None:
        values = (
            self.stiffness_n_per_m,
            self.damping_n_s_per_m,
            self.effective_mass_kg,
        )
        if not all(np.isfinite(values)) or self.stiffness_n_per_m <= 0.0:
            raise ValueError("target stiffness must be positive and parameters finite")
        if self.damping_n_s_per_m < 0.0 or self.effective_mass_kg <= 0.0:
            raise ValueError("target damping cannot be negative and mass must be positive")


@dataclass(frozen=True)
class TargetState:
    displacement_m: float = 0.0
    velocity_m_per_s: float = 0.0

    def vector(self) -> NDArray[np.float64]:
        return np.asarray([self.displacement_m, self.velocity_m_per_s], dtype=float)


class Target1D:
    def __init__(self, parameters: TargetParameters) -> None:
        self.parameters = parameters

    def acceleration(self, state: TargetState, applied_force_n: float) -> float:
        p = self.parameters
        return float(
            (applied_force_n - p.damping_n_s_per_m * state.velocity_m_per_s - p.stiffness_n_per_m * state.displacement_m)
            / p.effective_mass_kg
        )

    def derivative(self, vector: ArrayLike, applied_force_n: float) -> NDArray[np.float64]:
        state = TargetState(*np.asarray(vector, dtype=float))
        return np.asarray([state.velocity_m_per_s, self.acceleration(state, applied_force_n)])

    def mechanical_energy_j(self, state: TargetState) -> float:
        p = self.parameters
        return float(
            0.5 * p.effective_mass_kg * state.velocity_m_per_s**2
            + 0.5 * p.stiffness_n_per_m * state.displacement_m**2
        )

    def dissipation_power_w(self, state: TargetState) -> float:
        return float(self.parameters.damping_n_s_per_m * state.velocity_m_per_s**2)
