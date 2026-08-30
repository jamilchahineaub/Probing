"""Perfect and Gaussian-noise measurements from a known simulation truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from probeing.models import InteractionSimulation


@dataclass(frozen=True)
class MeasurementNoise:
    displacement_std_m: float = 0.0
    velocity_std_m_per_s: float = 0.0
    acceleration_std_m_per_s2: float = 0.0
    force_std_n: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.displacement_std_m,
                self.velocity_std_m_per_s,
                self.acceleration_std_m_per_s2,
                self.force_std_n,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("measurement standard deviations must be finite and non-negative")


@dataclass(frozen=True)
class SyntheticMeasurements:
    time_s: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]
    acceleration_m_per_s2: NDArray[np.float64]
    contact_force_n: NDArray[np.float64]
    noise: MeasurementNoise
    random_seed: int


def generate_measurements(
    truth: InteractionSimulation,
    noise: MeasurementNoise,
    *,
    random_seed: int,
) -> SyntheticMeasurements:
    """Add independent, zero-mean Gaussian noise to each simulated channel."""

    if not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    generator = np.random.default_rng(int(random_seed))

    def noisy(values: NDArray[np.float64], standard_deviation: float) -> NDArray[np.float64]:
        if standard_deviation == 0.0:
            return values.copy()
        return values + generator.normal(0.0, standard_deviation, size=values.shape)

    return SyntheticMeasurements(
        time_s=truth.time_s.copy(),
        displacement_m=noisy(truth.displacement_m, noise.displacement_std_m),
        velocity_m_per_s=noisy(truth.velocity_m_per_s, noise.velocity_std_m_per_s),
        acceleration_m_per_s2=noisy(
            truth.acceleration_m_per_s2, noise.acceleration_std_m_per_s2
        ),
        contact_force_n=noisy(truth.applied_force_n, noise.force_std_n),
        noise=noise,
        random_seed=int(random_seed),
    )
