"""Deterministic probe trajectories with analytical derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProbeTrajectory:
    time_s: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]
    acceleration_m_per_s2: NDArray[np.float64]


def _integer_intervals(duration_s: float, sample_period_s: float, name: str) -> int:
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    ratio = duration_s / sample_period_s
    intervals = int(round(ratio))
    if not np.isclose(ratio, intervals, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of sample_period_s")
    return intervals


def raised_cosine_ramp(
    amplitude_m: float,
    rise_duration_s: float,
    hold_duration_s: float,
    sample_period_s: float,
) -> ProbeTrajectory:
    """Ramp from zero to ``amplitude_m`` with zero endpoint velocity, then hold.

    During the rise, ``x = A/2 * (1 - cos(pi*t/T))``.  This is a bounded,
    smooth version of a step/ramp probe and has analytical velocity and
    acceleration.  Duration inputs must contain an integer number of sample
    periods so that boundary samples are represented exactly.
    """

    scalars = np.asarray(
        [amplitude_m, rise_duration_s, hold_duration_s, sample_period_s], dtype=float
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("probe parameters must be finite")
    if amplitude_m <= 0.0:
        raise ValueError("amplitude_m must be strictly positive")
    if rise_duration_s <= 0.0:
        raise ValueError("rise_duration_s must be strictly positive")
    if hold_duration_s < 0.0:
        raise ValueError("hold_duration_s must be non-negative")
    if sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be strictly positive")

    rise_intervals = _integer_intervals(
        rise_duration_s, sample_period_s, "rise_duration_s"
    )
    hold_intervals = _integer_intervals(
        hold_duration_s, sample_period_s, "hold_duration_s"
    )
    total_intervals = rise_intervals + hold_intervals
    time = np.arange(total_intervals + 1, dtype=float) * sample_period_s

    displacement = np.full_like(time, amplitude_m)
    velocity = np.zeros_like(time)
    acceleration = np.zeros_like(time)

    rise = np.arange(rise_intervals + 1)
    phase = np.pi * time[rise] / rise_duration_s
    displacement[rise] = 0.5 * amplitude_m * (1.0 - np.cos(phase))
    velocity[rise] = (
        0.5 * amplitude_m * np.pi / rise_duration_s * np.sin(phase)
    )
    acceleration[rise] = (
        0.5
        * amplitude_m
        * np.pi**2
        / rise_duration_s**2
        * np.cos(phase)
    )

    return ProbeTrajectory(
        time_s=time,
        displacement_m=displacement,
        velocity_m_per_s=velocity,
        acceleration_m_per_s2=acceleration,
    )

