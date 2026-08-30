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


@dataclass(frozen=True)
class ProbeSignal:
    """A sampled bounded applied-force probe."""

    name: str
    time_s: NDArray[np.float64]
    force_n: NDArray[np.float64]
    amplitude_bound_n: float

    def __post_init__(self) -> None:
        if self.time_s.ndim != 1 or self.force_n.shape != self.time_s.shape:
            raise ValueError("probe time and force must be one-dimensional and equal length")
        if self.time_s.size < 2 or not np.all(np.diff(self.time_s) > 0.0):
            raise ValueError("probe time must be strictly increasing")
        if not np.all(np.isfinite(self.force_n)):
            raise ValueError("probe force must be finite")
        if not np.isfinite(self.amplitude_bound_n) or self.amplitude_bound_n <= 0.0:
            raise ValueError("amplitude_bound_n must be finite and positive")
        tolerance = 32.0 * np.finfo(float).eps * self.amplitude_bound_n
        if np.max(np.abs(self.force_n)) > self.amplitude_bound_n + tolerance:
            raise ValueError("probe exceeds its declared amplitude bound")


def _integer_intervals(duration_s: float, sample_period_s: float, name: str) -> int:
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    ratio = duration_s / sample_period_s
    intervals = int(round(ratio))
    if not np.isclose(ratio, intervals, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of sample_period_s")
    return intervals


def _probe_time(duration_s: float, sample_period_s: float) -> NDArray[np.float64]:
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and strictly positive")
    intervals = _integer_intervals(duration_s, sample_period_s, "duration_s")
    if intervals < 1:
        raise ValueError("duration_s must contain at least one sample interval")
    return np.arange(intervals + 1, dtype=float) * sample_period_s


def _positive_amplitude(amplitude_n: float) -> float:
    if not np.isfinite(amplitude_n) or amplitude_n <= 0.0:
        raise ValueError("amplitude_n must be finite and strictly positive")
    return float(amplitude_n)


def force_ramp(
    amplitude_n: float,
    rise_duration_s: float,
    duration_s: float,
    sample_period_s: float,
) -> ProbeSignal:
    """Linear force ramp followed by a constant hold."""

    amplitude = _positive_amplitude(amplitude_n)
    time = _probe_time(duration_s, sample_period_s)
    if not np.isfinite(rise_duration_s) or rise_duration_s <= 0.0:
        raise ValueError("rise_duration_s must be finite and strictly positive")
    if rise_duration_s > duration_s:
        raise ValueError("rise_duration_s cannot exceed duration_s")
    _integer_intervals(rise_duration_s, sample_period_s, "rise_duration_s")
    force = amplitude * np.minimum(time / rise_duration_s, 1.0)
    return ProbeSignal("ramp", time, force, amplitude)


def half_sine_pulse(
    amplitude_n: float,
    pulse_duration_s: float,
    duration_s: float,
    sample_period_s: float,
) -> ProbeSignal:
    """Positive half-sine force pulse followed by zero force."""

    amplitude = _positive_amplitude(amplitude_n)
    time = _probe_time(duration_s, sample_period_s)
    if not np.isfinite(pulse_duration_s) or pulse_duration_s <= 0.0:
        raise ValueError("pulse_duration_s must be finite and strictly positive")
    if pulse_duration_s > duration_s:
        raise ValueError("pulse_duration_s cannot exceed duration_s")
    _integer_intervals(pulse_duration_s, sample_period_s, "pulse_duration_s")
    force = np.where(
        time <= pulse_duration_s,
        amplitude * np.sin(np.pi * time / pulse_duration_s),
        0.0,
    )
    force[np.abs(force) < 4.0 * np.finfo(float).eps * amplitude] = 0.0
    return ProbeSignal("half_sine", time, force, amplitude)


def sinusoid(
    amplitude_n: float,
    frequency_hz: float,
    duration_s: float,
    sample_period_s: float,
) -> ProbeSignal:
    """Zero-mean single-frequency sinusoidal force."""

    amplitude = _positive_amplitude(amplitude_n)
    time = _probe_time(duration_s, sample_period_s)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and strictly positive")
    return ProbeSignal(
        "sinusoid", time, amplitude * np.sin(2.0 * np.pi * frequency_hz * time), amplitude
    )


def chirp(
    amplitude_n: float,
    start_frequency_hz: float,
    end_frequency_hz: float,
    duration_s: float,
    sample_period_s: float,
) -> ProbeSignal:
    """Constant-amplitude linear-frequency chirp."""

    amplitude = _positive_amplitude(amplitude_n)
    time = _probe_time(duration_s, sample_period_s)
    frequencies = np.asarray([start_frequency_hz, end_frequency_hz], dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("chirp frequencies must be finite and strictly positive")
    sweep_rate = (end_frequency_hz - start_frequency_hz) / duration_s
    phase = 2.0 * np.pi * (start_frequency_hz * time + 0.5 * sweep_rate * time**2)
    return ProbeSignal("chirp", time, amplitude * np.sin(phase), amplitude)


def multisine(
    amplitude_n: float,
    frequencies_hz: ArrayLike,
    duration_s: float,
    sample_period_s: float,
    *,
    weights: ArrayLike | None = None,
    phases_rad: ArrayLike | None = None,
) -> ProbeSignal:
    """Weighted multisine normalized to guarantee the requested force bound."""

    amplitude = _positive_amplitude(amplitude_n)
    time = _probe_time(duration_s, sample_period_s)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("multisine frequencies must be finite and strictly positive")
    coefficient = (
        np.ones_like(frequencies) if weights is None else np.asarray(weights, dtype=float)
    )
    phase = np.zeros_like(frequencies) if phases_rad is None else np.asarray(phases_rad, dtype=float)
    if coefficient.shape != frequencies.shape or phase.shape != frequencies.shape:
        raise ValueError("weights and phases_rad must match frequencies_hz")
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(phase)):
        raise ValueError("multisine weights and phases must be finite")
    normalizer = float(np.sum(np.abs(coefficient)))
    if normalizer <= 0.0:
        raise ValueError("at least one multisine weight must be non-zero")
    components = coefficient[:, None] * np.sin(
        2.0 * np.pi * frequencies[:, None] * time[None, :] + phase[:, None]
    )
    force = amplitude * np.sum(components, axis=0) / normalizer
    return ProbeSignal("multisine", time, force, amplitude)


def make_probe(config: dict[str, object], sample_period_s: float, duration_s: float) -> ProbeSignal:
    """Construct one of the Milestone A force probes from configuration."""

    name = str(config["name"])
    amplitude = float(config["amplitude_n"])
    if name == "ramp":
        return force_ramp(amplitude, float(config["rise_duration_s"]), duration_s, sample_period_s)
    if name == "half_sine":
        return half_sine_pulse(
            amplitude, float(config["pulse_duration_s"]), duration_s, sample_period_s
        )
    if name == "sinusoid":
        return sinusoid(amplitude, float(config["frequency_hz"]), duration_s, sample_period_s)
    if name == "chirp":
        return chirp(
            amplitude,
            float(config["start_frequency_hz"]),
            float(config["end_frequency_hz"]),
            duration_s,
            sample_period_s,
        )
    if name == "multisine":
        return multisine(
            amplitude,
            np.asarray(config["frequencies_hz"], dtype=float),
            duration_s,
            sample_period_s,
            weights=np.asarray(config.get("weights", []), dtype=float) if "weights" in config else None,
            phases_rad=np.asarray(config.get("phases_rad", []), dtype=float) if "phases_rad" in config else None,
        )
    raise ValueError(f"unknown probe signal: {name}")


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
