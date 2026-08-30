"""Practical Stage 1 sensing and derivative-estimation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.models import ContactInteractionSimulation

from .synthetic import MeasurementNoise, SyntheticMeasurements


@dataclass(frozen=True)
class PracticalSensingResult:
    """Processed estimator inputs and aligned truth for one sensing trial."""

    measurements: SyntheticMeasurements
    true_displacement_m: NDArray[np.float64]
    true_velocity_m_per_s: NDArray[np.float64]
    true_acceleration_m_per_s2: NDArray[np.float64]
    true_contact_force_n: NDArray[np.float64]
    commanded_force_n: NDArray[np.float64]
    raw_displacement_m: NDArray[np.float64]
    raw_acceleration_m_per_s2: NDArray[np.float64]
    raw_force_n: NDArray[np.float64]
    filter_delay_s: float
    nominal_sample_rate_hz: float
    position_sample_rate_hz: float
    timestamp_offsets_s: Mapping[str, float]
    force_is_inferred: bool


def uniform_time_grid(duration_s: float, sample_rate_hz: float) -> NDArray[np.float64]:
    """Return an endpoint-inclusive uniform grid with an integer interval count."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    intervals = int(round(duration_s * sample_rate_hz))
    if intervals < 2 or not np.isclose(
        intervals, duration_s * sample_rate_hz, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("duration_s * sample_rate_hz must be an integer >= 2")
    return np.linspace(0.0, duration_s, intervals + 1)


def finite_difference(
    values: ArrayLike, time_s: ArrayLike, *, derivative_order: int = 1
) -> NDArray[np.float64]:
    """Repeated second-order finite differences on a uniform sample grid."""

    result = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if result.ndim != 1 or time.ndim != 1 or result.shape != time.shape:
        raise ValueError("values and time_s must be equal one-dimensional arrays")
    if derivative_order not in {0, 1, 2}:
        raise ValueError("derivative_order must be 0, 1, or 2")
    if time.size < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("at least three increasing time samples are required")
    for _ in range(derivative_order):
        result = np.gradient(result, time, edge_order=2)
    return np.asarray(result, dtype=float)


def causal_low_pass(
    values: ArrayLike, sample_period_s: float, cutoff_frequency_hz: float
) -> NDArray[np.float64]:
    """First-order causal low-pass filter with explicit time constant."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("values must be a finite one-dimensional array")
    if sample_period_s <= 0.0 or cutoff_frequency_hz <= 0.0:
        raise ValueError("sample period and cutoff must be positive")
    time_constant = 1.0 / (2.0 * np.pi * cutoff_frequency_hz)
    alpha = sample_period_s / (time_constant + sample_period_s)
    filtered = np.empty_like(samples)
    filtered[0] = samples[0]
    for index in range(1, samples.size):
        filtered[index] = filtered[index - 1] + alpha * (
            samples[index] - filtered[index - 1]
        )
    return filtered


def _savgol_window_samples(
    window_duration_s: float, sample_period_s: float, sample_count: int, polynomial_order: int
) -> int:
    window = int(round(window_duration_s / sample_period_s))
    window = max(window, polynomial_order + 2)
    if window % 2 == 0:
        window += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    window = min(window, maximum)
    if window <= polynomial_order:
        raise ValueError("not enough samples for the requested Savitzky-Golay order")
    return window


def savitzky_golay(
    values: ArrayLike,
    sample_period_s: float,
    *,
    window_duration_s: float,
    polynomial_order: int,
    derivative_order: int = 0,
) -> NDArray[np.float64]:
    """Local-polynomial smoothing/differentiation without a SciPy dependency."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("values must be a finite one-dimensional array")
    if derivative_order < 0 or derivative_order > polynomial_order:
        raise ValueError("derivative_order must not exceed polynomial_order")
    if sample_period_s <= 0.0 or window_duration_s <= 0.0:
        raise ValueError("sample and window durations must be positive")
    window = _savgol_window_samples(
        window_duration_s,
        sample_period_s,
        samples.size,
        polynomial_order,
    )
    half = window // 2
    output = np.empty_like(samples)

    def coefficients(relative_indices: NDArray[np.float64]) -> NDArray[np.float64]:
        relative_time = relative_indices * sample_period_s
        vandermonde = np.column_stack(
            [relative_time**order for order in range(polynomial_order + 1)]
        )
        return factorial(derivative_order) * np.linalg.pinv(vandermonde)[
            derivative_order
        ]

    centered = coefficients(np.arange(-half, half + 1, dtype=float))
    output[half : samples.size - half] = np.convolve(
        samples, centered[::-1], mode="valid"
    )
    for index in list(range(half)) + list(range(samples.size - half, samples.size)):
        start = min(max(index - half, 0), samples.size - window)
        stop = start + window
        relative = np.arange(start, stop, dtype=float) - index
        output[index] = coefficients(relative) @ samples[start:stop]
    return output


def _sample_channel(
    source_time: NDArray[np.float64],
    source_values: NDArray[np.float64],
    reported_time: NDArray[np.float64],
    *,
    latency_s: float,
    timestamp_offset_s: float,
    standard_deviation: float,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    actual_time = np.clip(
        reported_time - latency_s - timestamp_offset_s,
        source_time[0],
        source_time[-1],
    )
    values = np.interp(actual_time, source_time, source_values)
    if standard_deviation > 0.0:
        values = values + generator.normal(0.0, standard_deviation, values.shape)
    return values


def _pipeline_signal(
    values: NDArray[np.float64],
    time: NDArray[np.float64],
    pipeline: str,
    settings: Mapping[str, Any],
    derivative_order: int,
) -> NDArray[np.float64]:
    sample_period = float(time[1] - time[0])
    if pipeline == "finite_difference":
        return finite_difference(values, time, derivative_order=derivative_order)
    if pipeline == "low_pass":
        filtered = causal_low_pass(
            values, sample_period, float(settings["low_pass_cutoff_hz"])
        )
        return finite_difference(filtered, time, derivative_order=derivative_order)
    if pipeline == "savitzky_golay":
        return savitzky_golay(
            values,
            sample_period,
            window_duration_s=float(settings["savgol_window_duration_s"]),
            polynomial_order=int(settings["savgol_polynomial_order"]),
            derivative_order=derivative_order,
        )
    raise ValueError(f"unknown derivative pipeline: {pipeline}")


def _filtered_direct_signal(
    values: NDArray[np.float64],
    time: NDArray[np.float64],
    pipeline: str,
    settings: Mapping[str, Any],
) -> NDArray[np.float64]:
    if pipeline == "finite_difference":
        return values.copy()
    return _pipeline_signal(values, time, pipeline, settings, 0)


def _pipeline_delay_s(
    pipeline: str, sample_period_s: float, settings: Mapping[str, Any], sample_count: int
) -> float:
    if pipeline in {"direct", "finite_difference"}:
        return 0.0
    if pipeline == "low_pass":
        return 1.0 / (2.0 * np.pi * float(settings["low_pass_cutoff_hz"]))
    if pipeline == "savitzky_golay":
        window = _savgol_window_samples(
            float(settings["savgol_window_duration_s"]),
            sample_period_s,
            sample_count,
            int(settings["savgol_polynomial_order"]),
        )
        return 0.5 * (window - 1) * sample_period_s
    raise ValueError(f"unknown pipeline: {pipeline}")


def process_practical_sensing(
    truth: ContactInteractionSimulation,
    *,
    regime: str,
    pipeline: str,
    imperfection: Mapping[str, Any],
    pipeline_settings: Mapping[str, Any],
    random_seed: int,
) -> PracticalSensingResult:
    """Create realistic, deliberately imperfect estimator inputs."""

    if regime not in {
        "optimistic_reference",
        "no_direct_velocity",
        "no_direct_acceleration",
        "imu_like",
        "sensorless_force_exploratory",
    }:
        raise ValueError(f"unknown sensing regime: {regime}")
    if regime == "optimistic_reference" and pipeline != "direct":
        raise ValueError("optimistic_reference requires the direct pipeline")
    if regime != "optimistic_reference" and pipeline == "direct":
        raise ValueError("direct pipeline is only valid for optimistic_reference")

    source = truth.response
    duration = float(source.time_s[-1])
    sample_rate = float(imperfection["sample_rate_hz"])
    position_rate = (
        float(imperfection["position_sample_rate_hz"])
        if regime in {"imu_like", "sensorless_force_exploratory"}
        else sample_rate
    )
    time = uniform_time_grid(duration, sample_rate)
    generator = np.random.default_rng(int(random_seed))
    mismatch_std = float(imperfection["timestamp_mismatch_std_s"])
    offsets = {
        name: float(generator.normal(0.0, mismatch_std))
        for name in ("position", "velocity", "acceleration", "force")
    }

    position_time = (
        uniform_time_grid(duration, position_rate)
        if position_rate != sample_rate
        else time
    )
    position_sparse = _sample_channel(
        source.time_s,
        source.displacement_m,
        position_time,
        latency_s=float(imperfection["position_latency_s"]),
        timestamp_offset_s=offsets["position"],
        standard_deviation=float(imperfection["displacement_std_m"]),
        generator=generator,
    )
    raw_position = (
        np.interp(time, position_time, position_sparse)
        if position_time.shape != time.shape
        else position_sparse
    )
    raw_velocity = _sample_channel(
        source.time_s,
        source.velocity_m_per_s,
        time,
        latency_s=float(imperfection["position_latency_s"]),
        timestamp_offset_s=offsets["velocity"],
        standard_deviation=float(imperfection["velocity_std_m_per_s"]),
        generator=generator,
    )
    raw_acceleration = _sample_channel(
        source.time_s,
        source.acceleration_m_per_s2,
        time,
        latency_s=float(imperfection["acceleration_latency_s"]),
        timestamp_offset_s=offsets["acceleration"],
        standard_deviation=float(imperfection["acceleration_std_m_per_s2"]),
        generator=generator,
    )
    measured_force = _sample_channel(
        source.time_s,
        truth.contact_force_n,
        time,
        latency_s=float(imperfection["force_latency_s"]),
        timestamp_offset_s=offsets["force"],
        standard_deviation=float(imperfection["force_std_n"]),
        generator=generator,
    )

    if regime == "sensorless_force_exploratory":
        force_proxy = _sample_channel(
            source.time_s,
            truth.commanded_force_n,
            time,
            latency_s=float(imperfection["sensorless_command_latency_s"]),
            timestamp_offset_s=offsets["force"],
            standard_deviation=float(imperfection["force_std_n"]),
            generator=generator,
        )
        measured_force = causal_low_pass(
            force_proxy,
            float(time[1] - time[0]),
            float(pipeline_settings["sensorless_force_cutoff_hz"]),
        )

    if regime == "optimistic_reference":
        displacement = raw_position
        velocity = raw_velocity
        acceleration = raw_acceleration
        force = measured_force
    else:
        displacement = _pipeline_signal(raw_position, time, pipeline, pipeline_settings, 0)
        position_velocity = _pipeline_signal(
            raw_position, time, pipeline, pipeline_settings, 1
        )
        force = _filtered_direct_signal(
            measured_force, time, pipeline, pipeline_settings
        )
        if regime == "no_direct_velocity":
            velocity = position_velocity
            acceleration = _filtered_direct_signal(
                raw_acceleration, time, pipeline, pipeline_settings
            )
        elif regime == "no_direct_acceleration":
            velocity = position_velocity
            acceleration = _pipeline_signal(
                raw_position, time, pipeline, pipeline_settings, 2
            )
        else:
            acceleration = _filtered_direct_signal(
                raw_acceleration, time, pipeline, pipeline_settings
            )
            velocity = np.empty_like(time)
            velocity[0] = position_velocity[0]
            step = float(time[1] - time[0])
            alpha = float(
                np.exp(
                    -step
                    / float(pipeline_settings["complementary_time_constant_s"])
                )
            )
            for index in range(1, time.size):
                integrated = velocity[index - 1] + 0.5 * step * (
                    acceleration[index - 1] + acceleration[index]
                )
                velocity[index] = alpha * integrated + (1.0 - alpha) * position_velocity[
                    index
                ]

    true_displacement = np.interp(time, source.time_s, source.displacement_m)
    true_velocity = np.interp(time, source.time_s, source.velocity_m_per_s)
    true_acceleration = np.interp(time, source.time_s, source.acceleration_m_per_s2)
    true_force = np.interp(time, source.time_s, truth.contact_force_n)
    command = np.interp(time, source.time_s, truth.commanded_force_n)
    noise = MeasurementNoise(
        displacement_std_m=float(imperfection["displacement_std_m"]),
        velocity_std_m_per_s=float(imperfection["velocity_std_m_per_s"]),
        acceleration_std_m_per_s2=float(imperfection["acceleration_std_m_per_s2"]),
        force_std_n=float(imperfection["force_std_n"]),
    )
    measurements = SyntheticMeasurements(
        time_s=time,
        displacement_m=np.asarray(displacement, dtype=float),
        velocity_m_per_s=np.asarray(velocity, dtype=float),
        acceleration_m_per_s2=np.asarray(acceleration, dtype=float),
        contact_force_n=np.asarray(force, dtype=float),
        noise=noise,
        random_seed=int(random_seed),
    )
    return PracticalSensingResult(
        measurements=measurements,
        true_displacement_m=true_displacement,
        true_velocity_m_per_s=true_velocity,
        true_acceleration_m_per_s2=true_acceleration,
        true_contact_force_n=true_force,
        commanded_force_n=command,
        raw_displacement_m=raw_position,
        raw_acceleration_m_per_s2=raw_acceleration,
        raw_force_n=measured_force,
        filter_delay_s=_pipeline_delay_s(
            pipeline, float(time[1] - time[0]), pipeline_settings, time.size
        ),
        nominal_sample_rate_hz=sample_rate,
        position_sample_rate_hz=position_rate,
        timestamp_offsets_s=offsets,
        force_is_inferred=regime == "sensorless_force_exploratory",
    )
