"""Synchronized and causal sensing pipelines for EXP-0003.

The functions in this module are intentionally small, deterministic Stage 1
baselines.  They do not model a flight stack or claim deployment readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.models import ContactInteractionSimulation

from .practical import causal_low_pass, savitzky_golay, uniform_time_grid
from .synthetic import MeasurementNoise, SyntheticMeasurements


CAUSAL_PIPELINES = (
    "backward_difference",
    "causal_low_pass",
    "causal_polynomial",
    "alpha_beta_gamma",
)
OFFLINE_REFERENCE_PIPELINE = "centered_savitzky_golay"


@dataclass(frozen=True)
class CausalSensingResult:
    """Aligned estimator inputs plus timing and implementation metadata."""

    measurements: SyntheticMeasurements
    true_displacement_m: NDArray[np.float64]
    true_velocity_m_per_s: NDArray[np.float64]
    true_acceleration_m_per_s2: NDArray[np.float64]
    true_contact_force_n: NDArray[np.float64]
    commanded_force_n: NDArray[np.float64]
    raw_displacement_m: NDArray[np.float64]
    raw_velocity_m_per_s: NDArray[np.float64]
    raw_acceleration_m_per_s2: NDArray[np.float64]
    raw_force_n: NDArray[np.float64]
    pipeline: str
    is_causal: bool
    nominal_delay_s: float
    required_lookahead_s: float
    computational_cost_units_per_sample: float
    timestamp_offsets_s: Mapping[str, float]
    kinematic_group_delay_s: float


def backward_difference(
    values: ArrayLike,
    sample_period_s: float,
    *,
    derivative_order: int = 1,
) -> NDArray[np.float64]:
    """First- or second-order strictly backward difference.

    Startup entries repeat the first available causal estimate.  EXP-0003
    trims the common startup interval before identification.
    """

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("values must contain at least three finite samples")
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    if derivative_order not in {0, 1, 2}:
        raise ValueError("derivative_order must be 0, 1, or 2")
    if derivative_order == 0:
        return samples.copy()
    if derivative_order == 1:
        output = np.empty_like(samples)
        output[1:] = np.diff(samples) / sample_period_s
        output[0] = output[1]
        return output
    output = np.empty_like(samples)
    output[2:] = (
        samples[2:] - 2.0 * samples[1:-1] + samples[:-2]
    ) / sample_period_s**2
    output[:2] = output[2]
    return output


def causal_polynomial(
    values: ArrayLike,
    sample_period_s: float,
    *,
    window_duration_s: float,
    polynomial_order: int,
    derivative_order: int = 0,
) -> NDArray[np.float64]:
    """Trailing-window polynomial estimate evaluated at the current sample.

    This is the causal counterpart of a centered Savitzky-Golay estimator: it
    only uses the current and previous samples.  Fixed full-window
    coefficients are reused after startup.
    """

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("values must contain at least three finite samples")
    if sample_period_s <= 0.0 or window_duration_s <= 0.0:
        raise ValueError("sample and window durations must be positive")
    if polynomial_order < 1 or derivative_order not in range(polynomial_order + 1):
        raise ValueError("invalid polynomial or derivative order")
    window = max(int(round(window_duration_s / sample_period_s)) + 1, polynomial_order + 2)
    window = min(window, samples.size)

    def coefficients(count: int) -> NDArray[np.float64]:
        order = min(polynomial_order, count - 1)
        relative_time = np.arange(-(count - 1), 1, dtype=float) * sample_period_s
        vandermonde = np.column_stack(
            [relative_time**power for power in range(order + 1)]
        )
        if derivative_order > order:
            return np.zeros(count, dtype=float)
        return factorial(derivative_order) * np.linalg.pinv(vandermonde)[
            derivative_order
        ]

    output = np.empty_like(samples)
    full_coefficients = coefficients(window)
    for index in range(samples.size):
        count = min(index + 1, window)
        if count <= derivative_order:
            output[index] = 0.0
            continue
        coefficient = full_coefficients if count == window else coefficients(count)
        output[index] = coefficient @ samples[index - count + 1 : index + 1]
    return output


def alpha_beta_gamma_filter(
    position_m: ArrayLike,
    sample_period_s: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Constant-acceleration alpha-beta-gamma tracking filter."""

    samples = np.asarray(position_m, dtype=float)
    gains = np.asarray([alpha, beta, gamma], dtype=float)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("position_m must contain at least three finite samples")
    if sample_period_s <= 0.0 or not np.all(np.isfinite(gains)):
        raise ValueError("sample period and gains must be finite")
    if not 0.0 < alpha <= 1.0 or beta < 0.0 or gamma < 0.0:
        raise ValueError("alpha must be in (0, 1], beta and gamma non-negative")

    position = np.empty_like(samples)
    velocity = np.zeros_like(samples)
    acceleration = np.zeros_like(samples)
    position[0] = samples[0]
    step = float(sample_period_s)
    for index in range(1, samples.size):
        predicted_position = (
            position[index - 1]
            + step * velocity[index - 1]
            + 0.5 * step**2 * acceleration[index - 1]
        )
        predicted_velocity = velocity[index - 1] + step * acceleration[index - 1]
        residual = samples[index] - predicted_position
        position[index] = predicted_position + alpha * residual
        velocity[index] = predicted_velocity + beta * residual / step
        acceleration[index] = acceleration[index - 1] + 2.0 * gamma * residual / step**2
    return position, velocity, acceleration


def estimate_signal_delay(
    reference: ArrayLike,
    estimate: ArrayLike,
    sample_period_s: float,
    *,
    maximum_delay_s: float = 0.25,
) -> float:
    """Estimate signed integer-sample delay by normalized cross-correlation."""

    truth = np.asarray(reference, dtype=float)
    signal = np.asarray(estimate, dtype=float)
    if truth.shape != signal.shape or truth.ndim != 1:
        raise ValueError("reference and estimate must be equal one-dimensional arrays")
    maximum_lag = min(int(round(maximum_delay_s / sample_period_s)), truth.size // 3)
    best_lag = 0
    best_score = -np.inf
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag >= 0:
            left, right = truth[: truth.size - lag or None], signal[lag:]
        else:
            left, right = truth[-lag:], signal[: signal.size + lag]
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        score = float(left @ right / denominator) if denominator > 0.0 else -np.inf
        if score > best_score:
            best_score = score
            best_lag = lag
    return float(best_lag * sample_period_s)


def _sample_channel(
    source_time: NDArray[np.float64],
    source_values: NDArray[np.float64],
    reported_time: NDArray[np.float64],
    *,
    delay_s: float,
    standard_deviation: float,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    actual_time = np.clip(reported_time - delay_s, source_time[0], source_time[-1])
    sampled = np.interp(actual_time, source_time, source_values)
    if standard_deviation > 0.0:
        sampled = sampled + generator.normal(0.0, standard_deviation, sampled.shape)
    return sampled


def _pipeline_metadata(
    pipeline: str, sample_period_s: float, settings: Mapping[str, float]
) -> tuple[bool, float, float, float]:
    window = int(round(float(settings["polynomial_window_duration_s"]) / sample_period_s)) + 1
    if pipeline == "direct":
        return True, 0.0, 0.0, 4.0
    if pipeline == "backward_difference":
        return True, sample_period_s, 0.0, 8.0
    if pipeline == "causal_low_pass":
        delay = 1.0 / (2.0 * np.pi * float(settings["low_pass_cutoff_hz"])) + sample_period_s
        return True, delay, 0.0, 20.0
    if pipeline == "causal_polynomial":
        return True, 0.5 * (window - 1) * sample_period_s, 0.0, float(4 * window)
    if pipeline == "alpha_beta_gamma":
        return True, sample_period_s, 0.0, 24.0
    if pipeline == OFFLINE_REFERENCE_PIPELINE:
        lookahead = 0.5 * (window - 1) * sample_period_s
        return False, 0.0, lookahead, float(4 * window)
    raise ValueError(f"unknown EXP-0003 pipeline: {pipeline}")


def process_causal_sensing(
    truth: ContactInteractionSimulation,
    *,
    pipeline: str,
    sample_rate_hz: float,
    noise: MeasurementNoise,
    pipeline_settings: Mapping[str, float],
    random_seed: int,
    timestamp_offsets_s: Mapping[str, float] | None = None,
    kinematic_group_delay_s: float = 0.0,
) -> CausalSensingResult:
    """Sample synchronized channels and apply one fixed EXP-0003 pipeline."""

    allowed = {"direct", *CAUSAL_PIPELINES, OFFLINE_REFERENCE_PIPELINE}
    if pipeline not in allowed:
        raise ValueError(f"unknown EXP-0003 pipeline: {pipeline}")
    offsets = {
        "displacement": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
        "force": 0.0,
        **({} if timestamp_offsets_s is None else timestamp_offsets_s),
    }
    if not all(np.isfinite(float(value)) for value in offsets.values()):
        raise ValueError("timestamp offsets must be finite")
    if not np.isfinite(kinematic_group_delay_s) or kinematic_group_delay_s < 0.0:
        raise ValueError("kinematic_group_delay_s must be finite and non-negative")

    source = truth.response
    time = uniform_time_grid(float(source.time_s[-1]), float(sample_rate_hz))
    step = float(time[1] - time[0])
    generator = np.random.default_rng(int(random_seed))
    raw_position = _sample_channel(
        source.time_s,
        source.displacement_m,
        time,
        delay_s=float(offsets["displacement"]) + kinematic_group_delay_s,
        standard_deviation=noise.displacement_std_m,
        generator=generator,
    )
    raw_velocity = _sample_channel(
        source.time_s,
        source.velocity_m_per_s,
        time,
        delay_s=float(offsets["velocity"]) + kinematic_group_delay_s,
        standard_deviation=noise.velocity_std_m_per_s,
        generator=generator,
    )
    raw_acceleration = _sample_channel(
        source.time_s,
        source.acceleration_m_per_s2,
        time,
        delay_s=float(offsets["acceleration"]) + kinematic_group_delay_s,
        standard_deviation=noise.acceleration_std_m_per_s2,
        generator=generator,
    )
    raw_force = _sample_channel(
        source.time_s,
        truth.contact_force_n,
        time,
        delay_s=float(offsets["force"]),
        standard_deviation=noise.force_std_n,
        generator=generator,
    )

    window_duration = float(pipeline_settings["polynomial_window_duration_s"])
    order = int(pipeline_settings["polynomial_order"])
    if pipeline == "direct":
        displacement, velocity, acceleration, force = (
            raw_position.copy(),
            raw_velocity.copy(),
            raw_acceleration.copy(),
            raw_force.copy(),
        )
    elif pipeline == "backward_difference":
        displacement = raw_position.copy()
        velocity = backward_difference(raw_position, step, derivative_order=1)
        acceleration = backward_difference(raw_position, step, derivative_order=2)
        force = raw_force.copy()
    elif pipeline == "causal_low_pass":
        cutoff = float(pipeline_settings["low_pass_cutoff_hz"])
        displacement = causal_low_pass(raw_position, step, cutoff)
        velocity = backward_difference(displacement, step, derivative_order=1)
        acceleration = backward_difference(displacement, step, derivative_order=2)
        force = causal_low_pass(raw_force, step, cutoff)
    elif pipeline == "causal_polynomial":
        kwargs = {
            "window_duration_s": window_duration,
            "polynomial_order": order,
        }
        displacement = causal_polynomial(raw_position, step, derivative_order=0, **kwargs)
        velocity = causal_polynomial(raw_position, step, derivative_order=1, **kwargs)
        acceleration = causal_polynomial(raw_position, step, derivative_order=2, **kwargs)
        force = causal_polynomial(raw_force, step, derivative_order=0, **kwargs)
    elif pipeline == "alpha_beta_gamma":
        displacement, velocity, acceleration = alpha_beta_gamma_filter(
            raw_position,
            step,
            alpha=float(pipeline_settings["observer_alpha"]),
            beta=float(pipeline_settings["observer_beta"]),
            gamma=float(pipeline_settings["observer_gamma"]),
        )
        force = causal_low_pass(
            raw_force, step, float(pipeline_settings["observer_force_cutoff_hz"])
        )
    else:
        kwargs = {
            "window_duration_s": window_duration,
            "polynomial_order": order,
        }
        displacement = savitzky_golay(raw_position, step, derivative_order=0, **kwargs)
        velocity = savitzky_golay(raw_position, step, derivative_order=1, **kwargs)
        acceleration = savitzky_golay(raw_position, step, derivative_order=2, **kwargs)
        force = savitzky_golay(raw_force, step, derivative_order=0, **kwargs)

    is_causal, nominal_delay, lookahead, cost = _pipeline_metadata(
        pipeline, step, pipeline_settings
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
    return CausalSensingResult(
        measurements=measurements,
        true_displacement_m=np.interp(time, source.time_s, source.displacement_m),
        true_velocity_m_per_s=np.interp(time, source.time_s, source.velocity_m_per_s),
        true_acceleration_m_per_s2=np.interp(time, source.time_s, source.acceleration_m_per_s2),
        true_contact_force_n=np.interp(time, source.time_s, truth.contact_force_n),
        commanded_force_n=np.interp(time, source.time_s, truth.commanded_force_n),
        raw_displacement_m=raw_position,
        raw_velocity_m_per_s=raw_velocity,
        raw_acceleration_m_per_s2=raw_acceleration,
        raw_force_n=raw_force,
        pipeline=pipeline,
        is_causal=is_causal,
        nominal_delay_s=nominal_delay,
        required_lookahead_s=lookahead,
        computational_cost_units_per_sample=cost,
        timestamp_offsets_s=dict(offsets),
        kinematic_group_delay_s=float(kinematic_group_delay_s),
    )
