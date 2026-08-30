"""EXP-0005 decision-sufficient Stage 1 interaction probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from probeing.estimators import (
    delayed_input_instruments,
    instrumental_variables,
    ordinary_least_squares_eiv,
    total_least_squares,
)
from probeing.measurements import MeasurementNoise, SyntheticMeasurements, process_causal_sensing
from probeing.models import (
    ContactInteractionSimulation,
    InteractionParameters,
    InteractionSimulation,
)
from probeing.probing import ProbeSignal, chirp


RISK_LABELS = ("SAFE", "CAUTION", "UNSAFE")
CLASS_TO_INDEX = {label: index for index, label in enumerate(RISK_LABELS)}
PARAMETERS = ("stiffness", "damping", "effective_mass")

FREQUENCY_FEATURES = (
    "fr_peak_gain_log",
    "fr_peak_frequency_hz",
    "fr_centroid_hz",
    "fr_low_gain_log",
    "fr_mid_gain_log",
    "fr_high_gain_log",
    "fr_high_low_log_ratio",
    "fr_peak_phase_sin",
    "fr_peak_phase_cos",
    "probe_dominant_frequency_hz",
)
TIME_FEATURES = (
    "probe_peak_displacement_log",
    "probe_rms_displacement_log",
    "probe_peak_velocity_log",
    "probe_rms_velocity_log",
    "probe_peak_acceleration_log",
    "probe_rms_acceleration_log",
    "probe_free_displacement_rms_log",
    "probe_persistence_ratio_log",
    "probe_peak_gain_log",
    "probe_rms_gain_log",
    "probe_absolute_work_log",
    "probe_final_displacement_signed_log",
    "probe_velocity_zero_crossing_log",
)
FEATURE_SETS = {
    "full_parameters": (
        "estimated_stiffness_signed_log",
        "estimated_damping_signed_log",
        "estimated_mass_signed_log",
    ),
    "stiffness_only": ("estimated_stiffness_signed_log",),
    "frequency_response": FREQUENCY_FEATURES,
    "time_domain": TIME_FEATURES,
    "combined_task": ("estimated_stiffness_signed_log", *FREQUENCY_FEATURES, *TIME_FEATURES),
}


@dataclass(frozen=True)
class TargetCase:
    target_id: str
    partition: str
    seed: int
    case_index: int
    stiffness_n_per_m: float
    damping_n_s_per_m: float
    effective_mass_kg: float


@dataclass(frozen=True)
class PopulationSimulation:
    time_s: NDArray[np.float64]
    commanded_force_n: NDArray[np.float64]
    contact_force_n: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]
    acceleration_m_per_s2: NDArray[np.float64]


@dataclass(frozen=True)
class RidgeModel:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]


@dataclass(frozen=True)
class LogisticModel:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]


@dataclass(frozen=True)
class OutcomeBoundModel:
    regressors: Mapping[str, RidgeModel]
    upper_log_residuals: Mapping[str, float]
    safe_force_lower_log_residual: float


@dataclass(frozen=True)
class DecisionSufficiencyResult:
    validation_rows: tuple[Mapping[str, Any], ...]
    classification_summary: tuple[Mapping[str, Any], ...]
    quantitative_summary: tuple[Mapping[str, Any], ...]
    class_distribution: tuple[Mapping[str, Any], ...]
    mass_decision_comparison: tuple[Mapping[str, Any], ...]
    dynamics_performance: tuple[Mapping[str, Any], ...]
    false_safe_cases: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    feature_space_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool
    stage1_decision: str


def generate_target_cases(
    seeds: Sequence[int],
    cases_per_seed: int,
    bounds: Mapping[str, Sequence[float]],
    *,
    partition: str,
) -> tuple[TargetCase, ...]:
    """Generate reproducible log-Latin-hypercube target dynamics."""

    if cases_per_seed < 3:
        raise ValueError("cases_per_seed must be at least three")
    names = ("stiffness_n_per_m", "damping_n_s_per_m", "effective_mass_kg")
    log_bounds = {
        name: np.log(np.asarray(bounds[name], dtype=float)) for name in names
    }
    if any(value.shape != (2,) or value[0] >= value[1] for value in log_bounds.values()):
        raise ValueError("target population bounds must be increasing positive pairs")
    output: list[TargetCase] = []
    for seed in seeds:
        generator = np.random.default_rng(int(seed))
        coordinates: dict[str, NDArray[np.float64]] = {}
        for name in names:
            bins = (np.arange(cases_per_seed, dtype=float) + generator.random(cases_per_seed)) / cases_per_seed
            coordinates[name] = bins[generator.permutation(cases_per_seed)]
        for index in range(cases_per_seed):
            values = {
                name: float(
                    np.exp(
                        log_bounds[name][0]
                        + coordinates[name][index] * (log_bounds[name][1] - log_bounds[name][0])
                    )
                )
                for name in names
            }
            output.append(
                TargetCase(
                    target_id=f"{partition}_s{int(seed)}_c{index:02d}",
                    partition=partition,
                    seed=int(seed),
                    case_index=index,
                    **values,
                )
            )
    return tuple(output)


def _probe_signal(config: Mapping[str, Any]) -> ProbeSignal:
    values = config["probe"]
    return chirp(
        float(values["amplitude_n"]),
        float(values["start_frequency_hz"]),
        float(values["end_frequency_hz"]),
        float(values["duration_s"]),
        float(values["sample_period_s"]),
    )


def _maneuver_signal(config: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = config["future_contact_maneuver"]
    step = float(values["sample_period_s"])
    duration = float(values["total_duration_s"])
    time = np.arange(int(round(duration / step)) + 1, dtype=float) * step
    amplitude = float(values["contact_force_n"])
    ramp_up = float(values["ramp_up_s"])
    hold_end = float(values["hold_end_s"])
    ramp_down_end = float(values["ramp_down_end_s"])
    force = np.zeros_like(time)
    rising = time <= ramp_up
    force[rising] = 0.5 * amplitude * (1.0 - np.cos(np.pi * time[rising] / ramp_up))
    holding = (time > ramp_up) & (time <= hold_end)
    force[holding] = amplitude
    falling = (time > hold_end) & (time <= ramp_down_end)
    force[falling] = 0.5 * amplitude * (
        1.0 + np.cos(np.pi * (time[falling] - hold_end) / (ramp_down_end - hold_end))
    )
    force[np.abs(force) < 1.0e-15] = 0.0
    return time, force


def simulate_population(
    cases: Sequence[TargetCase],
    time_s: NDArray[np.float64],
    commanded_force_n: NDArray[np.float64],
    *,
    contact_mode: str,
) -> PopulationSimulation:
    """Vectorized RK4 population simulation with the validated force convention."""

    time = np.asarray(time_s, dtype=float)
    command = np.asarray(commanded_force_n, dtype=float)
    if time.ndim != 1 or command.shape != time.shape or time.size < 2:
        raise ValueError("population time and command must align")
    if contact_mode == "unilateral":
        force = np.maximum(command, 0.0)
    elif contact_mode == "bilateral":
        force = command.copy()
    else:
        raise ValueError("contact mode must be unilateral or bilateral")
    stiffness = np.asarray([case.stiffness_n_per_m for case in cases], dtype=float)
    damping = np.asarray([case.damping_n_s_per_m for case in cases], dtype=float)
    mass = np.asarray([case.effective_mass_kg for case in cases], dtype=float)
    displacement = np.zeros((len(cases), time.size), dtype=float)
    velocity = np.zeros_like(displacement)

    def derivative(x: NDArray[np.float64], v: NDArray[np.float64], applied: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return v, (applied - damping * v - stiffness * x) / mass

    for index in range(time.size - 1):
        step = float(time[index + 1] - time[index])
        force_start = float(force[index])
        force_end = float(force[index + 1])
        force_mid = 0.5 * (force_start + force_end)
        x = displacement[:, index]
        v = velocity[:, index]
        k1x, k1v = derivative(x, v, force_start)
        k2x, k2v = derivative(x + 0.5 * step * k1x, v + 0.5 * step * k1v, force_mid)
        k3x, k3v = derivative(x + 0.5 * step * k2x, v + 0.5 * step * k2v, force_mid)
        k4x, k4v = derivative(x + step * k3x, v + step * k3v, force_end)
        displacement[:, index + 1] = x + step * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        velocity[:, index + 1] = v + step * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    acceleration = (
        force[None, :] - damping[:, None] * velocity - stiffness[:, None] * displacement
    ) / mass[:, None]
    return PopulationSimulation(
        time_s=time,
        commanded_force_n=command,
        contact_force_n=force,
        displacement_m=displacement,
        velocity_m_per_s=velocity,
        acceleration_m_per_s2=acceleration,
    )


def _case_simulation(
    population: PopulationSimulation, case: TargetCase, index: int, contact_mode: str
) -> ContactInteractionSimulation:
    displacement = population.displacement_m[index]
    velocity = population.velocity_m_per_s[index]
    mass = case.effective_mass_kg
    response = InteractionSimulation(
        time_s=population.time_s,
        applied_force_n=population.contact_force_n,
        displacement_m=displacement,
        velocity_m_per_s=velocity,
        acceleration_m_per_s2=population.acceleration_m_per_s2[index],
        kinetic_energy_j=0.5 * mass * velocity**2,
        elastic_energy_j=0.5 * case.stiffness_n_per_m * displacement**2,
    )
    active = population.contact_force_n > 0.0
    return ContactInteractionSimulation(
        response=response,
        commanded_force_n=population.commanded_force_n,
        contact_force_n=population.contact_force_n,
        contact_active=active,
        contact_mode=contact_mode,
        contact_loss_count=int(np.count_nonzero(active[:-1] & ~active[1:])),
    )


def _risk_class(metrics: Mapping[str, float | bool], config: Mapping[str, Any]) -> str:
    envelope = config["risk_envelope"]
    mapping = {
        "peak_displacement_m": "peak_displacement_m",
        "peak_velocity_m_per_s": "peak_velocity_m_per_s",
        "late_hold_oscillation_rms_m": "late_hold_oscillation_rms_m",
        "hold_settling_time_s": "hold_settling_time_s",
    }
    safe = all(
        float(metrics[metric]) <= float(envelope["safe"][threshold])
        for metric, threshold in mapping.items()
    ) and not bool(metrics.get("contact_loss_proxy", False))
    unsafe = any(
        float(metrics[metric]) > float(envelope["unsafe"][threshold])
        for metric, threshold in mapping.items()
    ) or bool(metrics.get("contact_loss_proxy", False))
    return "SAFE" if safe else ("UNSAFE" if unsafe else "CAUTION")


def _dominant_frequency(time: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    if time.size < 4:
        return 0.0
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    frequency = np.fft.rfftfreq(centered.size, float(time[1] - time[0]))
    if spectrum.size <= 1 or np.max(spectrum[1:]) <= np.finfo(float).tiny:
        return 0.0
    return float(frequency[1 + int(np.argmax(spectrum[1:]))])


def evaluate_future_response(
    population: PopulationSimulation,
    cases: Sequence[TargetCase],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Create labels strictly from the hidden future maneuver response."""

    maneuver = config["future_contact_maneuver"]
    envelope = config["risk_envelope"]
    time = population.time_s
    ramp_end = float(maneuver["ramp_up_s"])
    hold_end = float(maneuver["hold_end_s"])
    down_end = float(maneuver["ramp_down_end_s"])
    forced = time <= hold_end
    late = (time >= hold_end - 1.0) & (time <= hold_end)
    ringdown = time >= down_end
    output: list[Mapping[str, Any]] = []
    for index, case in enumerate(cases):
        displacement = population.displacement_m[index]
        velocity = population.velocity_m_per_s[index]
        peak_displacement = float(np.max(np.abs(displacement[forced])))
        peak_velocity = float(np.max(np.abs(velocity[forced])))
        late_center = float(np.mean(displacement[late]))
        oscillation = float(np.sqrt(np.mean((displacement[late] - late_center) ** 2)))
        tolerance = max(
            float(envelope["settling_displacement_fraction"]) * abs(late_center),
            float(envelope["settling_displacement_floor_m"]),
        )
        eligible = np.flatnonzero((time >= ramp_end) & (time <= hold_end))
        settled = (
            np.abs(displacement - late_center) <= tolerance
        ) & (np.abs(velocity) <= float(envelope["settling_velocity_m_per_s"]))
        settling_time = hold_end - ramp_end
        for sample in eligible:
            if bool(np.all(settled[sample : eligible[-1] + 1])):
                settling_time = float(time[sample] - ramp_end)
                break
        dominant = _dominant_frequency(time[ringdown], displacement[ringdown])
        amplitude = float(maneuver["contact_force_n"])
        gain = peak_displacement / amplitude
        contact_loss_proxy = bool(
            peak_displacement > float(envelope["contact_tracking_displacement_m"])
        )
        ratios = (
            float(envelope["safe"]["peak_displacement_m"])
            / max(peak_displacement, np.finfo(float).eps),
            float(envelope["safe"]["peak_velocity_m_per_s"])
            / max(peak_velocity, np.finfo(float).eps),
            float(envelope["safe"]["late_hold_oscillation_rms_m"])
            / max(oscillation, np.finfo(float).eps),
        )
        safe_force = min(
            float(config["safe_force_definition"]["maximum_considered_force_n"]),
            amplitude * min(ratios),
        )
        metrics: dict[str, Any] = {
            "target_id": case.target_id,
            "peak_displacement_m": peak_displacement,
            "peak_velocity_m_per_s": peak_velocity,
            "dominant_response_frequency_hz": dominant,
            "late_hold_oscillation_rms_m": oscillation,
            "hold_settling_time_s": float(settling_time),
            "force_to_displacement_gain_m_per_n": gain,
            "disturbance_limited_safe_force_n": float(max(safe_force, 0.0)),
            "contact_loss_proxy": contact_loss_proxy,
        }
        metrics["risk_class"] = _risk_class(metrics, config)
        output.append(metrics)
    return tuple(output)


def _noise(config: Mapping[str, Any], multiplier: float) -> MeasurementNoise:
    base = config["sensing"]["base_noise"]
    return MeasurementNoise(
        displacement_std_m=multiplier * float(base["displacement_std_m"]),
        velocity_std_m_per_s=multiplier * float(base["velocity_std_m_per_s"]),
        acceleration_std_m_per_s2=multiplier * float(base["acceleration_std_m_per_s2"]),
        force_std_n=multiplier * float(base["force_std_n"]),
    )


def _trim(
    measurements: SyntheticMeasurements, trim_s: float
) -> tuple[SyntheticMeasurements, NDArray[np.bool_]]:
    time = measurements.time_s
    mask = (time >= time[0] + trim_s) & (time <= time[-1] - trim_s)
    return (
        SyntheticMeasurements(
            time_s=time[mask],
            displacement_m=measurements.displacement_m[mask],
            velocity_m_per_s=measurements.velocity_m_per_s[mask],
            acceleration_m_per_s2=measurements.acceleration_m_per_s2[mask],
            contact_force_n=measurements.contact_force_n[mask],
            noise=measurements.noise,
            random_seed=measurements.random_seed,
        ),
        mask,
    )


def _signed_log(value: float, scale: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value) / scale))


def _positive_log(value: float, floor: float = 1.0e-12) -> float:
    return float(np.log(max(abs(value), floor)))


def _frequency_features(measurements: SyntheticMeasurements) -> Mapping[str, float]:
    time = measurements.time_s
    displacement = measurements.displacement_m - np.mean(measurements.displacement_m)
    force = measurements.contact_force_n - np.mean(measurements.contact_force_n)
    frequency = np.fft.rfftfreq(time.size, float(time[1] - time[0]))
    displacement_spectrum = np.fft.rfft(displacement)
    force_spectrum = np.fft.rfft(force)
    denominator_floor = 0.05 * max(float(np.max(np.abs(force_spectrum))), np.finfo(float).eps)
    transfer = displacement_spectrum / np.where(
        np.abs(force_spectrum) >= denominator_floor,
        force_spectrum,
        denominator_floor + 0.0j,
    )
    gain = np.abs(transfer)
    band = (frequency >= 0.5) & (frequency <= 5.0)
    band_indices = np.flatnonzero(band)
    peak_index = int(band_indices[np.argmax(gain[band])])

    def band_gain(low: float, high: float) -> float:
        mask = (frequency >= low) & (frequency < high)
        return float(np.median(gain[mask])) if np.any(mask) else 0.0

    energy = np.abs(displacement_spectrum[band]) ** 2
    centroid = float(
        np.sum(frequency[band] * energy) / max(float(np.sum(energy)), np.finfo(float).eps)
    )
    low_gain = band_gain(0.5, 1.5)
    mid_gain = band_gain(1.5, 3.0)
    high_gain = band_gain(3.0, 5.01)
    phase = float(np.angle(transfer[peak_index]))
    return {
        "fr_peak_gain_log": _positive_log(float(gain[peak_index])),
        "fr_peak_frequency_hz": float(frequency[peak_index]),
        "fr_centroid_hz": centroid,
        "fr_low_gain_log": _positive_log(low_gain),
        "fr_mid_gain_log": _positive_log(mid_gain),
        "fr_high_gain_log": _positive_log(high_gain),
        "fr_high_low_log_ratio": float(np.log(max(high_gain, 1.0e-12) / max(low_gain, 1.0e-12))),
        "fr_peak_phase_sin": float(np.sin(phase)),
        "fr_peak_phase_cos": float(np.cos(phase)),
        "probe_dominant_frequency_hz": _dominant_frequency(time, displacement),
    }


def _time_features(
    measurements: SyntheticMeasurements, commanded_force: NDArray[np.float64]
) -> Mapping[str, float]:
    time = measurements.time_s
    displacement = measurements.displacement_m
    velocity = measurements.velocity_m_per_s
    acceleration = measurements.acceleration_m_per_s2
    force = measurements.contact_force_n
    free = commanded_force <= 1.0e-12
    free_values = displacement[free] if np.any(free) else displacement
    free_time = time[free] if np.any(free) else time
    midpoint = float(np.median(free_time))
    early = free_time <= midpoint
    late = free_time > midpoint
    early_rms = float(np.sqrt(np.mean(free_values[early] ** 2))) if np.any(early) else 0.0
    late_rms = float(np.sqrt(np.mean(free_values[late] ** 2))) if np.any(late) else early_rms
    persistence = late_rms / max(early_rms, 1.0e-12)
    peak_force = max(float(np.max(np.abs(force))), 1.0e-12)
    rms_force = max(float(np.sqrt(np.mean(force**2))), 1.0e-12)
    absolute_work = float(np.trapz(np.abs(force * velocity), time))
    signs = np.signbit(velocity)
    crossings = int(np.count_nonzero(signs[:-1] != signs[1:]))
    return {
        "probe_peak_displacement_log": _positive_log(float(np.max(np.abs(displacement)))),
        "probe_rms_displacement_log": _positive_log(float(np.sqrt(np.mean(displacement**2)))),
        "probe_peak_velocity_log": _positive_log(float(np.max(np.abs(velocity)))),
        "probe_rms_velocity_log": _positive_log(float(np.sqrt(np.mean(velocity**2)))),
        "probe_peak_acceleration_log": _positive_log(float(np.max(np.abs(acceleration)))),
        "probe_rms_acceleration_log": _positive_log(float(np.sqrt(np.mean(acceleration**2)))),
        "probe_free_displacement_rms_log": _positive_log(float(np.sqrt(np.mean(free_values**2)))),
        "probe_persistence_ratio_log": float(np.log(max(persistence, 1.0e-12))),
        "probe_peak_gain_log": _positive_log(float(np.max(np.abs(displacement))) / peak_force),
        "probe_rms_gain_log": _positive_log(float(np.sqrt(np.mean(displacement**2))) / rms_force),
        "probe_absolute_work_log": _positive_log(absolute_work),
        "probe_final_displacement_signed_log": _signed_log(float(displacement[-1]), 1.0e-3),
        "probe_velocity_zero_crossing_log": float(np.log1p(crossings)),
        "probe_peak_displacement_m": float(np.max(np.abs(displacement))),
        "probe_peak_velocity_m_per_s": float(np.max(np.abs(velocity))),
        "probe_free_displacement_rms_m": float(np.sqrt(np.mean(free_values**2))),
        "probe_persistence_ratio": float(persistence),
    }


def _probe_features(
    truth: ContactInteractionSimulation,
    case: TargetCase,
    config: Mapping[str, Any],
    *,
    noise_name: str,
    noise_multiplier: float,
    random_seed: int,
) -> tuple[Mapping[str, Any], Any]:
    sensing_config = config["sensing"]
    common = dict(
        sample_rate_hz=float(sensing_config["sample_rate_hz"]),
        noise=_noise(config, noise_multiplier),
        pipeline_settings=sensing_config["pipeline_settings"],
        random_seed=int(random_seed),
        timestamp_offsets_s={"displacement": 0.0, "velocity": 0.0, "acceleration": 0.0, "force": 0.0},
    )
    primary = process_causal_sensing(
        truth, pipeline=str(sensing_config["primary_pipeline"]), **common
    )
    stiffness_sensing = process_causal_sensing(
        truth, pipeline=str(sensing_config["stiffness_diagnostic_pipeline"]), **common
    )
    trim = float(sensing_config["analysis_trim_s"])
    measurements, mask = _trim(primary.measurements, trim)
    stiffness_measurements, stiffness_mask = _trim(stiffness_sensing.measurements, trim)
    if not np.array_equal(mask, stiffness_mask):
        raise RuntimeError("diagnostic pipelines are not aligned")
    command = np.maximum(np.asarray(primary.commanded_force_n[mask], dtype=float), 0.0)
    instruments = delayed_input_instruments(
        command,
        measurements.time_s,
        config["diagnostic_parameter_estimation"]["iv_input_delays_s"],
    )
    ols = ordinary_least_squares_eiv(measurements)
    iv = instrumental_variables(measurements, instruments)
    tls = total_least_squares(stiffness_measurements)
    stiffness_estimate = (
        float(tls.parameters[0])
        if tls.valid and np.isfinite(tls.parameters[0])
        else float(ols.parameters[0])
    )
    mass_estimate = (
        float(iv.parameters[2])
        if iv.valid and np.isfinite(iv.parameters[2])
        else float(ols.parameters[2])
    )
    estimated = np.asarray([stiffness_estimate, ols.parameters[1], mass_estimate], dtype=float)
    true = np.asarray(
        [case.stiffness_n_per_m, case.damping_n_s_per_m, case.effective_mass_kg], dtype=float
    )
    features: dict[str, Any] = {
        "target_id": case.target_id,
        "partition": case.partition,
        "source_seed": case.seed,
        "case_index": case.case_index,
        "noise_regime": noise_name,
        "noise_multiplier": noise_multiplier,
        "estimated_stiffness_n_per_m": float(estimated[0]),
        "estimated_damping_n_s_per_m": float(estimated[1]),
        "estimated_effective_mass_kg": float(estimated[2]),
        "estimated_stiffness_signed_log": _signed_log(float(estimated[0]), 100.0),
        "estimated_damping_signed_log": _signed_log(float(estimated[1]), 5.0),
        "estimated_mass_signed_log": _signed_log(float(estimated[2]), 1.0),
        "stiffness_abs_relative_error": float(abs((estimated[0] - true[0]) / true[0])),
        "damping_abs_relative_error": float(abs((estimated[1] - true[1]) / true[1])),
        "effective_mass_abs_relative_error": float(abs((estimated[2] - true[2]) / true[2])),
        "true_stiffness_n_per_m": true[0],
        "true_damping_n_s_per_m": true[1],
        "true_effective_mass_kg": true[2],
        **_frequency_features(measurements),
        **_time_features(measurements, command),
    }
    return features, primary


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> NDArray[np.float64]:
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("predictor features must be a finite matrix")
    return matrix


def _standardize_fit(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def _fit_ridge(
    matrix: NDArray[np.float64], values: NDArray[np.float64], penalty: float
) -> RidgeModel:
    normalized, mean, scale = _standardize_fit(matrix)
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ values)
    return RidgeModel(mean=mean, scale=scale, coefficients=coefficients)


def _ridge_predict(model: RidgeModel, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    normalized = (matrix - model.mean) / model.scale
    return np.column_stack((np.ones(normalized.shape[0]), normalized)) @ model.coefficients


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _fit_logistic(
    matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: Mapping[str, Any],
) -> LogisticModel:
    normalized, mean, scale = _standardize_fit(matrix)
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    coefficients = np.zeros((design.shape[1], len(RISK_LABELS)), dtype=float)
    one_hot = np.eye(len(RISK_LABELS))[labels]
    counts = np.bincount(labels, minlength=len(RISK_LABELS)).astype(float)
    weights = labels.size / (len(RISK_LABELS) * np.maximum(counts, 1.0))
    sample_weights = weights[labels]
    iterations = int(config["iterations"])
    learning_rate = float(config["learning_rate"])
    penalty = float(config["l2_penalty"])
    regularizer_mask = np.ones_like(coefficients)
    regularizer_mask[0] = 0.0
    for iteration in range(iterations):
        probabilities = _softmax(design @ coefficients)
        gradient = (
            design.T @ ((probabilities - one_hot) * sample_weights[:, None])
            / np.sum(sample_weights)
            + penalty * coefficients * regularizer_mask
        )
        step = learning_rate / np.sqrt(1.0 + iteration / 200.0)
        coefficients -= step * gradient
    return LogisticModel(mean=mean, scale=scale, coefficients=coefficients)


def _logistic_probabilities(
    model: LogisticModel, matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    normalized = (matrix - model.mean) / model.scale
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    return _softmax(design @ model.coefficients)


def _probability_class(probabilities: NDArray[np.float64], config: Mapping[str, Any]) -> str:
    safe_threshold = float(config["safe_probability_threshold"])
    unsafe_threshold = float(config["unsafe_probability_threshold"])
    if probabilities[0] >= safe_threshold:
        return "SAFE"
    if probabilities[2] >= unsafe_threshold:
        return "UNSAFE"
    return "CAUTION"


def _outcome_map(outcomes: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    return {str(row["target_id"]): row for row in outcomes}


def _transform_outcome(outcome: str, value: float) -> float:
    """Map positive outcomes to a stable regression scale.

    Frequency and settling time legitimately reach zero in the simulated
    population.  ``log1p`` preserves that endpoint without turning small
    numerical changes around zero into many orders of magnitude.
    """

    if outcome in {"dominant_response_frequency_hz", "hold_settling_time_s"}:
        return float(np.log1p(max(value, 0.0)))
    return float(np.log(max(value, 1.0e-9)))


def _inverse_outcome(outcome: str, transformed_value: float) -> float:
    clipped = float(np.clip(transformed_value, -30.0, 30.0))
    if outcome in {"dominant_response_frequency_hz", "hold_settling_time_s"}:
        return float(max(np.expm1(clipped), 0.0))
    return float(np.exp(clipped))


def _fit_models(
    training_features: Sequence[Mapping[str, Any]],
    training_outcomes: Sequence[Mapping[str, Any]],
    calibration_features: Sequence[Mapping[str, Any]],
    calibration_outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, OutcomeBoundModel], Mapping[str, LogisticModel]]:
    training_truth = _outcome_map(training_outcomes)
    calibration_truth = _outcome_map(calibration_outcomes)
    quantitative = tuple(str(name) for name in config["quantitative_outcomes"])
    regression_models: dict[str, OutcomeBoundModel] = {}
    logistic_models: dict[str, LogisticModel] = {}
    ridge_penalty = float(config["predictors"]["outcome_bound_regression"]["ridge_penalty"])
    upper_quantile = float(
        config["predictors"]["outcome_bound_regression"]["upper_residual_quantile"]
    )
    lower_quantile = float(
        config["predictors"]["outcome_bound_regression"]["safe_force_lower_residual_quantile"]
    )
    for feature_set, names in FEATURE_SETS.items():
        training_matrix = _feature_matrix(training_features, names)
        calibration_matrix = _feature_matrix(calibration_features, names)
        regressors: dict[str, RidgeModel] = {}
        upper: dict[str, float] = {}
        safe_force_lower = 0.0
        for outcome in quantitative:
            training_values = np.asarray(
                [
                    _transform_outcome(
                        outcome,
                        float(training_truth[str(row["target_id"])][outcome]),
                    )
                    for row in training_features
                ],
                dtype=float,
            )
            model = _fit_ridge(training_matrix, training_values, ridge_penalty)
            regressors[outcome] = model
            calibration_values = np.asarray(
                [
                    _transform_outcome(
                        outcome,
                        float(calibration_truth[str(row["target_id"])][outcome]),
                    )
                    for row in calibration_features
                ]
            )
            residuals = calibration_values - _ridge_predict(model, calibration_matrix)
            if outcome == "disturbance_limited_safe_force_n":
                safe_force_lower = float(np.quantile(residuals, lower_quantile))
            else:
                upper[outcome] = float(np.quantile(residuals, upper_quantile))
        regression_models[feature_set] = OutcomeBoundModel(
            regressors=regressors,
            upper_log_residuals=upper,
            safe_force_lower_log_residual=safe_force_lower,
        )
        labels = np.asarray(
            [CLASS_TO_INDEX[str(training_truth[str(row["target_id"])]["risk_class"])] for row in training_features],
            dtype=np.int64,
        )
        logistic_models[feature_set] = _fit_logistic(
            training_matrix, labels, config["predictors"]["multinomial_logistic"]
        )
    return regression_models, logistic_models


def _bound_prediction(
    model: OutcomeBoundModel,
    matrix: NDArray[np.float64],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    median: dict[str, float] = {}
    upper: dict[str, float] = {}
    for outcome, regressor in model.regressors.items():
        log_prediction = float(_ridge_predict(regressor, matrix)[0])
        median[outcome] = _inverse_outcome(outcome, log_prediction)
        if outcome != "disturbance_limited_safe_force_n":
            upper[outcome] = _inverse_outcome(
                outcome,
                log_prediction + float(model.upper_log_residuals[outcome]),
            )
    safe_force_lower = _inverse_outcome(
        "disturbance_limited_safe_force_n",
        float(
            _ridge_predict(
                model.regressors["disturbance_limited_safe_force_n"], matrix
            )[0]
        )
        + model.safe_force_lower_log_residual,
    )
    conservative_safe_metrics = {
        "peak_displacement_m": upper["peak_displacement_m"],
        "peak_velocity_m_per_s": upper["peak_velocity_m_per_s"],
        "late_hold_oscillation_rms_m": upper["late_hold_oscillation_rms_m"],
        "hold_settling_time_s": upper["hold_settling_time_s"],
        "contact_loss_proxy": upper["peak_displacement_m"]
        > float(config["risk_envelope"]["contact_tracking_displacement_m"]),
    }
    median_risk_metrics = {
        "peak_displacement_m": median["peak_displacement_m"],
        "peak_velocity_m_per_s": median["peak_velocity_m_per_s"],
        "late_hold_oscillation_rms_m": median["late_hold_oscillation_rms_m"],
        "hold_settling_time_s": median["hold_settling_time_s"],
        "contact_loss_proxy": median["peak_displacement_m"]
        > float(config["risk_envelope"]["contact_tracking_displacement_m"]),
    }
    envelope = config["risk_envelope"]
    conservative_safe = all(
        float(conservative_safe_metrics[name]) <= float(envelope["safe"][name])
        for name in (
            "peak_displacement_m",
            "peak_velocity_m_per_s",
            "late_hold_oscillation_rms_m",
            "hold_settling_time_s",
        )
    ) and not bool(conservative_safe_metrics["contact_loss_proxy"])
    conservative_unsafe = any(
        float(conservative_safe_metrics[name]) > float(envelope["unsafe"][name])
        for name in (
            "peak_displacement_m",
            "peak_velocity_m_per_s",
            "late_hold_oscillation_rms_m",
            "hold_settling_time_s",
        )
    ) or bool(conservative_safe_metrics["contact_loss_proxy"])
    predicted_class = (
        "SAFE" if conservative_safe else ("UNSAFE" if conservative_unsafe else "CAUTION")
    )
    severity = max(
        upper["peak_displacement_m"] / float(config["risk_envelope"]["safe"]["peak_displacement_m"]),
        upper["peak_velocity_m_per_s"] / float(config["risk_envelope"]["safe"]["peak_velocity_m_per_s"]),
        upper["late_hold_oscillation_rms_m"]
        / float(config["risk_envelope"]["safe"]["late_hold_oscillation_rms_m"]),
        upper["hold_settling_time_s"]
        / float(config["risk_envelope"]["safe"]["hold_settling_time_s"]),
    )
    return {
        "predicted_risk_class": predicted_class,
        "predicted_risk_score": float(severity),
        "predicted_contact_loss_proxy": bool(conservative_safe_metrics["contact_loss_proxy"]),
        "predicted_outcomes": median,
        "upper_outcomes": upper,
        "safe_force_lower_bound_n": safe_force_lower,
    }


def _threshold_prediction(features: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    ratio = float(config["predictors"]["threshold_rule"]["future_to_probe_force_ratio"])
    peak_displacement = ratio * float(features["probe_peak_displacement_m"])
    peak_velocity = ratio * float(features["probe_peak_velocity_m_per_s"])
    oscillation = ratio * float(features["probe_free_displacement_rms_m"])
    settling = float(
        np.clip(3.0 * float(features["probe_persistence_ratio"]), 0.0, 3.0)
    )
    metrics = {
        "peak_displacement_m": peak_displacement,
        "peak_velocity_m_per_s": peak_velocity,
        "late_hold_oscillation_rms_m": oscillation,
        "hold_settling_time_s": settling,
        "contact_loss_proxy": peak_displacement
        > float(config["risk_envelope"]["contact_tracking_displacement_m"]),
    }
    severity = max(
        peak_displacement / float(config["risk_envelope"]["safe"]["peak_displacement_m"]),
        peak_velocity / float(config["risk_envelope"]["safe"]["peak_velocity_m_per_s"]),
        oscillation / float(config["risk_envelope"]["safe"]["late_hold_oscillation_rms_m"]),
        settling / float(config["risk_envelope"]["safe"]["hold_settling_time_s"]),
    )
    safe_force = min(
        float(config["safe_force_definition"]["maximum_considered_force_n"]),
        float(config["probe"]["amplitude_n"])
        * min(
            float(config["risk_envelope"]["safe"]["peak_displacement_m"])
            / max(float(features["probe_peak_displacement_m"]), 1.0e-12),
            float(config["risk_envelope"]["safe"]["peak_velocity_m_per_s"])
            / max(float(features["probe_peak_velocity_m_per_s"]), 1.0e-12),
            float(config["risk_envelope"]["safe"]["late_hold_oscillation_rms_m"])
            / max(float(features["probe_free_displacement_rms_m"]), 1.0e-12),
        ),
    )
    predicted_outcomes = {name: float("nan") for name in config["quantitative_outcomes"]}
    predicted_outcomes.update(
        {
            "peak_displacement_m": peak_displacement,
            "peak_velocity_m_per_s": peak_velocity,
            "late_hold_oscillation_rms_m": oscillation,
            "hold_settling_time_s": settling,
            "disturbance_limited_safe_force_n": float(max(safe_force, 0.0)),
        }
    )
    return {
        "predicted_risk_class": _risk_class(metrics, config),
        "predicted_risk_score": float(severity),
        "predicted_contact_loss_proxy": bool(metrics["contact_loss_proxy"]),
        "predicted_outcomes": predicted_outcomes,
        "upper_outcomes": dict(predicted_outcomes),
        "safe_force_lower_bound_n": float(max(safe_force, 0.0)),
    }


def _prediction_row(
    feature: Mapping[str, Any],
    *,
    predictor: str,
    feature_set: str,
    prediction: Mapping[str, Any],
    probabilities: Sequence[float] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    probability = [float("nan")] * 3 if probabilities is None else [float(value) for value in probabilities]
    predicted_outcomes = prediction["predicted_outcomes"]
    upper_outcomes = prediction["upper_outcomes"]
    row: dict[str, Any] = {
        "target_id": str(feature["target_id"]),
        "source_seed": int(feature["source_seed"]),
        "case_index": int(feature["case_index"]),
        "noise_regime": str(feature["noise_regime"]),
        "noise_multiplier": float(feature["noise_multiplier"]),
        "predictor": predictor,
        "feature_set": feature_set,
        "predicted_risk_class": str(prediction["predicted_risk_class"]),
        "predicted_risk_score": float(prediction["predicted_risk_score"]),
        "probability_safe": probability[0],
        "probability_caution": probability[1],
        "probability_unsafe": probability[2],
        "predicted_contact_loss_proxy": bool(prediction["predicted_contact_loss_proxy"]),
        "estimated_stiffness_n_per_m": float(feature["estimated_stiffness_n_per_m"]),
        "estimated_damping_n_s_per_m": float(feature["estimated_damping_n_s_per_m"]),
        "estimated_effective_mass_kg": float(feature["estimated_effective_mass_kg"]),
        "stiffness_abs_relative_error": float(feature["stiffness_abs_relative_error"]),
        "damping_abs_relative_error": float(feature["damping_abs_relative_error"]),
        "effective_mass_abs_relative_error": float(feature["effective_mass_abs_relative_error"]),
        "true_stiffness_n_per_m": float(feature["true_stiffness_n_per_m"]),
        "true_damping_n_s_per_m": float(feature["true_damping_n_s_per_m"]),
        "true_effective_mass_kg": float(feature["true_effective_mass_kg"]),
    }
    for outcome in config["quantitative_outcomes"]:
        row[f"predicted_{outcome}"] = float(predicted_outcomes.get(outcome, float("nan")))
        row[f"upper_{outcome}"] = float(upper_outcomes.get(outcome, float("nan")))
    row["lower_disturbance_limited_safe_force_n"] = float(
        prediction["safe_force_lower_bound_n"]
    )
    return row


def _measurement_seed(case: TargetCase, regime_index: int) -> int:
    return int(case.seed * 100_000 + case.case_index * 100 + regime_index)


def _extract_partition_features(
    cases: Sequence[TargetCase],
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None,
    progress_state: list[int],
    progress_total: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], PopulationSimulation]:
    signal = _probe_signal(config)
    population = simulate_population(
        cases,
        signal.time_s,
        signal.force_n,
        contact_mode=str(config["probe"]["contact_mode"]),
    )
    limits = config["probe_safety_limits"]
    safety_events: list[Mapping[str, Any]] = []
    feature_rows: list[Mapping[str, Any]] = []
    regimes = config["sensing"]["noise_regimes"]
    for index, case in enumerate(cases):
        peak_values = {
            "peak_force_n": float(np.max(np.abs(population.contact_force_n))),
            "peak_target_displacement_m": float(np.max(np.abs(population.displacement_m[index]))),
            "peak_target_velocity_m_per_s": float(np.max(np.abs(population.velocity_m_per_s[index]))),
            "peak_target_acceleration_m_per_s2": float(np.max(np.abs(population.acceleration_m_per_s2[index]))),
        }
        comparisons = (
            ("peak_force_n", "max_peak_force_n"),
            ("peak_target_displacement_m", "max_peak_target_displacement_m"),
            ("peak_target_velocity_m_per_s", "max_peak_target_velocity_m_per_s"),
            ("peak_target_acceleration_m_per_s2", "max_peak_target_acceleration_m_per_s2"),
        )
        for metric, limit in comparisons:
            if peak_values[metric] > float(limits[limit]):
                safety_events.append(
                    {
                        "target_id": case.target_id,
                        "partition": case.partition,
                        "metric": metric,
                        "value": peak_values[metric],
                        "limit": float(limits[limit]),
                    }
                )
        truth = _case_simulation(population, case, index, str(config["probe"]["contact_mode"]))
        for regime_index, regime in enumerate(regimes):
            features, _ = _probe_features(
                truth,
                case,
                config,
                noise_name=str(regime["name"]),
                noise_multiplier=float(regime["multiplier"]),
                random_seed=_measurement_seed(case, regime_index),
            )
            feature_rows.append(features)
            progress_state[0] += 1
            if progress_callback and (
                progress_state[0] % 50 == 0 or progress_state[0] == progress_total
            ):
                progress_callback(progress_state[0], progress_total)
    return tuple(feature_rows), tuple(safety_events), population


def _simulate_outcomes(
    cases: Sequence[TargetCase], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    time, force = _maneuver_signal(config)
    population = simulate_population(cases, time, force, contact_mode="unilateral")
    return evaluate_future_response(population, cases, config)


def _actual_severity(outcome: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    safe = config["risk_envelope"]["safe"]
    return float(
        max(
            float(outcome["peak_displacement_m"]) / float(safe["peak_displacement_m"]),
            float(outcome["peak_velocity_m_per_s"]) / float(safe["peak_velocity_m_per_s"]),
            float(outcome["late_hold_oscillation_rms_m"])
            / float(safe["late_hold_oscillation_rms_m"]),
            float(outcome["hold_settling_time_s"]) / float(safe["hold_settling_time_s"]),
        )
    )


def _attach_actual_outcomes(
    pending_rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    truth = _outcome_map(outcomes)
    output: list[Mapping[str, Any]] = []
    for pending in pending_rows:
        actual = truth[str(pending["target_id"])]
        row = dict(pending)
        row.update(
            {
                "actual_risk_class": str(actual["risk_class"]),
                "actual_risk_score": _actual_severity(actual, config),
                "false_safe": bool(
                    pending["predicted_risk_class"] == "SAFE"
                    and actual["risk_class"] != "SAFE"
                ),
                "false_unsafe": bool(
                    pending["predicted_risk_class"] == "UNSAFE"
                    and actual["risk_class"] == "SAFE"
                ),
                "classification_correct": bool(
                    pending["predicted_risk_class"] == actual["risk_class"]
                ),
                "actual_contact_loss_proxy": bool(actual["contact_loss_proxy"]),
            }
        )
        for outcome in config["quantitative_outcomes"]:
            row[f"actual_{outcome}"] = float(actual[outcome])
        output.append(row)
    return tuple(output)


def _classification_metrics(
    rows: Sequence[Mapping[str, Any]], probabilities_available: bool
) -> Mapping[str, float | int]:
    actual = np.asarray([CLASS_TO_INDEX[str(row["actual_risk_class"])] for row in rows], dtype=int)
    predicted = np.asarray(
        [CLASS_TO_INDEX[str(row["predicted_risk_class"])] for row in rows], dtype=int
    )
    confusion = np.zeros((3, 3), dtype=int)
    for truth, estimate in zip(actual, predicted):
        confusion[truth, estimate] += 1
    non_safe = actual != CLASS_TO_INDEX["SAFE"]
    safe = actual == CLASS_TO_INDEX["SAFE"]
    false_safe = (predicted == CLASS_TO_INDEX["SAFE"]) & non_safe
    false_unsafe = (predicted == CLASS_TO_INDEX["UNSAFE"]) & safe
    output: dict[str, float | int] = {
        "trial_count": len(rows),
        "accuracy": float(np.mean(actual == predicted)),
        "false_safe_rate": float(np.sum(false_safe) / max(np.sum(non_safe), 1)),
        "false_safe_fraction": float(np.mean(false_safe)),
        "false_unsafe_rate": float(np.sum(false_unsafe) / max(np.sum(safe), 1)),
        "contact_loss_accuracy": float(
            np.mean(
                [
                    bool(row["predicted_contact_loss_proxy"])
                    == bool(row["actual_contact_loss_proxy"])
                    for row in rows
                ]
            )
        ),
    }
    actual_loss = np.asarray([bool(row["actual_contact_loss_proxy"]) for row in rows])
    predicted_loss = np.asarray([bool(row["predicted_contact_loss_proxy"]) for row in rows])
    output["contact_loss_recall"] = float(
        np.sum(actual_loss & predicted_loss) / max(np.sum(actual_loss), 1)
    )
    for index, label in enumerate(RISK_LABELS):
        true_positive = confusion[index, index]
        output[f"{label.lower()}_precision"] = float(
            true_positive / max(np.sum(confusion[:, index]), 1)
        )
        output[f"{label.lower()}_recall"] = float(
            true_positive / max(np.sum(confusion[index, :]), 1)
        )
    for actual_index, actual_label in enumerate(RISK_LABELS):
        for predicted_index, predicted_label in enumerate(RISK_LABELS):
            output[f"confusion_{actual_label.lower()}_{predicted_label.lower()}"] = int(
                confusion[actual_index, predicted_index]
            )
    if probabilities_available:
        probabilities = np.asarray(
            [
                [row["probability_safe"], row["probability_caution"], row["probability_unsafe"]]
                for row in rows
            ],
            dtype=float,
        )
        one_hot = np.eye(3)[actual]
        output["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        confidence = np.max(probabilities, axis=1)
        probability_prediction = np.argmax(probabilities, axis=1)
        correct = probability_prediction == actual
        ece = 0.0
        for low, high in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
            mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
        output["expected_calibration_error"] = float(ece)
    else:
        output["brier_score"] = float("nan")
        output["expected_calibration_error"] = float("nan")
    return output


def _classification_summary(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    groups = sorted(
        {
            (str(row["predictor"]), str(row["feature_set"]), str(row["noise_regime"]))
            for row in rows
        }
    )
    output: list[Mapping[str, Any]] = []
    for predictor, feature_set, noise in groups:
        selected = [
            row
            for row in rows
            if row["predictor"] == predictor
            and row["feature_set"] == feature_set
            and row["noise_regime"] == noise
        ]
        output.append(
            {
                "predictor": predictor,
                "feature_set": feature_set,
                "noise_regime": noise,
                **_classification_metrics(selected, predictor == "logistic"),
            }
        )
    return tuple(output)


def _quantitative_summary(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    regression_rows = [row for row in rows if row["predictor"] == "outcome_bound"]
    for feature_set in FEATURE_SETS:
        for noise in [str(regime["name"]) for regime in config["sensing"]["noise_regimes"]]:
            selected = [
                row
                for row in regression_rows
                if row["feature_set"] == feature_set and row["noise_regime"] == noise
            ]
            for outcome in config["quantitative_outcomes"]:
                actual = np.asarray([row[f"actual_{outcome}"] for row in selected], dtype=float)
                predicted = np.asarray([row[f"predicted_{outcome}"] for row in selected], dtype=float)
                floor = max(float(np.percentile(actual, 10.0)) * 0.1, 1.0e-6)
                relative = (predicted - actual) / np.maximum(np.abs(actual), floor)
                if outcome == "disturbance_limited_safe_force_n":
                    bound = np.asarray(
                        [row["lower_disturbance_limited_safe_force_n"] for row in selected], dtype=float
                    )
                    coverage = float(np.mean(bound <= actual))
                    bound_kind = "lower"
                else:
                    bound = np.asarray([row[f"upper_{outcome}"] for row in selected], dtype=float)
                    coverage = float(np.mean(bound >= actual))
                    bound_kind = "upper"
                output.append(
                    {
                        "feature_set": feature_set,
                        "noise_regime": noise,
                        "outcome": str(outcome),
                        "trial_count": len(selected),
                        "median_abs_relative_error": float(np.median(np.abs(relative))),
                        "p95_abs_relative_error": float(np.percentile(np.abs(relative), 95.0)),
                        "relative_rmse": float(np.sqrt(np.mean(relative**2))),
                        "relative_bias": float(np.mean(relative)),
                        "bound_kind": bound_kind,
                        "bound_coverage": coverage,
                        "median_bound_ratio": float(np.median(bound / np.maximum(actual, floor))),
                    }
                )
    return tuple(output)


def _class_distribution(
    outcomes: Sequence[Mapping[str, Any]], partition: str
) -> tuple[Mapping[str, Any], ...]:
    total = len(outcomes)
    return tuple(
        {
            "partition": partition,
            "risk_class": label,
            "case_count": sum(row["risk_class"] == label for row in outcomes),
            "fraction": float(sum(row["risk_class"] == label for row in outcomes) / total),
        }
        for label in RISK_LABELS
    )


def _primary_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["predictor"] == "outcome_bound" and row["feature_set"] == "combined_task"
    ]


def _mass_decision_comparison(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for noise in [str(regime["name"]) for regime in config["sensing"]["noise_regimes"]]:
        noise_rows = [row for row in _primary_rows(rows) if row["noise_regime"] == noise]
        groups = (
            ("below_30_percent", 0.0, 0.30),
            ("30_to_60_percent", 0.30, 0.60),
            ("above_60_percent", 0.60, float("inf")),
            ("above_30_percent", 0.30, float("inf")),
        )
        for name, lower, upper in groups:
            selected = [
                row
                for row in noise_rows
                if float(row["effective_mass_abs_relative_error"]) >= lower
                and float(row["effective_mass_abs_relative_error"]) < upper
            ]
            if not selected:
                continue
            output.append(
                {
                    "noise_regime": noise,
                    "mass_error_group": name,
                    "trial_count": len(selected),
                    "median_mass_abs_relative_error": float(
                        np.median([row["effective_mass_abs_relative_error"] for row in selected])
                    ),
                    **_classification_metrics(selected, False),
                }
            )
        quartile_threshold = float(
            np.quantile(
                [row["effective_mass_abs_relative_error"] for row in noise_rows],
                0.75,
            )
        )
        worst_quartile = [
            row
            for row in noise_rows
            if float(row["effective_mass_abs_relative_error"]) >= quartile_threshold
        ]
        output.append(
            {
                "noise_regime": noise,
                "mass_error_group": "worst_mass_error_quartile",
                "trial_count": len(worst_quartile),
                "median_mass_abs_relative_error": float(
                    np.median(
                        [row["effective_mass_abs_relative_error"] for row in worst_quartile]
                    )
                ),
                **_classification_metrics(worst_quartile, False),
            }
        )
    return tuple(output)


def _dynamics_performance(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    selected = [row for row in _primary_rows(rows) if row["noise_regime"] == "nominal"]
    bounds = config["target_population"]
    output: list[Mapping[str, Any]] = []
    for parameter, field, bounds_field in (
        ("stiffness", "true_stiffness_n_per_m", "stiffness_n_per_m"),
        ("damping", "true_damping_n_s_per_m", "damping_n_s_per_m"),
        ("effective_mass", "true_effective_mass_kg", "effective_mass_kg"),
    ):
        edges = np.exp(
            np.linspace(
                np.log(float(bounds[bounds_field][0])),
                np.log(float(bounds[bounds_field][1])),
                4,
            )
        )
        for index, label in enumerate(("low", "middle", "high")):
            subset = [
                row
                for row in selected
                if float(row[field]) >= edges[index]
                and (
                    float(row[field]) < edges[index + 1]
                    if index < 2
                    else float(row[field]) <= edges[index + 1]
                )
            ]
            output.append(
                {
                    "parameter": parameter,
                    "dynamics_bin": label,
                    "lower_bound": float(edges[index]),
                    "upper_bound": float(edges[index + 1]),
                    **_classification_metrics(subset, False),
                }
            )
    return tuple(output)


def _feature_space(
    features: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    selected = [feature for feature in features if feature["noise_regime"] == "nominal"]
    matrix = _feature_matrix(selected, FEATURE_SETS["combined_task"])
    normalized, _, _ = _standardize_fit(matrix)
    _, _, right = np.linalg.svd(normalized, full_matrices=False)
    components = normalized @ right[:2].T
    truth = _outcome_map(outcomes)
    predictions = {
        str(row["target_id"]): str(row["predicted_risk_class"])
        for row in _primary_rows(rows)
        if row["noise_regime"] == "nominal"
    }
    return tuple(
        {
            "target_id": str(feature["target_id"]),
            "component_1": float(components[index, 0]),
            "component_2": float(components[index, 1]),
            "actual_risk_class": str(truth[str(feature["target_id"])]["risk_class"]),
            "predicted_risk_class": predictions[str(feature["target_id"])],
        }
        for index, feature in enumerate(selected)
    )


def _representative_raw(
    cases: Sequence[TargetCase],
    outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, NDArray[Any]]:
    truth = _outcome_map(outcomes)
    case_by_id = {case.target_id: case for case in cases}
    selected: list[TargetCase] = []
    for label in RISK_LABELS:
        target_id = next(
            str(row["target_id"]) for row in outcomes if row["risk_class"] == label
        )
        selected.append(case_by_id[target_id])
    signal = _probe_signal(config)
    population = simulate_population(
        selected,
        signal.time_s,
        signal.force_n,
        contact_mode=str(config["probe"]["contact_mode"]),
    )
    regime_index = next(
        index
        for index, regime in enumerate(config["sensing"]["noise_regimes"])
        if str(regime["name"]) == str(config["representative_raw"]["noise_regime"])
    )
    regime = config["sensing"]["noise_regimes"][regime_index]
    parts: dict[str, list[NDArray[Any]]] = {}
    for index, case in enumerate(selected):
        simulation = _case_simulation(
            population, case, index, str(config["probe"]["contact_mode"])
        )
        _, sensing = _probe_features(
            simulation,
            case,
            config,
            noise_name=str(regime["name"]),
            noise_multiplier=float(regime["multiplier"]),
            random_seed=_measurement_seed(case, regime_index),
        )
        count = sensing.measurements.time_s.size
        values = {
            "target_id": np.full(count, case.target_id),
            "risk_class": np.full(count, truth[case.target_id]["risk_class"]),
            "time_s": sensing.measurements.time_s,
            "true_displacement_m": sensing.true_displacement_m,
            "measured_displacement_m": sensing.measurements.displacement_m,
            "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
            "estimated_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
            "true_acceleration_m_per_s2": sensing.true_acceleration_m_per_s2,
            "estimated_acceleration_m_per_s2": sensing.measurements.acceleration_m_per_s2,
            "true_force_n": sensing.true_contact_force_n,
            "measured_force_n": sensing.measurements.contact_force_n,
            "commanded_force_n": sensing.commanded_force_n,
        }
        for field, array in values.items():
            parts.setdefault(field, []).append(np.asarray(array))
    return {field: np.concatenate(arrays) for field, arrays in parts.items()}


def _summary_row(
    summary: Sequence[Mapping[str, Any]],
    predictor: str,
    feature_set: str,
    noise: str,
) -> Mapping[str, Any]:
    return next(
        row
        for row in summary
        if row["predictor"] == predictor
        and row["feature_set"] == feature_set
        and row["noise_regime"] == noise
    )


def _quant_row(
    summary: Sequence[Mapping[str, Any]], feature_set: str, noise: str, outcome: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in summary
        if row["feature_set"] == feature_set
        and row["noise_regime"] == noise
        and row["outcome"] == outcome
    )


def _summarize_result(
    validation_rows: Sequence[Mapping[str, Any]],
    classification: Sequence[Mapping[str, Any]],
    quantitative: Sequence[Mapping[str, Any]],
    mass_comparison: Sequence[Mapping[str, Any]],
    safety_events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    primary = {
        noise: _summary_row(classification, "outcome_bound", "combined_task", noise)
        for noise in ("low", "nominal", "high")
    }
    full_parameters = _summary_row(
        classification, "outcome_bound", "full_parameters", "nominal"
    )
    combined = primary["nominal"]
    mass_above_30 = next(
        (
            row
            for row in mass_comparison
            if row["noise_regime"] == "nominal"
            and row["mass_error_group"] == "above_30_percent"
        ),
        None,
    )
    mass_bad = next(
        row
        for row in mass_comparison
        if row["noise_regime"] == "nominal"
        and row["mass_error_group"] == "worst_mass_error_quartile"
    )
    displacement_bounds = {
        noise: _quant_row(
            quantitative, "combined_task", noise, "peak_displacement_m"
        )
        for noise in ("low", "nominal", "high")
    }
    safe_force_bounds = {
        noise: _quant_row(
            quantitative,
            "combined_task",
            noise,
            "disturbance_limited_safe_force_n",
        )
        for noise in ("low", "nominal", "high")
    }
    criterion = config["kill_criterion"]
    false_safe_pass = all(
        float(row["false_safe_rate"])
        <= float(criterion["maximum_false_safe_rate_each_noise_regime"])
        for row in primary.values()
    )
    accuracy_pass = bool(
        float(primary["nominal"]["accuracy"])
        >= float(criterion["minimum_accuracy_nominal"])
        and float(primary["high"]["accuracy"])
        >= float(criterion["minimum_accuracy_high_noise"])
    )
    unsafe_recall_pass = all(
        float(row["unsafe_recall"])
        >= float(criterion["minimum_unsafe_recall_each_noise_regime"])
        for row in primary.values()
    )
    displacement_bound_pass = all(
        float(row["bound_coverage"])
        >= float(criterion["minimum_displacement_upper_bound_coverage"])
        for row in displacement_bounds.values()
    )
    safe_force_bound_pass = all(
        float(row["bound_coverage"])
        >= float(criterion["minimum_safe_force_lower_bound_coverage"])
        for row in safe_force_bounds.values()
    )
    kill_pass = bool(
        false_safe_pass
        and accuracy_pass
        and unsafe_recall_pass
        and displacement_bound_pass
        and safe_force_bound_pass
    )
    mass_bad_accuracy = float(mass_bad["accuracy"])
    stage_gate_pass = bool(
        kill_pass
        and np.isfinite(mass_bad_accuracy)
        and mass_bad_accuracy
        >= float(
            config["stage1_gate"][
                "require_decision_accuracy_in_worst_mass_error_quartile"
            ]
        )
        and not safety_events
    )
    explicit_parameters_needed = bool(
        float(combined["accuracy"]) + 0.02 < float(full_parameters["accuracy"])
        or float(combined["false_safe_rate"]) > float(full_parameters["false_safe_rate"]) + 0.01
    )
    feature_comparison = tuple(
        {
            "feature_set": feature_set,
            "outcome_bound_accuracy": float(
                _summary_row(classification, "outcome_bound", feature_set, "nominal")["accuracy"]
            ),
            "outcome_bound_false_safe_rate": float(
                _summary_row(classification, "outcome_bound", feature_set, "nominal")["false_safe_rate"]
            ),
            "logistic_accuracy": float(
                _summary_row(classification, "logistic", feature_set, "nominal")["accuracy"]
            ),
            "logistic_false_safe_rate": float(
                _summary_row(classification, "logistic", feature_set, "nominal")["false_safe_rate"]
            ),
        }
        for feature_set in FEATURE_SETS
    )
    stage1_decision = (
        "READY_FOR_INDEPENDENT_MATLAB_SIMULINK_VALIDATION"
        if stage_gate_pass
        else "CONTINUE_STAGE_1"
    )
    summary = {
        "primary_predictor": "combined_task_outcome_bound",
        "primary_by_noise": {name: dict(row) for name, row in primary.items()},
        "feature_set_comparison_nominal": feature_comparison,
        "full_parameter_baseline_nominal": dict(full_parameters),
        "mass_error_above_30_percent_nominal": (
            None if mass_above_30 is None else dict(mass_above_30)
        ),
        "worst_mass_error_quartile_nominal": dict(mass_bad),
        "explicit_full_parameters_needed": explicit_parameters_needed,
        "displacement_bound_by_noise": {
            name: dict(row) for name, row in displacement_bounds.items()
        },
        "safe_force_bound_by_noise": {
            name: dict(row) for name, row in safe_force_bounds.items()
        },
        "kill_criterion_checks": {
            "false_safe": false_safe_pass,
            "accuracy": accuracy_pass,
            "unsafe_recall": unsafe_recall_pass,
            "displacement_upper_bound": displacement_bound_pass,
            "safe_force_lower_bound": safe_force_bound_pass,
        },
        "decision_sufficiency_supported": kill_pass,
        "kill_criterion_decision": (
            "DECISION_SUFFICIENCY_SUPPORTED"
            if kill_pass
            else "RECONSIDER_PROBING_ARCHITECTURE"
        ),
        "stage1_gate_pass": stage_gate_pass,
        "stage1_decision": stage1_decision,
        "validation_predictions_created_before_future_maneuver": True,
    }
    metrics = {
        "validation_case_count": len(
            {row["target_id"] for row in validation_rows}
        ),
        "validation_prediction_count": len(validation_rows),
        "probe_safety_event_count": len(safety_events),
        "nominal_accuracy": float(primary["nominal"]["accuracy"]),
        "nominal_false_safe_rate": float(primary["nominal"]["false_safe_rate"]),
        "high_noise_accuracy": float(primary["high"]["accuracy"]),
        "high_noise_false_safe_rate": float(primary["high"]["false_safe_rate"]),
        "mass_bad_nominal_accuracy": mass_bad_accuracy,
        "nominal_displacement_bound_coverage": float(
            displacement_bounds["nominal"]["bound_coverage"]
        ),
        "nominal_safe_force_bound_coverage": float(
            safe_force_bounds["nominal"]["bound_coverage"]
        ),
        "decision_sufficiency_supported": kill_pass,
        "stage1_gate_pass": stage_gate_pass,
    }
    return summary, metrics, stage1_decision


def run_decision_sufficiency(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DecisionSufficiencyResult:
    """Run training/calibration and untouched EXP-0005 validation."""

    partitions = config["seed_partitions"]
    case_count = int(partitions["cases_per_seed"])
    target_bounds = config["target_population"]
    training_cases = generate_target_cases(
        partitions["training"], case_count, target_bounds, partition="training"
    )
    calibration_cases = generate_target_cases(
        partitions["calibration"], case_count, target_bounds, partition="calibration"
    )
    validation_cases = generate_target_cases(
        partitions["validation"], case_count, target_bounds, partition="validation"
    )
    regime_count = len(config["sensing"]["noise_regimes"])
    progress_total = regime_count * (
        len(training_cases) + len(calibration_cases) + len(validation_cases)
    )
    progress_state = [0]
    training_features, training_safety, _ = _extract_partition_features(
        training_cases,
        config,
        progress_callback=progress_callback,
        progress_state=progress_state,
        progress_total=progress_total,
    )
    training_outcomes = _simulate_outcomes(training_cases, config)
    calibration_features, calibration_safety, _ = _extract_partition_features(
        calibration_cases,
        config,
        progress_callback=progress_callback,
        progress_state=progress_state,
        progress_total=progress_total,
    )
    calibration_outcomes = _simulate_outcomes(calibration_cases, config)
    regression_models, logistic_models = _fit_models(
        training_features,
        training_outcomes,
        calibration_features,
        calibration_outcomes,
        config,
    )

    validation_features, validation_safety, _ = _extract_partition_features(
        validation_cases,
        config,
        progress_callback=progress_callback,
        progress_state=progress_state,
        progress_total=progress_total,
    )
    # Critical separation: all validation predictions are frozen before the
    # hidden future-contact response is simulated or joined.
    pending_predictions: list[Mapping[str, Any]] = []
    logistic_config = config["predictors"]["multinomial_logistic"]
    for feature in validation_features:
        threshold = _threshold_prediction(feature, config)
        pending_predictions.append(
            _prediction_row(
                feature,
                predictor="threshold_rule",
                feature_set="time_domain",
                prediction=threshold,
                probabilities=None,
                config=config,
            )
        )
        for feature_set, names in FEATURE_SETS.items():
            matrix = _feature_matrix([feature], names)
            bound = _bound_prediction(regression_models[feature_set], matrix, config)
            pending_predictions.append(
                _prediction_row(
                    feature,
                    predictor="outcome_bound",
                    feature_set=feature_set,
                    prediction=bound,
                    probabilities=None,
                    config=config,
                )
            )
            probabilities = _logistic_probabilities(
                logistic_models[feature_set], matrix
            )[0]
            logistic_prediction = {
                "predicted_risk_class": _probability_class(probabilities, logistic_config),
                "predicted_risk_score": float(1.0 - probabilities[0]),
                "predicted_contact_loss_proxy": bool(probabilities[2] >= float(logistic_config["unsafe_probability_threshold"])),
                "predicted_outcomes": {
                    outcome: float("nan") for outcome in config["quantitative_outcomes"]
                },
                "upper_outcomes": {
                    outcome: float("nan") for outcome in config["quantitative_outcomes"]
                },
                "safe_force_lower_bound_n": float("nan"),
            }
            pending_predictions.append(
                _prediction_row(
                    feature,
                    predictor="logistic",
                    feature_set=feature_set,
                    prediction=logistic_prediction,
                    probabilities=probabilities,
                    config=config,
                )
            )

    validation_outcomes = _simulate_outcomes(validation_cases, config)
    validation_rows = _attach_actual_outcomes(
        pending_predictions, validation_outcomes, config
    )
    classification = _classification_summary(validation_rows)
    quantitative = _quantitative_summary(validation_rows, config)
    class_distribution = _class_distribution(validation_outcomes, "validation")
    mass_comparison = _mass_decision_comparison(validation_rows, config)
    dynamics = _dynamics_performance(validation_rows, config)
    primary = _primary_rows(validation_rows)
    false_safe = tuple(row for row in primary if bool(row["false_safe"]))
    feature_space = _feature_space(
        validation_features, validation_outcomes, validation_rows
    )
    representative = _representative_raw(
        validation_cases, validation_outcomes, config
    )
    all_safety = tuple((*training_safety, *calibration_safety, *validation_safety))
    summary, metrics, stage1_decision = _summarize_result(
        validation_rows,
        classification,
        quantitative,
        mass_comparison,
        validation_safety,
        config,
    )
    minimum_fraction = float(
        config["integrity_acceptance"]["minimum_fraction_per_risk_class"]
    )
    validation_seeds = [int(seed) for seed in partitions["validation"]]
    acceptance_checks = {
        "minimum_validation_seeds": len(validation_seeds)
        >= int(config["integrity_acceptance"]["minimum_validation_seeds"]),
        "unique_disjoint_seed_partitions": len(
            set(partitions["training"])
            | set(partitions["calibration"])
            | set(partitions["validation"])
        )
        == len(partitions["training"])
        + len(partitions["calibration"])
        + len(partitions["validation"]),
        "minimum_validation_cases": len(validation_cases)
        >= int(config["integrity_acceptance"]["minimum_validation_cases"]),
        "risk_class_balance": all(
            float(row["fraction"]) >= minimum_fraction for row in class_distribution
        ),
        "validation_predictions_precede_future_maneuver": True,
        "combined_task_excludes_damping_and_mass_estimates": (
            "estimated_damping_signed_log" not in FEATURE_SETS["combined_task"]
            and "estimated_mass_signed_log" not in FEATURE_SETS["combined_task"]
        ),
        "primary_pipeline_is_causal": str(config["sensing"]["primary_pipeline"])
        == "causal_low_pass",
        "no_noncausal_primary_result": bool(config["sensing"]["no_noncausal_primary_result"]),
        "no_validation_probe_safety_events": not validation_safety,
        "representative_all_classes_present": len(
            set(str(value) for value in representative["risk_class"])
        )
        == 3,
    }
    metrics = {**metrics, "all_partition_probe_safety_event_count": len(all_safety)}
    return DecisionSufficiencyResult(
        validation_rows=validation_rows,
        classification_summary=classification,
        quantitative_summary=quantitative,
        class_distribution=class_distribution,
        mass_decision_comparison=mass_comparison,
        dynamics_performance=dynamics,
        false_safe_cases=false_safe,
        representative_raw=representative,
        feature_space_rows=feature_space,
        summary=summary,
        metrics=metrics,
        safety_events=all_safety,
        acceptance_checks=acceptance_checks,
        success=bool(all(acceptance_checks.values())),
        stage1_decision=stage1_decision,
    )
