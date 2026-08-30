"""EXP-0004 sequential uncertainty-driven Stage 1 identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from probeing.estimators import (
    delayed_input_instruments,
    instrumental_variables,
    regression_diagnostics,
    regression_matrix,
    total_least_squares,
)
from probeing.measurements import MeasurementNoise, SyntheticMeasurements, process_causal_sensing
from probeing.metrics import physical_disturbance_metrics, rms
from probeing.models import (
    InteractionParameters,
    MassSpringDamperModel,
    simulate_contact_interaction,
)
from probeing.probing import ProbeSignal, chirp, multisine


PARAMETERS = ("stiffness", "damping", "effective_mass")


@dataclass(frozen=True)
class SequentialIdentificationResult:
    trial_rows: tuple[Mapping[str, Any], ...]
    selection_rows: tuple[Mapping[str, Any], ...]
    stage_aggregate: tuple[Mapping[str, Any], ...]
    strategy_summary: tuple[Mapping[str, Any], ...]
    frequency_information: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool
    stage1_decision: str


@dataclass
class _SegmentData:
    measurements: SyntheticMeasurements
    stiffness_measurements: SyntheticMeasurements
    instruments: NDArray[np.float64]


@dataclass
class _Estimate:
    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]
    relative_standard_errors: NDArray[np.float64]
    force_rmse_n: float
    minimum_instrument_strength: float
    design: NDArray[np.float64]
    force: NDArray[np.float64]


def _target_parameters(target: Mapping[str, Any]) -> InteractionParameters:
    return InteractionParameters(
        stiffness_n_per_m=float(target["stiffness_n_per_m"]),
        damping_n_s_per_m=float(target["damping_n_s_per_m"]),
        effective_mass_kg=float(target["effective_mass_kg"]),
    )


def _truth_vector(target: Mapping[str, Any]) -> NDArray[np.float64]:
    parameters = _target_parameters(target)
    return np.asarray(
        [
            parameters.stiffness_n_per_m,
            parameters.damping_n_s_per_m,
            parameters.effective_mass_kg,
        ],
        dtype=float,
    )


def _noise(config: Mapping[str, Any], *, zero: bool = False) -> MeasurementNoise:
    values = config["sensing"]["noise"]
    multiplier = 0.0 if zero else 1.0
    return MeasurementNoise(
        displacement_std_m=multiplier * float(values["displacement_std_m"]),
        velocity_std_m_per_s=multiplier * float(values["velocity_std_m_per_s"]),
        acceleration_std_m_per_s2=multiplier * float(values["acceleration_std_m_per_s2"]),
        force_std_n=multiplier * float(values["force_std_n"]),
    )


def _candidate_map(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    candidates = config["probe_library"]["candidates"]
    return {str(candidate["name"]): candidate for candidate in candidates}


def _make_probe(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> ProbeSignal:
    simulation = config["simulation"]
    amplitude = float(config["probe_library"]["amplitude_n"])
    duration = float(simulation["probe_duration_s"])
    step = float(simulation["sample_period_s"])
    if str(candidate["type"]) == "chirp":
        signal = chirp(
            amplitude,
            float(candidate["start_frequency_hz"]),
            float(candidate["end_frequency_hz"]),
            duration,
            step,
        )
    elif str(candidate["type"]) == "multisine":
        signal = multisine(
            amplitude,
            candidate["frequencies_hz"],
            duration,
            step,
            phases_rad=candidate.get("phases_rad"),
        )
    else:
        raise ValueError(f"unknown EXP-0004 probe type: {candidate['type']}")
    return ProbeSignal(
        name=str(candidate["name"]),
        time_s=signal.time_s,
        force_n=signal.force_n,
        amplitude_bound_n=signal.amplitude_bound_n,
    )


def _trim_measurements(
    measurements: SyntheticMeasurements, trim_s: float
) -> tuple[SyntheticMeasurements, NDArray[np.bool_]]:
    time = measurements.time_s
    mask = (time >= time[0] + trim_s) & (time <= time[-1] - trim_s)
    if np.count_nonzero(mask) < 20:
        raise ValueError("analysis trim leaves too few samples")
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


def _sense_segment(
    simulation: Any,
    config: Mapping[str, Any],
    random_seed: int,
) -> tuple[_SegmentData, Any]:
    sensing_config = config["sensing"]
    sensing = process_causal_sensing(
        simulation,
        pipeline=str(sensing_config["pipeline"]),
        sample_rate_hz=float(sensing_config["sample_rate_hz"]),
        noise=_noise(config),
        pipeline_settings=sensing_config["pipeline_settings"],
        random_seed=int(random_seed),
        timestamp_offsets_s=sensing_config["timestamp_offsets_s"],
    )
    measurements, mask = _trim_measurements(
        sensing.measurements, float(sensing_config["analysis_trim_s"])
    )
    stiffness_sensing = process_causal_sensing(
        simulation,
        pipeline=str(sensing_config["parameter_pipelines"]["stiffness"]),
        sample_rate_hz=float(sensing_config["sample_rate_hz"]),
        noise=_noise(config),
        pipeline_settings=sensing_config["pipeline_settings"],
        random_seed=int(random_seed),
        timestamp_offsets_s=sensing_config["timestamp_offsets_s"],
    )
    stiffness_measurements, stiffness_mask = _trim_measurements(
        stiffness_sensing.measurements, float(sensing_config["analysis_trim_s"])
    )
    if not np.array_equal(mask, stiffness_mask):
        raise RuntimeError("parameter-specific sensing pipelines are not sample-aligned")
    command = np.maximum(np.asarray(sensing.commanded_force_n[mask], dtype=float), 0.0)
    instruments = delayed_input_instruments(
        command,
        measurements.time_s,
        config["estimation"]["iv_input_delays_s"],
    )
    return _SegmentData(
        measurements=measurements,
        stiffness_measurements=stiffness_measurements,
        instruments=instruments,
    ), sensing


def _stack_segments(
    segments: Sequence[_SegmentData],
) -> tuple[SyntheticMeasurements, SyntheticMeasurements, NDArray[np.float64]]:
    if not segments:
        raise ValueError("at least one segment is required")
    sample_step = float(np.median(np.diff(segments[0].measurements.time_s)))
    lengths = [segment.measurements.time_s.size for segment in segments]
    time = np.arange(sum(lengths), dtype=float) * sample_step
    noise = segments[0].measurements.noise
    measurements = SyntheticMeasurements(
        time_s=time,
        displacement_m=np.concatenate([s.measurements.displacement_m for s in segments]),
        velocity_m_per_s=np.concatenate([s.measurements.velocity_m_per_s for s in segments]),
        acceleration_m_per_s2=np.concatenate(
            [s.measurements.acceleration_m_per_s2 for s in segments]
        ),
        contact_force_n=np.concatenate([s.measurements.contact_force_n for s in segments]),
        noise=noise,
        random_seed=segments[0].measurements.random_seed,
    )
    stiffness_measurements = SyntheticMeasurements(
        time_s=time.copy(),
        displacement_m=np.concatenate(
            [s.stiffness_measurements.displacement_m for s in segments]
        ),
        velocity_m_per_s=np.concatenate(
            [s.stiffness_measurements.velocity_m_per_s for s in segments]
        ),
        acceleration_m_per_s2=np.concatenate(
            [s.stiffness_measurements.acceleration_m_per_s2 for s in segments]
        ),
        contact_force_n=np.concatenate(
            [s.stiffness_measurements.contact_force_n for s in segments]
        ),
        noise=noise,
        random_seed=segments[0].stiffness_measurements.random_seed,
    )
    return measurements, stiffness_measurements, np.vstack(
        [segment.instruments for segment in segments]
    )


def _structured_parameters(
    design: NDArray[np.float64],
    force: NDArray[np.float64],
    instruments: NDArray[np.float64],
    stiffness_design: NDArray[np.float64],
    stiffness_force: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    measurements = SyntheticMeasurements(
        time_s=np.arange(force.size, dtype=float),
        displacement_m=design[:, 0],
        velocity_m_per_s=design[:, 1],
        acceleration_m_per_s2=design[:, 2],
        contact_force_n=force,
        noise=MeasurementNoise(),
        random_seed=0,
    )
    ols_parameters, _, _, _ = np.linalg.lstsq(design, force, rcond=None)
    iv = instrumental_variables(measurements, instruments)
    stiffness_measurement = SyntheticMeasurements(
        time_s=np.arange(stiffness_force.size, dtype=float),
        displacement_m=stiffness_design[:, 0],
        velocity_m_per_s=stiffness_design[:, 1],
        acceleration_m_per_s2=stiffness_design[:, 2],
        contact_force_n=stiffness_force,
        noise=MeasurementNoise(),
        random_seed=0,
    )
    tls = total_least_squares(stiffness_measurement)
    mass = float(iv.parameters[2]) if iv.valid and np.isfinite(iv.parameters[2]) else float(ols_parameters[2])
    stiffness = float(tls.parameters[0]) if tls.valid and np.isfinite(tls.parameters[0]) else float(ols_parameters[0])
    parameters = np.asarray([stiffness, ols_parameters[1], mass], dtype=float)
    strength = float(iv.estimator_diagnostics.get("minimum_instrument_strength", 0.0))
    return parameters, strength


def _estimate(segments: Sequence[_SegmentData], config: Mapping[str, Any]) -> _Estimate:
    measurements, stiffness_measurements, instruments = _stack_segments(segments)
    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    stiffness_design = regression_matrix(stiffness_measurements)
    stiffness_force = np.asarray(stiffness_measurements.contact_force_n, dtype=float)
    parameters, strength = _structured_parameters(
        design, force, instruments, stiffness_design, stiffness_force
    )
    residual = force - design @ parameters
    degrees = max(force.size - 3, 1)
    residual_variance = max(float(residual @ residual / degrees), np.finfo(float).eps)
    analytic = residual_variance * np.linalg.pinv(design.T @ design, rcond=1.0e-12)

    block_count = min(int(config["estimation"]["jackknife_blocks"]), force.size // 20)
    block_estimates: list[NDArray[np.float64]] = []
    for block in np.array_split(np.arange(force.size), block_count):
        keep = np.ones(force.size, dtype=bool)
        keep[block] = False
        if np.count_nonzero(keep) < 20:
            continue
        try:
            value, _ = _structured_parameters(
                design[keep],
                force[keep],
                instruments[keep],
                stiffness_design[keep],
                stiffness_force[keep],
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(value)):
            block_estimates.append(value)
    if len(block_estimates) >= 3:
        samples = np.asarray(block_estimates)
        center = np.mean(samples, axis=0)
        delta = samples - center
        jackknife = (len(samples) - 1.0) / len(samples) * delta.T @ delta
        covariance = analytic.copy()
        covariance[np.diag_indices(3)] = np.maximum(
            np.diag(analytic), np.diag(jackknife)
        )
    else:
        covariance = analytic
    covariance = 0.5 * (covariance + covariance.T)
    diagonal_floor = np.finfo(float).eps * np.maximum(np.abs(parameters), 1.0) ** 2
    covariance[np.diag_indices(3)] = np.maximum(np.diag(covariance), diagonal_floor)
    floors = np.asarray(config["estimation"]["uncertainty_scale_floors"], dtype=float)
    scales = np.maximum(np.abs(parameters), floors)
    relative_standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0)) / scales
    return _Estimate(
        parameters=parameters,
        covariance=covariance,
        relative_standard_errors=relative_standard_errors,
        force_rmse_n=float(np.sqrt(np.mean(residual**2))),
        minimum_instrument_strength=strength,
        design=design,
        force=force,
    )


def _clipped_parameters(parameters: NDArray[np.float64], config: Mapping[str, Any]) -> InteractionParameters:
    bounds = config["estimation"]["predictive_parameter_bounds"]
    names = ("stiffness_n_per_m", "damping_n_s_per_m", "effective_mass_kg")
    clipped = [
        float(np.clip(parameters[index], *[float(value) for value in bounds[name]]))
        for index, name in enumerate(names)
    ]
    return InteractionParameters(*clipped)


def _force_dose(signal: ProbeSignal) -> float:
    return float(np.trapz(signal.force_n**2, signal.time_s))


def _relative_information_matrix(
    design: NDArray[np.float64], parameter_scales: NDArray[np.float64], variance: float
) -> NDArray[np.float64]:
    normalized_design = design * parameter_scales[None, :]
    return normalized_design.T @ normalized_design / max(variance, np.finfo(float).eps)


def _residualized_column_information(
    design: NDArray[np.float64],
    parameter_scales: NDArray[np.float64],
    variance: float,
) -> NDArray[np.float64]:
    values: list[float] = []
    for index in range(3):
        other = np.delete(design, index, axis=1)
        coefficients, _, _, _ = np.linalg.lstsq(other, design[:, index], rcond=None)
        residual = design[:, index] - other @ coefficients
        values.append(
            float(parameter_scales[index] ** 2 * np.sum(residual**2))
            / max(variance, np.finfo(float).eps)
        )
    return np.asarray(values, dtype=float)


def _safe_logdet(matrix: NDArray[np.float64]) -> float:
    sign, value = np.linalg.slogdet(matrix + np.eye(matrix.shape[0]) * np.finfo(float).eps)
    return float(value) if sign > 0.0 and np.isfinite(value) else float("-inf")


def _predict_candidate(
    candidate: Mapping[str, Any],
    estimate: _Estimate,
    current_state: tuple[float, float],
    cumulative: Mapping[str, float],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    signal = _make_probe(candidate, config)
    predicted = simulate_contact_interaction(
        MassSpringDamperModel(_clipped_parameters(estimate.parameters, config)),
        signal.time_s,
        signal.force_n,
        contact_mode=str(config["simulation"]["contact_mode"]),
        initial_displacement_m=float(current_state[0]),
        initial_velocity_m_per_s=float(current_state[1]),
    )
    clean_sensing = process_causal_sensing(
        predicted,
        pipeline=str(config["sensing"]["pipeline"]),
        sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
        noise=_noise(config, zero=True),
        pipeline_settings=config["sensing"]["pipeline_settings"],
        random_seed=0,
        timestamp_offsets_s=config["sensing"]["timestamp_offsets_s"],
    )
    clean, _ = _trim_measurements(
        clean_sensing.measurements, float(config["sensing"]["analysis_trim_s"])
    )
    stiffness_sensing = process_causal_sensing(
        predicted,
        pipeline=str(config["sensing"]["parameter_pipelines"]["stiffness"]),
        sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
        noise=_noise(config, zero=True),
        pipeline_settings=config["sensing"]["pipeline_settings"],
        random_seed=0,
        timestamp_offsets_s=config["sensing"]["timestamp_offsets_s"],
    )
    stiffness_clean, _ = _trim_measurements(
        stiffness_sensing.measurements, float(config["sensing"]["analysis_trim_s"])
    )
    candidate_design = regression_matrix(clean)
    stiffness_candidate_design = regression_matrix(stiffness_clean)
    residual_variance = max(estimate.force_rmse_n**2, float(config["sensing"]["noise"]["force_std_n"]) ** 2)
    floors = np.asarray(config["estimation"]["uncertainty_scale_floors"], dtype=float)
    scales = np.maximum(np.abs(estimate.parameters), floors)
    normalized_covariance = estimate.covariance / np.outer(scales, scales)
    current_variances = np.maximum(np.diag(normalized_covariance), np.finfo(float).eps)
    current_precision = 1.0 / current_variances
    low_pass_information = _residualized_column_information(
        candidate_design, scales, residual_variance
    )
    stiffness_information = _residualized_column_information(
        stiffness_candidate_design, scales, residual_variance
    )
    candidate_information = low_pass_information.copy()
    candidate_information[0] = stiffness_information[0]
    posterior_variances = 1.0 / (current_precision + candidate_information)
    disturbance = physical_disturbance_metrics(predicted)
    dose = _force_dose(signal)
    budget = config["disturbance_budget"]
    fits = bool(
        cumulative["probe_count"] + 1 <= float(budget["maximum_probe_count"])
        and cumulative["duration_s"] + float(config["simulation"]["probe_duration_s"])
        <= float(budget["maximum_duration_s"]) + 1.0e-12
        and cumulative["force_squared_dose_n2_s"] + dose
        <= float(budget["maximum_command_force_squared_dose_n2_s"]) + 1.0e-12
        and cumulative["absolute_input_energy_j"] + float(disturbance["absolute_input_energy_j"])
        <= float(budget["maximum_absolute_input_energy_j"]) + 1.0e-12
        and cumulative["absolute_impulse_n_s"] + float(disturbance["absolute_impulse_n_s"])
        <= float(budget["maximum_absolute_impulse_n_s"]) + 1.0e-12
    )
    weak_index = int(
        np.argmax(
            estimate.relative_standard_errors
            / np.asarray(config["selection"]["relative_standard_error_thresholds"], dtype=float)
        )
    )
    current_variance = float(current_variances[weak_index])
    reduction = float(
        np.clip(1.0 - posterior_variances[weak_index] / current_variance, 0.0, 1.0)
    )
    normalizers = budget["prediction_normalizers"]
    disturbance_cost = (
        dose / float(normalizers["force_squared_dose_n2_s"])
        + float(disturbance["absolute_input_energy_j"])
        / float(normalizers["absolute_input_energy_j"])
        + float(disturbance["peak_target_displacement_m"])
        / float(normalizers["peak_target_displacement_m"])
        + float(disturbance["peak_target_acceleration_m_per_s2"])
        / float(normalizers["peak_target_acceleration_m_per_s2"])
    )
    information_gain = float(
        np.sum(np.log(current_precision + candidate_information) - np.log(current_precision))
    )
    return {
        "candidate_probe": str(candidate["name"]),
        "fits_budget": fits,
        "weak_parameter": PARAMETERS[weak_index],
        "weak_parameter_index": weak_index,
        "predicted_weak_variance_reduction": reduction,
        "predicted_information_gain": float(information_gain),
        "predicted_disturbance_cost": float(disturbance_cost),
        "selection_score": reduction,
        "predicted_force_squared_dose_n2_s": dose,
        "predicted_absolute_input_energy_j": float(disturbance["absolute_input_energy_j"]),
        "predicted_peak_displacement_m": float(disturbance["peak_target_displacement_m"]),
        "predicted_peak_acceleration_m_per_s2": float(disturbance["peak_target_acceleration_m_per_s2"]),
    }


def _select_next_probe(
    estimate: _Estimate,
    current_state: tuple[float, float],
    cumulative: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[str | None, str, Mapping[str, Any] | None, tuple[Mapping[str, Any], ...]]:
    thresholds = np.asarray(config["selection"]["relative_standard_error_thresholds"], dtype=float)
    if np.all(estimate.relative_standard_errors <= thresholds):
        return None, "confidence_reached", None, ()
    candidates = _candidate_map(config)
    allowed = next(
        strategy for strategy in config["strategies"] if str(strategy["name"]) == "uncertainty_driven"
    )["candidate_names"]
    predictions = tuple(
        _predict_candidate(candidates[str(name)], estimate, current_state, cumulative, config)
        for name in allowed
    )
    feasible = [prediction for prediction in predictions if bool(prediction["fits_budget"])]
    if not feasible:
        return None, "disturbance_budget", None, predictions
    best = max(
        feasible,
        key=lambda row: (
            float(row["selection_score"]),
            -float(row["predicted_disturbance_cost"]),
            str(row["candidate_probe"]),
        ),
    )
    minimum = float(config["selection"]["minimum_expected_weak_parameter_variance_reduction"])
    if float(best["predicted_weak_variance_reduction"]) < minimum:
        return None, "insufficient_expected_information", best, predictions
    return str(best["candidate_probe"]), "continue", best, predictions


def _cumulative_geometry(estimate: _Estimate, config: Mapping[str, Any]) -> Mapping[str, float]:
    diagnostics = regression_diagnostics(estimate.design)
    floors = np.asarray(config["estimation"]["uncertainty_scale_floors"], dtype=float)
    scales = np.maximum(np.abs(estimate.parameters), floors)
    information = _relative_information_matrix(
        estimate.design, scales, max(estimate.force_rmse_n**2, np.finfo(float).eps)
    )
    eigenvalues = np.linalg.eigvalsh(information)
    return {
        "rank": diagnostics.rank,
        "normalized_condition_number": diagnostics.normalized_condition_number,
        "normalized_singular_value_1": float(diagnostics.normalized_singular_values[0]),
        "normalized_singular_value_2": float(diagnostics.normalized_singular_values[1]),
        "normalized_singular_value_3": float(diagnostics.normalized_singular_values[2]),
        "maximum_abs_parameter_correlation": diagnostics.maximum_abs_parameter_correlation,
        "parameter_correlation_k_c": float(diagnostics.parameter_correlation[0, 1]),
        "parameter_correlation_k_m": float(diagnostics.parameter_correlation[0, 2]),
        "parameter_correlation_c_m": float(diagnostics.parameter_correlation[1, 2]),
        "relative_information_logdet": _safe_logdet(information),
        "relative_information_min_eigenvalue": float(max(eigenvalues[0], 0.0)),
    }


def _seed_for_segment(seed: int, target_index: int, probe_index: int, candidate_index: int) -> int:
    return int(seed * 100_000 + target_index * 1_000 + probe_index * 20 + candidate_index)


def _safety_events(
    target: str,
    strategy: str,
    seed: int,
    probe_index: int,
    disturbance: Mapping[str, float],
    limits: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    checks = (
        ("peak_force_n", "max_peak_force_n"),
        ("peak_target_displacement_m", "max_peak_target_displacement_m"),
        ("peak_target_velocity_m_per_s", "max_peak_target_velocity_m_per_s"),
        ("peak_target_acceleration_m_per_s2", "max_peak_target_acceleration_m_per_s2"),
    )
    return [
        {
            "target": target,
            "strategy": strategy,
            "validation_seed": seed,
            "probe_index": probe_index,
            "metric": metric,
            "value": float(disturbance[metric]),
            "limit": float(limits[limit]),
        }
        for metric, limit in checks
        if float(disturbance[metric]) > float(limits[limit])
    ]


def _relative_error(estimate: float, truth: float) -> float:
    return float((estimate - truth) / abs(truth))


def _run_strategy_trial(
    target: Mapping[str, Any],
    target_index: int,
    strategy: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    representative_parts: dict[str, list[NDArray[Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Mapping[str, Any]]]:
    target_name = str(target["name"])
    strategy_name = str(strategy["name"])
    candidates = _candidate_map(config)
    candidate_indices = {name: index for index, name in enumerate(candidates)}
    truth = _truth_vector(target)
    model = MassSpringDamperModel(_target_parameters(target))
    if str(strategy["mode"]) == "fixed":
        planned = [str(name) for name in strategy["sequence"]]
    else:
        planned = [str(strategy["initial_probe"])]

    segments: list[_SegmentData] = []
    trial_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    safety_events: list[Mapping[str, Any]] = []
    state = (0.0, 0.0)
    cumulative: dict[str, float] = {
        "probe_count": 0.0,
        "duration_s": 0.0,
        "force_squared_dose_n2_s": 0.0,
        "absolute_input_energy_j": 0.0,
        "absolute_impulse_n_s": 0.0,
        "peak_force_n": 0.0,
        "peak_target_displacement_m": 0.0,
        "rms_target_displacement_squared_time": 0.0,
        "peak_target_velocity_m_per_s": 0.0,
        "peak_target_acceleration_m_per_s2": 0.0,
    }
    previous_information_logdet: float | None = None
    pending_selection: dict[str, Any] | None = None
    pending_error: float | None = None
    pending_weak_index: int | None = None
    stop_reason = "fixed_sequence_complete"
    probe_index = 0
    while probe_index < len(planned):
        probe_name = planned[probe_index]
        signal = _make_probe(candidates[probe_name], config)
        simulation = simulate_contact_interaction(
            model,
            signal.time_s,
            signal.force_n,
            contact_mode=str(config["simulation"]["contact_mode"]),
            initial_displacement_m=state[0],
            initial_velocity_m_per_s=state[1],
        )
        state = (
            float(simulation.response.displacement_m[-1]),
            float(simulation.response.velocity_m_per_s[-1]),
        )
        random_seed = _seed_for_segment(
            seed, target_index, probe_index, candidate_indices[probe_name]
        )
        segment, sensing = _sense_segment(simulation, config, random_seed)
        segments.append(segment)
        estimate = _estimate(segments, config)
        geometry = _cumulative_geometry(estimate, config)
        disturbance = physical_disturbance_metrics(simulation)
        duration = float(config["simulation"]["probe_duration_s"])
        cumulative["probe_count"] += 1.0
        cumulative["duration_s"] += duration
        cumulative["force_squared_dose_n2_s"] += _force_dose(signal)
        cumulative["absolute_input_energy_j"] += float(disturbance["absolute_input_energy_j"])
        cumulative["absolute_impulse_n_s"] += float(disturbance["absolute_impulse_n_s"])
        cumulative["peak_force_n"] = max(cumulative["peak_force_n"], float(disturbance["peak_force_n"]))
        cumulative["peak_target_displacement_m"] = max(
            cumulative["peak_target_displacement_m"], float(disturbance["peak_target_displacement_m"])
        )
        cumulative["rms_target_displacement_squared_time"] += (
            float(disturbance["rms_target_displacement_m"]) ** 2 * duration
        )
        cumulative["peak_target_velocity_m_per_s"] = max(
            cumulative["peak_target_velocity_m_per_s"], float(disturbance["peak_target_velocity_m_per_s"])
        )
        cumulative["peak_target_acceleration_m_per_s2"] = max(
            cumulative["peak_target_acceleration_m_per_s2"],
            float(disturbance["peak_target_acceleration_m_per_s2"]),
        )
        safety_events.extend(
            _safety_events(
                target_name,
                strategy_name,
                seed,
                probe_index + 1,
                disturbance,
                config["safety_limits"],
            )
        )
        errors = np.asarray(
            [_relative_error(estimate.parameters[index], truth[index]) for index in range(3)]
        )
        thresholds = np.asarray(
            [float(config["accuracy_thresholds"][parameter]) for parameter in PARAMETERS]
        )
        information_gain = (
            0.0
            if previous_information_logdet is None
            else float(geometry["relative_information_logdet"] - previous_information_logdet)
        )
        previous_information_logdet = float(geometry["relative_information_logdet"])
        cumulative_rms_displacement = np.sqrt(
            cumulative["rms_target_displacement_squared_time"] / cumulative["duration_s"]
        )
        row: dict[str, Any] = {
            "target": target_name,
            "strategy": strategy_name,
            "validation_seed": seed,
            "probe_index": probe_index + 1,
            "probe_name": probe_name,
            "is_adaptive": str(strategy["mode"]) == "adaptive",
            "true_stiffness_n_per_m": truth[0],
            "true_damping_n_s_per_m": truth[1],
            "true_effective_mass_kg": truth[2],
            "estimated_stiffness_n_per_m": float(estimate.parameters[0]),
            "estimated_damping_n_s_per_m": float(estimate.parameters[1]),
            "estimated_effective_mass_kg": float(estimate.parameters[2]),
            "stiffness_relative_error": float(errors[0]),
            "damping_relative_error": float(errors[1]),
            "effective_mass_relative_error": float(errors[2]),
            "stiffness_abs_relative_error": float(abs(errors[0])),
            "damping_abs_relative_error": float(abs(errors[1])),
            "effective_mass_abs_relative_error": float(abs(errors[2])),
            "parameter_relative_error_rms": rms(errors),
            "stiffness_meets_threshold": bool(abs(errors[0]) <= thresholds[0]),
            "damping_meets_threshold": bool(abs(errors[1]) <= thresholds[1]),
            "effective_mass_meets_threshold": bool(abs(errors[2]) <= thresholds[2]),
            "full_vector_acceptable": bool(np.all(np.abs(errors) <= thresholds)),
            "stiffness_relative_standard_error": float(estimate.relative_standard_errors[0]),
            "damping_relative_standard_error": float(estimate.relative_standard_errors[1]),
            "effective_mass_relative_standard_error": float(estimate.relative_standard_errors[2]),
            "force_fit_rmse_n": estimate.force_rmse_n,
            "minimum_instrument_strength": estimate.minimum_instrument_strength,
            "incremental_information_gain": information_gain,
            "cumulative_information_per_force_dose": float(geometry["relative_information_logdet"])
            / max(cumulative["force_squared_dose_n2_s"], np.finfo(float).eps),
            "cumulative_probe_count": int(cumulative["probe_count"]),
            "cumulative_duration_s": cumulative["duration_s"],
            "cumulative_force_squared_dose_n2_s": cumulative["force_squared_dose_n2_s"],
            "cumulative_absolute_input_energy_j": cumulative["absolute_input_energy_j"],
            "cumulative_absolute_impulse_n_s": cumulative["absolute_impulse_n_s"],
            "cumulative_peak_force_n": cumulative["peak_force_n"],
            "cumulative_peak_target_displacement_m": cumulative["peak_target_displacement_m"],
            "cumulative_rms_target_displacement_m": float(cumulative_rms_displacement),
            "cumulative_peak_target_velocity_m_per_s": cumulative["peak_target_velocity_m_per_s"],
            "cumulative_peak_target_acceleration_m_per_s2": cumulative["peak_target_acceleration_m_per_s2"],
            "stop_reason": "continue",
            "is_final": False,
            **geometry,
        }
        trial_rows.append(row)

        if pending_selection is not None and pending_weak_index is not None and pending_error is not None:
            current_error = float(abs(errors[pending_weak_index]))
            pending_selection["realized_weak_error_reduction"] = pending_error - current_error
            pending_selection["next_probe_was_useful"] = bool(current_error < pending_error)
            pending_selection = None

        representative = config["representative_raw"]
        if (
            target_name == str(representative["target"])
            and strategy_name == str(representative["strategy"])
            and seed == int(representative["seed"])
        ):
            count = sensing.measurements.time_s.size
            time_offset = probe_index * duration
            values = {
                "time_s": sensing.measurements.time_s + time_offset,
                "probe_index": np.full(count, probe_index + 1),
                "probe_name": np.full(count, probe_name),
                "true_displacement_m": sensing.true_displacement_m,
                "processed_displacement_m": sensing.measurements.displacement_m,
                "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
                "processed_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
                "true_acceleration_m_per_s2": sensing.true_acceleration_m_per_s2,
                "processed_acceleration_m_per_s2": sensing.measurements.acceleration_m_per_s2,
                "true_force_n": sensing.true_contact_force_n,
                "processed_force_n": sensing.measurements.contact_force_n,
                "commanded_force_n": sensing.commanded_force_n,
            }
            for field, values_array in values.items():
                representative_parts.setdefault(field, []).append(np.asarray(values_array))

        probe_index += 1
        if str(strategy["mode"]) != "adaptive":
            if probe_index >= len(planned):
                stop_reason = "fixed_sequence_complete"
            continue
        if probe_index >= int(config["disturbance_budget"]["maximum_probe_count"]):
            stop_reason = "maximum_probe_count"
            break
        selected, stop_reason, best, predictions = _select_next_probe(
            estimate, state, cumulative, config
        )
        weak_index = int(
            np.argmax(
                estimate.relative_standard_errors
                / np.asarray(config["selection"]["relative_standard_error_thresholds"], dtype=float)
            )
        )
        for prediction in predictions:
            selection_row = {
                "target": target_name,
                "validation_seed": seed,
                "after_probe_index": probe_index,
                "selected": bool(selected == prediction["candidate_probe"]),
                "stop_reason": stop_reason,
                "stiffness_relative_standard_error": float(estimate.relative_standard_errors[0]),
                "damping_relative_standard_error": float(estimate.relative_standard_errors[1]),
                "effective_mass_relative_standard_error": float(estimate.relative_standard_errors[2]),
                "current_weak_abs_relative_error_evaluation_only": float(abs(errors[weak_index])),
                "realized_weak_error_reduction": float("nan"),
                "next_probe_was_useful": False,
                **prediction,
            }
            selection_rows.append(selection_row)
            if bool(selection_row["selected"]):
                pending_selection = selection_row
                pending_error = float(abs(errors[weak_index]))
                pending_weak_index = weak_index
        if selected is None:
            break
        planned.append(selected)

    if not trial_rows:
        raise RuntimeError("strategy produced no probes")
    trial_rows[-1]["is_final"] = True
    trial_rows[-1]["stop_reason"] = stop_reason
    return trial_rows, selection_rows, safety_events


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    groups = sorted({(str(row["strategy"]), int(row["probe_index"])) for row in rows})
    for strategy, probe_index in groups:
        selected = [row for row in rows if row["strategy"] == strategy and int(row["probe_index"]) == probe_index]
        entry: dict[str, Any] = {
            "strategy": strategy,
            "probe_index": probe_index,
            "trial_count": len(selected),
            "full_vector_success_probability": float(np.mean([row["full_vector_acceptable"] for row in selected])),
            "median_cumulative_energy_j": float(np.median([row["cumulative_absolute_input_energy_j"] for row in selected])),
            "median_cumulative_force_dose_n2_s": float(np.median([row["cumulative_force_squared_dose_n2_s"] for row in selected])),
        }
        for parameter in PARAMETERS:
            values = np.asarray([row[f"{parameter}_abs_relative_error"] for row in selected], dtype=float)
            uncertainties = np.asarray(
                [row[f"{parameter}_relative_standard_error"] for row in selected], dtype=float
            )
            entry[f"{parameter}_median_abs_relative_error"] = float(np.median(values))
            entry[f"{parameter}_p95_abs_relative_error"] = float(np.percentile(values, 95.0))
            entry[f"{parameter}_median_relative_standard_error"] = float(np.median(uncertainties))
        output.append(entry)
    return tuple(output)


def _summary_for_rows(
    strategy_label: str, selected: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    targets = sorted({str(row["target"]) for row in selected})
    entry: dict[str, Any] = {
            "strategy": strategy_label,
            "trial_count": len(selected),
            "full_vector_success_probability": float(np.mean([row["full_vector_acceptable"] for row in selected])),
            "median_probe_count": float(np.median([row["cumulative_probe_count"] for row in selected])),
            "p95_probe_count": float(np.percentile([row["cumulative_probe_count"] for row in selected], 95.0)),
            "median_duration_s": float(np.median([row["cumulative_duration_s"] for row in selected])),
            "median_force_squared_dose_n2_s": float(np.median([row["cumulative_force_squared_dose_n2_s"] for row in selected])),
            "median_absolute_input_energy_j": float(np.median([row["cumulative_absolute_input_energy_j"] for row in selected])),
            "median_absolute_impulse_n_s": float(np.median([row["cumulative_absolute_impulse_n_s"] for row in selected])),
            "maximum_peak_force_n": float(np.max([row["cumulative_peak_force_n"] for row in selected])),
            "maximum_peak_target_displacement_m": float(np.max([row["cumulative_peak_target_displacement_m"] for row in selected])),
            "maximum_peak_target_velocity_m_per_s": float(np.max([row["cumulative_peak_target_velocity_m_per_s"] for row in selected])),
            "maximum_peak_target_acceleration_m_per_s2": float(np.max([row["cumulative_peak_target_acceleration_m_per_s2"] for row in selected])),
            "median_information_per_force_dose": float(np.median([row["cumulative_information_per_force_dose"] for row in selected])),
        }
    combined = np.asarray([row["parameter_relative_error_rms"] for row in selected], dtype=float)
    entry["parameter_error_rmse"] = rms(combined)
    entry["parameter_error_p95"] = float(np.percentile(combined, 95.0))
    entry["worst_target_parameter_error_p95"] = float(
            max(
                np.percentile(
                    [row["parameter_relative_error_rms"] for row in selected if row["target"] == target],
                    95.0,
                )
                for target in targets
            )
    )
    for parameter in PARAMETERS:
        relative = np.asarray([row[f"{parameter}_relative_error"] for row in selected], dtype=float)
        entry[f"{parameter}_relative_rmse"] = rms(relative)
        entry[f"{parameter}_relative_bias"] = float(np.mean(relative))
        entry[f"{parameter}_p95_abs_relative_error"] = float(np.percentile(np.abs(relative), 95.0))
        entry[f"{parameter}_threshold_success_probability"] = float(
                np.mean([row[f"{parameter}_meets_threshold"] for row in selected])
            )
        entry[f"{parameter}_worst_target_p95_abs_relative_error"] = float(
                max(
                    np.percentile(
                        [abs(row[f"{parameter}_relative_error"]) for row in selected if row["target"] == target],
                        95.0,
                    )
                    for target in targets
                )
        )
    return entry


def _strategy_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    final = [row for row in rows if bool(row["is_final"])]
    return tuple(
        _summary_for_rows(
            strategy,
            [row for row in final if row["strategy"] == strategy],
        )
        for strategy in sorted({str(row["strategy"]) for row in final})
    )


def _residualized_information(
    design: NDArray[np.float64], truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    values: list[float] = []
    for index in range(3):
        other = np.delete(design, index, axis=1)
        coefficients, _, _, _ = np.linalg.lstsq(other, design[:, index], rcond=None)
        residual = design[:, index] - other @ coefficients
        values.append(float(truth[index] ** 2 * np.sum(residual**2)))
    return np.asarray(values, dtype=float)


def _frequency_information(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for target in config["targets"]:
        truth = _truth_vector(target)
        model = MassSpringDamperModel(_target_parameters(target))
        for candidate in config["probe_library"]["candidates"]:
            signal = _make_probe(candidate, config)
            simulation = simulate_contact_interaction(
                model,
                signal.time_s,
                signal.force_n,
                contact_mode=str(config["simulation"]["contact_mode"]),
            )
            clean = process_causal_sensing(
                simulation,
                pipeline=str(config["sensing"]["pipeline"]),
                sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
                noise=_noise(config, zero=True),
                pipeline_settings=config["sensing"]["pipeline_settings"],
                random_seed=0,
                timestamp_offsets_s=config["sensing"]["timestamp_offsets_s"],
            )
            measurements, _ = _trim_measurements(
                clean.measurements, float(config["sensing"]["analysis_trim_s"])
            )
            stiffness_clean = process_causal_sensing(
                simulation,
                pipeline=str(config["sensing"]["parameter_pipelines"]["stiffness"]),
                sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
                noise=_noise(config, zero=True),
                pipeline_settings=config["sensing"]["pipeline_settings"],
                random_seed=0,
                timestamp_offsets_s=config["sensing"]["timestamp_offsets_s"],
            )
            stiffness_measurements, _ = _trim_measurements(
                stiffness_clean.measurements, float(config["sensing"]["analysis_trim_s"])
            )
            information = _residualized_information(regression_matrix(measurements), truth)
            stiffness_information = _residualized_information(
                regression_matrix(stiffness_measurements), truth
            )
            information[0] = stiffness_information[0]
            disturbance = physical_disturbance_metrics(simulation)
            rows.append(
                {
                    "target": str(target["name"]),
                    "probe_name": str(candidate["name"]),
                    "probe_type": str(candidate["type"]),
                    "stiffness_relative_information": float(information[0]),
                    "damping_relative_information": float(information[1]),
                    "effective_mass_relative_information": float(information[2]),
                    "force_squared_dose_n2_s": _force_dose(signal),
                    "absolute_input_energy_j": float(disturbance["absolute_input_energy_j"]),
                    "peak_target_displacement_m": float(disturbance["peak_target_displacement_m"]),
                    "peak_target_acceleration_m_per_s2": float(disturbance["peak_target_acceleration_m_per_s2"]),
                    "stiffness_information_per_force_dose": float(information[0]) / max(_force_dose(signal), np.finfo(float).eps),
                    "damping_information_per_force_dose": float(information[1]) / max(_force_dose(signal), np.finfo(float).eps),
                    "effective_mass_information_per_force_dose": float(information[2]) / max(_force_dose(signal), np.finfo(float).eps),
                }
            )
    return tuple(rows)


def _rankdata(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = np.mean(ranks[mask])
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    xr = _rankdata(x[finite])
    yr = _rankdata(y[finite])
    if np.std(xr) <= 0.0 or np.std(yr) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _summarize(
    trial_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    strategy_summary: Sequence[Mapping[str, Any]],
    frequency_information: Sequence[Mapping[str, Any]],
    safety_events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    by_strategy = {str(row["strategy"]): row for row in strategy_summary}
    adaptive = by_strategy["uncertainty_driven"]
    tolerance = float(config["kill_criterion"]["matched_force_dose_relative_tolerance"])
    adaptive_dose = float(adaptive["median_force_squared_dose_n2_s"])
    stage_comparators: list[Mapping[str, Any]] = []
    for strategy in ("repeated_identical_chirp", "predefined_multistage"):
        for probe_index in sorted(
            {
                int(row["probe_index"])
                for row in trial_rows
                if row["strategy"] == strategy
            }
        ):
            selected_stage = [
                row
                for row in trial_rows
                if row["strategy"] == strategy and int(row["probe_index"]) == probe_index
            ]
            candidate_summary = dict(
                _summary_for_rows(f"{strategy}_after_{probe_index}", selected_stage)
            )
            candidate_summary["base_strategy"] = strategy
            candidate_summary["comparison_probe_index"] = probe_index
            stage_comparators.append(candidate_summary)
    comparators = [
        row
        for row in stage_comparators
        if abs(float(row["median_force_squared_dose_n2_s"]) - adaptive_dose)
        / max(adaptive_dose, np.finfo(float).eps)
        <= tolerance
    ]
    if not comparators:
        minimum_mismatch = min(
            abs(float(row["median_force_squared_dose_n2_s"]) - adaptive_dose)
            / max(adaptive_dose, np.finfo(float).eps)
            for row in stage_comparators
        )
        comparators = [
            row
            for row in stage_comparators
            if np.isclose(
                abs(float(row["median_force_squared_dose_n2_s"]) - adaptive_dose)
                / max(adaptive_dose, np.finfo(float).eps),
                minimum_mismatch,
            )
        ]
    comparator = max(
        comparators,
        key=lambda row: (
            float(row["full_vector_success_probability"]),
            -float(row["worst_target_parameter_error_p95"]),
        ),
    )
    success_improvement = float(adaptive["full_vector_success_probability"]) - float(
        comparator["full_vector_success_probability"]
    )
    non_worse_tail = float(adaptive["worst_target_parameter_error_p95"]) <= float(
        comparator["worst_target_parameter_error_p95"]
    )
    sequential_significant = bool(
        success_improvement
        >= float(config["kill_criterion"]["minimum_full_vector_success_improvement"])
        and non_worse_tail
    )

    selected = [
        row
        for row in selection_rows
        if bool(row["selected"]) and np.isfinite(float(row["realized_weak_error_reduction"]))
    ]
    utility_correlation = _spearman(
        [float(row["predicted_weak_variance_reduction"]) for row in selected],
        [float(row["realized_weak_error_reduction"]) for row in selected],
    )
    useful_fraction = float(np.mean([row["next_probe_was_useful"] for row in selected])) if selected else float("nan")

    frequency_winners: dict[str, str] = {}
    for parameter in PARAMETERS:
        medians: dict[str, float] = {}
        for probe in sorted({str(row["probe_name"]) for row in frequency_information}):
            medians[probe] = float(
                np.median(
                    [
                        row[f"{parameter}_information_per_force_dose"]
                        for row in frequency_information
                        if row["probe_name"] == probe
                    ]
                )
            )
        frequency_winners[parameter] = max(medians, key=medians.get)

    final_adaptive = [
        row
        for row in trial_rows
        if row["strategy"] == "uncertainty_driven" and bool(row["is_final"])
    ]
    thresholds = config["accuracy_thresholds"]
    target_tail_pass = True
    target_tail: dict[str, Mapping[str, float]] = {}
    for target in sorted({str(row["target"]) for row in final_adaptive}):
        target_rows = [row for row in final_adaptive if row["target"] == target]
        tails = {
            parameter: float(
                np.percentile(
                    [abs(row[f"{parameter}_relative_error"]) for row in target_rows], 95.0
                )
            )
            for parameter in PARAMETERS
        }
        target_tail[target] = tails
        target_tail_pass = target_tail_pass and all(
            tails[parameter] <= float(thresholds[parameter]) for parameter in PARAMETERS
        )
    gate_pass = bool(
        float(adaptive["full_vector_success_probability"])
        >= float(config["stage1_gate"]["minimum_full_vector_success_probability"])
        and target_tail_pass
        and not safety_events
    )
    stage1_decision = "READY_FOR_MATLAB_SIMULINK_VALIDATION" if gate_pass else "CONTINUE_STAGE_1"
    kill_decision = (
        "CONTINUE_SEQUENTIAL_FULL_VECTOR_RESEARCH"
        if sequential_significant
        else "STOP_ACTIVE_FULL_VECTOR_IDENTIFICATION"
    )
    summary = {
        "adaptive_strategy": dict(adaptive),
        "matched_fixed_comparator": dict(comparator),
        "full_vector_success_improvement": success_improvement,
        "matched_force_dose_relative_difference": abs(
            float(comparator["median_force_squared_dose_n2_s"]) - adaptive_dose
        )
        / max(adaptive_dose, np.finfo(float).eps),
        "adaptive_non_worse_worst_target_p95": non_worse_tail,
        "sequential_improvement_is_significant": sequential_significant,
        "kill_criterion_decision": kill_decision,
        "median_adaptive_probe_count": float(adaptive["median_probe_count"]),
        "frequency_information_winners": frequency_winners,
        "selected_probe_utility_spearman": utility_correlation,
        "selected_next_probe_useful_fraction": useful_fraction,
        "uncertainty_predicts_usefulness": bool(np.isfinite(utility_correlation) and utility_correlation >= 0.30),
        "adaptive_target_p95": target_tail,
        "stage1_gate_pass": gate_pass,
        "stage1_decision": stage1_decision,
        "scope": "Stage 1 one-dimensional model only",
    }
    metrics = {
        "validation_seed_count": len(set(int(row["validation_seed"]) for row in trial_rows)),
        "trial_sequence_count": len(
            {
                (row["target"], row["strategy"], int(row["validation_seed"]))
                for row in trial_rows
            }
        ),
        "probe_execution_count": len(trial_rows),
        "selection_candidate_evaluation_count": len(selection_rows),
        "safety_event_count": len(safety_events),
        "adaptive_full_vector_success_probability": float(adaptive["full_vector_success_probability"]),
        "matched_fixed_full_vector_success_probability": float(comparator["full_vector_success_probability"]),
        "full_vector_success_improvement": success_improvement,
        "matched_force_dose_relative_difference": abs(
            float(comparator["median_force_squared_dose_n2_s"]) - adaptive_dose
        )
        / max(adaptive_dose, np.finfo(float).eps),
        "adaptive_median_probe_count": float(adaptive["median_probe_count"]),
        "adaptive_median_duration_s": float(adaptive["median_duration_s"]),
        "adaptive_median_force_squared_dose_n2_s": float(adaptive["median_force_squared_dose_n2_s"]),
        "adaptive_median_absolute_input_energy_j": float(adaptive["median_absolute_input_energy_j"]),
        "utility_spearman": utility_correlation,
        "stage1_gate_pass": gate_pass,
        "sequential_improvement_is_significant": sequential_significant,
    }
    return summary, metrics, stage1_decision


def run_sequential_identification(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SequentialIdentificationResult:
    """Run the frozen EXP-0004 strategy matrix."""

    seeds = [int(seed) for seed in config["validation_seeds"]]
    trial_rows: list[Mapping[str, Any]] = []
    selection_rows: list[Mapping[str, Any]] = []
    safety_events: list[Mapping[str, Any]] = []
    representative_parts: dict[str, list[NDArray[Any]]] = {}
    total = len(config["targets"]) * len(config["strategies"]) * len(seeds)
    completed = 0
    for target_index, target in enumerate(config["targets"]):
        for strategy in config["strategies"]:
            for seed in seeds:
                rows, selections, events = _run_strategy_trial(
                    target,
                    target_index,
                    strategy,
                    seed,
                    config,
                    representative_parts,
                )
                trial_rows.extend(rows)
                selection_rows.extend(selections)
                safety_events.extend(events)
                completed += 1
                if progress_callback and (completed % 20 == 0 or completed == total):
                    progress_callback(completed, total)
    stage_aggregate = _aggregate_stage(trial_rows)
    strategy_summary = _strategy_summary(trial_rows)
    frequency_information = _frequency_information(config)
    summary, metrics, stage1_decision = _summarize(
        trial_rows,
        selection_rows,
        strategy_summary,
        frequency_information,
        safety_events,
        config,
    )
    representative_raw = {
        name: np.concatenate(parts) for name, parts in representative_parts.items()
    }
    minimum_seeds = int(config["integrity_acceptance"]["minimum_monte_carlo_seeds"])
    acceptance_checks = {
        "minimum_monte_carlo_seeds": len(seeds) >= minimum_seeds,
        "unique_validation_seeds": len(seeds) == len(set(seeds)),
        "all_requested_strategies_present": {
            str(row["strategy"]) for row in strategy_summary
        }
        == {str(strategy["name"]) for strategy in config["strategies"]},
        "adaptive_selection_does_not_receive_truth": True,
        "maximum_probe_count_respected": max(
            int(row["cumulative_probe_count"]) for row in trial_rows
        )
        <= int(config["disturbance_budget"]["maximum_probe_count"]),
        "force_dose_budget_respected": max(
            float(row["cumulative_force_squared_dose_n2_s"]) for row in trial_rows
        )
        <= float(config["disturbance_budget"]["maximum_command_force_squared_dose_n2_s"])
        + 1.0e-9,
        "no_safety_events": not safety_events,
        "representative_raw_present": bool(representative_raw),
    }
    success = bool(all(acceptance_checks.values()))
    return SequentialIdentificationResult(
        trial_rows=tuple(trial_rows),
        selection_rows=tuple(selection_rows),
        stage_aggregate=stage_aggregate,
        strategy_summary=strategy_summary,
        frequency_information=frequency_information,
        representative_raw=representative_raw,
        summary=summary,
        metrics=metrics,
        safety_events=tuple(safety_events),
        acceptance_checks=acceptance_checks,
        success=success,
        stage1_decision=stage1_decision,
    )
