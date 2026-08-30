"""EXP-0006 passive post-chirp ring-down safety observation."""

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
from probeing.measurements import SyntheticMeasurements, process_causal_sensing

from .decision_sufficiency import (
    CLASS_TO_INDEX,
    FREQUENCY_FEATURES,
    RISK_LABELS,
    TIME_FEATURES,
    PopulationSimulation,
    RidgeModel,
    TargetCase,
    _actual_severity,
    _case_simulation,
    _feature_matrix,
    _fit_ridge,
    _frequency_features,
    _inverse_outcome,
    _maneuver_signal,
    _measurement_seed,
    _noise,
    _ridge_predict,
    _signed_log,
    _time_features,
    _transform_outcome,
    evaluate_future_response,
    generate_target_cases,
    simulate_population,
)


CHIRP_FEATURES = (
    "estimated_stiffness_signed_log",
    *FREQUENCY_FEATURES,
    *TIME_FEATURES,
)
RINGDOWN_FEATURES = (
    "rd_observation_duration_s",
    "rd_valid",
    "rd_decay_rate_log",
    "rd_decay_fit_r2",
    "rd_decay_time_constant_log",
    "rd_dominant_frequency_hz",
    "rd_log_decrement_log",
    "rd_damping_ratio",
    "rd_peak_count_log",
    "rd_residual_peak_log",
    "rd_residual_rms_log",
    "rd_displacement_ratio_log",
    "rd_velocity_ratio_log",
    "rd_threshold_reached",
    "rd_time_to_threshold_fraction",
    "rd_threshold_dwell_fraction",
    "rd_zero_crossing_rate_log",
    "rd_energy_decay_rate_log",
    "rd_energy_ratio_log",
)
PHYSICAL_INCREMENT_FEATURES = (
    "estimated_damping_signed_log",
    "estimated_mass_signed_log",
)
FEATURE_SETS = {
    "chirp_only": CHIRP_FEATURES,
    "ringdown_only": RINGDOWN_FEATURES,
    "chirp_ringdown": (*CHIRP_FEATURES, *RINGDOWN_FEATURES),
    "physical_chirp_ringdown": (*PHYSICAL_INCREMENT_FEATURES, *CHIRP_FEATURES, *RINGDOWN_FEATURES),
}


@dataclass(frozen=True)
class OutcomeModels:
    regressors: Mapping[str, RidgeModel]
    upper_residuals: Mapping[str, float]


@dataclass(frozen=True)
class PassiveRingdownResult:
    validation_rows: tuple[Mapping[str, Any], ...]
    duration_summary: tuple[Mapping[str, Any], ...]
    feature_set_summary: tuple[Mapping[str, Any], ...]
    quantitative_summary: tuple[Mapping[str, Any], ...]
    early_stop_rows: tuple[Mapping[str, Any], ...]
    early_stop_summary: tuple[Mapping[str, Any], ...]
    legacy_audit_rows: tuple[Mapping[str, Any], ...]
    feature_importance: tuple[Mapping[str, Any], ...]
    class_distribution: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    legacy_raw: Mapping[str, NDArray[Any]]
    safety_events: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    acceptance_checks: Mapping[str, bool]
    success: bool
    stage1_decision: str


def _extended_probe_signal(
    config: Mapping[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probe = config["probe"]
    step = float(probe["sample_period_s"])
    probe_duration = float(probe["duration_s"])
    total_duration = probe_duration + float(config["passive_observation"]["maximum_duration_s"])
    time = np.arange(int(round(total_duration / step)) + 1, dtype=float) * step
    frequency_slope = (
        float(probe["end_frequency_hz"]) - float(probe["start_frequency_hz"])
    ) / probe_duration
    phase = 2.0 * np.pi * (
        float(probe["start_frequency_hz"]) * time
        + 0.5 * frequency_slope * time**2
    )
    force = np.zeros_like(time)
    active = time <= probe_duration
    force[active] = float(probe["amplitude_n"]) * np.sin(phase[active])
    force[np.abs(force) < 1.0e-15] = 0.0
    return time, force


def _slice_measurements(
    measurements: SyntheticMeasurements, mask: NDArray[np.bool_]
) -> SyntheticMeasurements:
    return SyntheticMeasurements(
        time_s=measurements.time_s[mask],
        displacement_m=measurements.displacement_m[mask],
        velocity_m_per_s=measurements.velocity_m_per_s[mask],
        acceleration_m_per_s2=measurements.acceleration_m_per_s2[mask],
        contact_force_n=measurements.contact_force_n[mask],
        noise=measurements.noise,
        random_seed=measurements.random_seed,
    )


def _chirp_features_and_diagnostics(
    primary: Any,
    stiffness_sensing: Any,
    case: TargetCase,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    duration = float(config["probe"]["duration_s"])
    trim = float(config["sensing"]["chirp_analysis_trim_s"])
    time = primary.measurements.time_s
    mask = (time >= trim) & (time <= duration - trim)
    measurements = _slice_measurements(primary.measurements, mask)
    stiffness_measurements = _slice_measurements(stiffness_sensing.measurements, mask)
    command = np.maximum(np.asarray(primary.commanded_force_n[mask], dtype=float), 0.0)
    ols = ordinary_least_squares_eiv(measurements)
    instruments = delayed_input_instruments(
        command,
        measurements.time_s,
        (0.0, 0.025, 0.050, 0.075, 0.100),
    )
    iv = instrumental_variables(measurements, instruments)
    tls = total_least_squares(stiffness_measurements)
    stiffness = float(tls.parameters[0]) if tls.valid else float(ols.parameters[0])
    damping = float(ols.parameters[1])
    mass = float(iv.parameters[2]) if iv.valid else float(ols.parameters[2])
    truth = np.asarray(
        [case.stiffness_n_per_m, case.damping_n_s_per_m, case.effective_mass_kg],
        dtype=float,
    )
    estimate = np.asarray([stiffness, damping, mass], dtype=float)
    return {
        "estimated_stiffness_n_per_m": stiffness,
        "estimated_damping_n_s_per_m": damping,
        "estimated_effective_mass_kg": mass,
        "estimated_stiffness_signed_log": _signed_log(stiffness, 100.0),
        "estimated_damping_signed_log": _signed_log(damping, 5.0),
        "estimated_mass_signed_log": _signed_log(mass, 1.0),
        "stiffness_abs_relative_error": float(abs((estimate[0] - truth[0]) / truth[0])),
        "damping_abs_relative_error": float(abs((estimate[1] - truth[1]) / truth[1])),
        "effective_mass_abs_relative_error": float(abs((estimate[2] - truth[2]) / truth[2])),
        **_frequency_features(measurements),
        **_time_features(measurements, command),
    }


def _block_rms(
    time: NDArray[np.float64], values: NDArray[np.float64], block_duration_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if time.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    relative = time - time[0]
    block_count = max(int(np.ceil((relative[-1] + np.finfo(float).eps) / block_duration_s)), 1)
    centers: list[float] = []
    rms: list[float] = []
    for index in range(block_count):
        low = index * block_duration_s
        high = (index + 1) * block_duration_s
        mask = (relative >= low) & (relative < high if index < block_count - 1 else relative <= high)
        if np.count_nonzero(mask) < 2:
            continue
        centers.append(float(np.mean(relative[mask])))
        rms.append(float(np.sqrt(np.mean(values[mask] ** 2))))
    return np.asarray(centers, dtype=float), np.asarray(rms, dtype=float)


def _decay_fit(
    centers: NDArray[np.float64],
    envelope: NDArray[np.float64],
    floor: float,
    minimum_points: int,
) -> tuple[float, float]:
    valid = envelope > max(floor, np.finfo(float).tiny)
    if np.count_nonzero(valid) < minimum_points:
        return 0.0, 0.0
    x = centers[valid]
    y = np.log(envelope[valid])
    design = np.column_stack((np.ones(x.size), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = max(0.0, 1.0 - residual / total) if total > 1.0e-15 else 0.0
    return float(max(-coefficients[1], 0.0)), float(min(r2, 1.0))


def _positive_peaks(values: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    if values.size < 3:
        return np.asarray([], dtype=float)
    indices = np.flatnonzero(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] > threshold)
    ) + 1
    return values[indices]


def ringdown_features(
    measurements: SyntheticMeasurements,
    probe_end_s: float,
    observation_duration_s: float,
    config: Mapping[str, Any],
) -> Mapping[str, float]:
    """Extract prefix-causal passive features available at a decision time."""

    if observation_duration_s <= 0.0:
        return {name: 0.0 for name in RINGDOWN_FEATURES}
    settings = config["ringdown_features"]
    end = probe_end_s + observation_duration_s + 1.0e-12
    mask = (measurements.time_s >= probe_end_s) & (measurements.time_s <= end)
    time = np.asarray(measurements.time_s[mask], dtype=float)
    displacement = np.asarray(measurements.displacement_m[mask], dtype=float)
    velocity = np.asarray(measurements.velocity_m_per_s[mask], dtype=float)
    if time.size < 3:
        return {
            **{name: 0.0 for name in RINGDOWN_FEATURES},
            "rd_observation_duration_s": float(observation_duration_s),
        }
    relative = time - probe_end_s
    block = float(settings["envelope_block_duration_s"])
    centers, envelope = _block_rms(time, displacement, block)
    noise_floor = float(measurements.noise.displacement_std_m)
    decay_rate, decay_r2 = _decay_fit(
        centers,
        envelope,
        1.5 * noise_floor,
        int(settings["minimum_envelope_points_for_decay_fit"]),
    )
    centered = displacement - float(np.mean(displacement))
    spectrum = np.abs(np.fft.rfft(centered))
    frequency = np.fft.rfftfreq(centered.size, float(time[1] - time[0]))
    dominant = (
        float(frequency[1 + int(np.argmax(spectrum[1:]))])
        if spectrum.size > 1 and float(np.max(spectrum[1:])) > np.finfo(float).tiny
        else 0.0
    )
    prominence = float(settings["minimum_peak_prominence_noise_multiplier"]) * noise_floor
    peaks = _positive_peaks(centered, prominence)
    decrements = np.log(peaks[:-1] / peaks[1:]) if peaks.size >= 2 else np.asarray([])
    decrements = decrements[np.isfinite(decrements) & (decrements > 0.0)]
    decrement = float(np.median(decrements)) if decrements.size else 0.0
    damping_ratio = (
        float(decrement / np.sqrt((2.0 * np.pi) ** 2 + decrement**2))
        if decrement > 0.0
        else 0.0
    )
    trailing = max(
        int(round(float(settings["trailing_amplitude_window_s"]) / (time[1] - time[0]))),
        2,
    )
    tail_x = displacement[-trailing:]
    tail_v = velocity[-trailing:]
    residual_peak = float(np.max(np.abs(tail_x)))
    residual_rms = float(np.sqrt(np.mean(tail_x**2)))
    initial_count = min(trailing, displacement.size)
    initial_x = float(np.sqrt(np.mean(displacement[:initial_count] ** 2)))
    initial_v = float(np.sqrt(np.mean(velocity[:initial_count] ** 2)))
    tail_v_rms = float(np.sqrt(np.mean(tail_v**2)))
    below = (
        np.abs(displacement) <= float(settings["displacement_threshold_m"])
    ) & (np.abs(velocity) <= float(settings["velocity_threshold_m_per_s"]))
    dwell_samples = 0
    for value in below[::-1]:
        if not bool(value):
            break
        dwell_samples += 1
    threshold_reached = dwell_samples > 0
    threshold_dwell = dwell_samples * float(time[1] - time[0])
    time_to_threshold = max(observation_duration_s - threshold_dwell, 0.0)
    signs = np.signbit(centered)
    zero_crossings = int(np.count_nonzero(signs[:-1] != signs[1:]))
    energy_proxy = displacement**2 + (velocity / (2.0 * np.pi)) ** 2
    energy_centers, energy_envelope = _block_rms(time, np.sqrt(energy_proxy), block)
    energy_amplitude_decay, _ = _decay_fit(
        energy_centers,
        energy_envelope,
        max(noise_floor, 1.0e-12),
        int(settings["minimum_envelope_points_for_decay_fit"]),
    )
    first_energy = float(np.mean(energy_proxy[:initial_count]))
    last_energy = float(np.mean(energy_proxy[-initial_count:]))
    return {
        "rd_observation_duration_s": float(observation_duration_s),
        "rd_valid": 1.0,
        "rd_decay_rate_log": float(np.log1p(decay_rate)),
        "rd_decay_fit_r2": decay_r2,
        "rd_decay_time_constant_log": float(np.log1p(1.0 / max(decay_rate, 1.0e-6))),
        "rd_dominant_frequency_hz": dominant,
        "rd_log_decrement_log": float(np.log1p(decrement)),
        "rd_damping_ratio": damping_ratio,
        "rd_peak_count_log": float(np.log1p(peaks.size)),
        "rd_residual_peak_log": float(np.log(max(residual_peak, 1.0e-12))),
        "rd_residual_rms_log": float(np.log(max(residual_rms, 1.0e-12))),
        "rd_displacement_ratio_log": float(
            np.log(max(residual_rms, 1.0e-12) / max(initial_x, 1.0e-12))
        ),
        "rd_velocity_ratio_log": float(
            np.log(max(tail_v_rms, 1.0e-12) / max(initial_v, 1.0e-12))
        ),
        "rd_threshold_reached": float(threshold_reached),
        "rd_time_to_threshold_fraction": float(
            time_to_threshold / max(observation_duration_s, 1.0e-12)
        ),
        "rd_threshold_dwell_fraction": float(
            threshold_dwell / max(observation_duration_s, 1.0e-12)
        ),
        "rd_zero_crossing_rate_log": float(
            np.log1p(zero_crossings / max(observation_duration_s, 1.0e-12))
        ),
        "rd_energy_decay_rate_log": float(np.log1p(2.0 * energy_amplitude_decay)),
        "rd_energy_ratio_log": float(
            np.log(max(last_energy, 1.0e-18) / max(first_energy, 1.0e-18))
        ),
    }


def _extract_partition_features(
    cases: Sequence[TargetCase],
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None,
    progress_state: list[int],
    progress_total: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], PopulationSimulation]:
    time, command = _extended_probe_signal(config)
    population = simulate_population(
        cases, time, command, contact_mode=str(config["probe"]["contact_mode"])
    )
    feature_rows: list[Mapping[str, Any]] = []
    safety_events: list[Mapping[str, Any]] = []
    limits = config["probe_safety_limits"]
    probe_mask = time <= float(config["probe"]["duration_s"])
    windows = [float(value) for value in config["passive_observation"]["windows_s"]]
    for index, case in enumerate(cases):
        peak_values = {
            "peak_force_n": float(np.max(np.abs(population.contact_force_n[probe_mask]))),
            "peak_target_displacement_m": float(np.max(np.abs(population.displacement_m[index]))),
            "peak_target_velocity_m_per_s": float(np.max(np.abs(population.velocity_m_per_s[index]))),
            "peak_target_acceleration_m_per_s2": float(np.max(np.abs(population.acceleration_m_per_s2[index]))),
        }
        for metric, limit_name in (
            ("peak_force_n", "max_peak_force_n"),
            ("peak_target_displacement_m", "max_peak_target_displacement_m"),
            ("peak_target_velocity_m_per_s", "max_peak_target_velocity_m_per_s"),
            ("peak_target_acceleration_m_per_s2", "max_peak_target_acceleration_m_per_s2"),
        ):
            if peak_values[metric] > float(limits[limit_name]):
                safety_events.append(
                    {
                        "target_id": case.target_id,
                        "partition": case.partition,
                        "metric": metric,
                        "value": peak_values[metric],
                        "limit": float(limits[limit_name]),
                    }
                )
        truth = _case_simulation(
            population, case, index, str(config["probe"]["contact_mode"])
        )
        for regime_index, regime in enumerate(config["sensing"]["noise_regimes"]):
            common = dict(
                sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
                noise=_noise(config, float(regime["multiplier"])),
                pipeline_settings=config["sensing"]["pipeline_settings"],
                random_seed=_measurement_seed(case, regime_index),
                timestamp_offsets_s={
                    "displacement": 0.0,
                    "velocity": 0.0,
                    "acceleration": 0.0,
                    "force": 0.0,
                },
            )
            primary = process_causal_sensing(
                truth, pipeline=str(config["sensing"]["primary_pipeline"]), **common
            )
            stiffness_sensing = process_causal_sensing(
                truth, pipeline="alpha_beta_gamma", **common
            )
            chirp_values = _chirp_features_and_diagnostics(
                primary, stiffness_sensing, case, config
            )
            for window in windows:
                ring_values = ringdown_features(
                    primary.measurements,
                    float(config["probe"]["duration_s"]),
                    window,
                    config,
                )
                feature_rows.append(
                    {
                        "target_id": case.target_id,
                        "partition": case.partition,
                        "source_seed": case.seed,
                        "case_index": case.case_index,
                        "noise_regime": str(regime["name"]),
                        "noise_multiplier": float(regime["multiplier"]),
                        "observation_duration_s": window,
                        "true_stiffness_n_per_m": case.stiffness_n_per_m,
                        "true_damping_n_s_per_m": case.damping_n_s_per_m,
                        "true_effective_mass_kg": case.effective_mass_kg,
                        "probe_force_squared_dose_n2_s": float(
                            np.trapz(population.contact_force_n[probe_mask] ** 2, time[probe_mask])
                        ),
                        **chirp_values,
                        **ring_values,
                    }
                )
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


def _outcome_map(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    return {str(row["target_id"]): row for row in rows}


def _fit_models(
    training_features: Sequence[Mapping[str, Any]],
    training_outcomes: Sequence[Mapping[str, Any]],
    calibration_features: Sequence[Mapping[str, Any]],
    calibration_outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[tuple[float, str], OutcomeModels]:
    truth = _outcome_map(training_outcomes)
    calibration_truth = _outcome_map(calibration_outcomes)
    outcomes = tuple(str(name) for name in config["predictor"]["predicted_outcomes"])
    penalty = float(config["predictor"]["ridge_penalty"])
    quantile = float(config["predictor"]["upper_residual_quantile"])
    models: dict[tuple[float, str], OutcomeModels] = {}
    for duration in [float(value) for value in config["passive_observation"]["windows_s"]]:
        train_duration = [
            row for row in training_features if float(row["observation_duration_s"]) == duration
        ]
        calibrate_duration = [
            row for row in calibration_features if float(row["observation_duration_s"]) == duration
        ]
        for feature_set, names in FEATURE_SETS.items():
            train_matrix = _feature_matrix(train_duration, names)
            calibration_matrix = _feature_matrix(calibrate_duration, names)
            regressors: dict[str, RidgeModel] = {}
            residuals: dict[str, float] = {}
            for outcome in outcomes:
                train_values = np.asarray(
                    [
                        _transform_outcome(
                            outcome, float(truth[str(row["target_id"])][outcome])
                        )
                        for row in train_duration
                    ],
                    dtype=float,
                )
                model = _fit_ridge(train_matrix, train_values, penalty)
                regressors[outcome] = model
                actual = np.asarray(
                    [
                        _transform_outcome(
                            outcome,
                            float(calibration_truth[str(row["target_id"])][outcome]),
                        )
                        for row in calibrate_duration
                    ],
                    dtype=float,
                )
                residuals[outcome] = float(
                    np.quantile(actual - _ridge_predict(model, calibration_matrix), quantile)
                )
            models[(duration, feature_set)] = OutcomeModels(regressors, residuals)
    return models


def _predict(
    model: OutcomeModels,
    feature: Mapping[str, Any],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    matrix = _feature_matrix([feature], feature_names)
    median: dict[str, float] = {}
    upper: dict[str, float] = {}
    for outcome, regressor in model.regressors.items():
        transformed = float(_ridge_predict(regressor, matrix)[0])
        median[outcome] = _inverse_outcome(outcome, transformed)
        upper[outcome] = _inverse_outcome(
            outcome, transformed + float(model.upper_residuals[outcome])
        )
    envelope = config["risk_envelope"]
    outcomes = (
        "peak_displacement_m",
        "peak_velocity_m_per_s",
        "late_hold_oscillation_rms_m",
        "hold_settling_time_s",
    )
    safe_ratios = np.asarray(
        [upper[name] / float(envelope["safe"][name]) for name in outcomes], dtype=float
    )
    unsafe_ratios = np.asarray(
        [upper[name] / float(envelope["unsafe"][name]) for name in outcomes], dtype=float
    )
    contact_loss = upper["peak_displacement_m"] > float(
        envelope["contact_tracking_displacement_m"]
    )
    safe = bool(np.max(safe_ratios) <= 1.0 and not contact_loss)
    unsafe = bool(np.max(unsafe_ratios) > 1.0 or contact_loss)
    predicted = "SAFE" if safe else ("UNSAFE" if unsafe else "CAUTION")
    veto = config["predictor"]["passive_persistence_veto"]
    decay_rate = float(np.expm1(float(feature["rd_decay_rate_log"])))
    residual_rms = float(np.exp(float(feature["rd_residual_rms_log"])))
    displacement_noise = (
        float(config["sensing"]["base_noise"]["displacement_std_m"])
        * float(feature["noise_multiplier"])
    )
    persistence_veto = bool(
        veto["enabled"]
        and "rd_decay_rate_log" in feature_names
        and float(feature["observation_duration_s"])
        >= float(veto["minimum_observation_s"])
        and decay_rate <= float(veto["maximum_safe_decay_rate_per_s"])
        and residual_rms
        >= float(veto["minimum_residual_displacement_snr"]) * displacement_noise
        and float(feature["rd_threshold_dwell_fraction"])
        <= float(veto["maximum_threshold_dwell_fraction"])
    )
    if predicted == "SAFE" and persistence_veto:
        predicted = "CAUTION"
    if predicted == "SAFE":
        margin = 1.0 - float(np.max(safe_ratios))
    elif predicted == "UNSAFE":
        margin = float(np.max(unsafe_ratios)) - 1.0
    else:
        margin = min(
            max(float(np.max(safe_ratios)) - 1.0, 0.0),
            max(1.0 - float(np.max(unsafe_ratios)), 0.0),
        )
        if persistence_veto:
            margin = max(
                margin,
                (
                    float(veto["maximum_safe_decay_rate_per_s"]) - decay_rate
                )
                / max(float(veto["maximum_safe_decay_rate_per_s"]), 1.0e-12),
            )
    return {
        "predicted_risk_class": predicted,
        "predicted_risk_score": float(np.max(safe_ratios)),
        "decision_margin": float(max(margin, 0.0)),
        "predicted_contact_loss_proxy": contact_loss,
        "passive_persistence_veto_active": persistence_veto,
        "predicted_outcomes": median,
        "upper_outcomes": upper,
    }


def _prediction_row(
    feature: Mapping[str, Any],
    feature_set: str,
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    row: dict[str, Any] = {
        "target_id": str(feature["target_id"]),
        "source_seed": int(feature["source_seed"]),
        "case_index": int(feature["case_index"]),
        "noise_regime": str(feature["noise_regime"]),
        "noise_multiplier": float(feature["noise_multiplier"]),
        "observation_duration_s": float(feature["observation_duration_s"]),
        "feature_set": feature_set,
        "predicted_risk_class": str(prediction["predicted_risk_class"]),
        "predicted_risk_score": float(prediction["predicted_risk_score"]),
        "decision_margin": float(prediction["decision_margin"]),
        "predicted_contact_loss_proxy": bool(prediction["predicted_contact_loss_proxy"]),
        "passive_persistence_veto_active": bool(
            prediction["passive_persistence_veto_active"]
        ),
        "probe_force_squared_dose_n2_s": float(feature["probe_force_squared_dose_n2_s"]),
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
    for name in RINGDOWN_FEATURES:
        row[name] = float(feature[name])
    for outcome in config["predictor"]["predicted_outcomes"]:
        row[f"predicted_{outcome}"] = float(prediction["predicted_outcomes"][outcome])
        row[f"upper_{outcome}"] = float(prediction["upper_outcomes"][outcome])
    return row


def _attach_actual(
    pending: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    truth = _outcome_map(outcomes)
    output: list[Mapping[str, Any]] = []
    for pending_row in pending:
        actual = truth[str(pending_row["target_id"])]
        row = dict(pending_row)
        row.update(
            {
                "actual_risk_class": str(actual["risk_class"]),
                "actual_risk_score": _actual_severity(actual, config),
                "classification_correct": bool(
                    pending_row["predicted_risk_class"] == actual["risk_class"]
                ),
                "false_safe": bool(
                    pending_row["predicted_risk_class"] == "SAFE"
                    and actual["risk_class"] != "SAFE"
                ),
                "false_unsafe": bool(
                    pending_row["predicted_risk_class"] == "UNSAFE"
                    and actual["risk_class"] == "SAFE"
                ),
                "actual_contact_loss_proxy": bool(actual["contact_loss_proxy"]),
            }
        )
        for outcome in config["predictor"]["predicted_outcomes"]:
            row[f"actual_{outcome}"] = float(actual[outcome])
        output.append(row)
    return tuple(output)


def _classification_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    actual = np.asarray([CLASS_TO_INDEX[str(row["actual_risk_class"])] for row in rows])
    predicted = np.asarray([CLASS_TO_INDEX[str(row["predicted_risk_class"])] for row in rows])
    confusion = np.zeros((3, 3), dtype=int)
    for truth, estimate in zip(actual, predicted):
        confusion[truth, estimate] += 1
    non_safe = actual != CLASS_TO_INDEX["SAFE"]
    safe = ~non_safe
    false_safe = (predicted == CLASS_TO_INDEX["SAFE"]) & non_safe
    false_unsafe = (predicted == CLASS_TO_INDEX["UNSAFE"]) & safe
    output: dict[str, Any] = {
        "trial_count": len(rows),
        "accuracy": float(np.mean(actual == predicted)),
        "false_safe_rate": float(np.sum(false_safe) / max(np.sum(non_safe), 1)),
        "false_safe_fraction": float(np.mean(false_safe)),
        "false_safe_count": int(np.sum(false_safe)),
        "false_unsafe_rate": float(np.sum(false_unsafe) / max(np.sum(safe), 1)),
    }
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
    return output


def _duration_summaries(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    durations = [float(value) for value in config["passive_observation"]["windows_s"]]
    noises = [str(row["name"]) for row in config["sensing"]["noise_regimes"]]
    primary = str(config["predictor"]["primary_feature_set"])
    duration_summary: list[Mapping[str, Any]] = []
    feature_summary: list[Mapping[str, Any]] = []
    baseline_class = {
        (str(row["target_id"]), str(row["noise_regime"])): str(row["predicted_risk_class"])
        for row in rows
        if row["feature_set"] == primary and float(row["observation_duration_s"]) == 0.0
    }
    for duration in durations:
        for noise in noises:
            selected = [
                row
                for row in rows
                if row["feature_set"] == primary
                and float(row["observation_duration_s"]) == duration
                and row["noise_regime"] == noise
            ]
            settling_error = np.asarray(
                [
                    abs(float(row["predicted_hold_settling_time_s"]) - float(row["actual_hold_settling_time_s"]))
                    for row in selected
                ],
                dtype=float,
            )
            displacement_relative = np.asarray(
                [
                    abs(
                        (float(row["predicted_peak_displacement_m"]) - float(row["actual_peak_displacement_m"]))
                        / max(float(row["actual_peak_displacement_m"]), 1.0e-12)
                    )
                    for row in selected
                ],
                dtype=float,
            )
            duration_summary.append(
                {
                    "observation_duration_s": duration,
                    "noise_regime": noise,
                    **_classification_metrics(selected),
                    "settling_median_absolute_error_s": float(np.median(settling_error)),
                    "settling_p95_absolute_error_s": float(np.quantile(settling_error, 0.95)),
                    "peak_displacement_median_abs_relative_error": float(np.median(displacement_relative)),
                    "peak_displacement_p95_abs_relative_error": float(np.quantile(displacement_relative, 0.95)),
                    "median_decision_margin": float(np.median([row["decision_margin"] for row in selected])),
                    "decision_change_probability": float(
                        np.mean(
                            [
                                str(row["predicted_risk_class"])
                                != baseline_class[(str(row["target_id"]), noise)]
                                for row in selected
                            ]
                        )
                    ),
                    "median_probe_force_squared_dose_n2_s": float(
                        np.median([row["probe_force_squared_dose_n2_s"] for row in selected])
                    ),
                }
            )
            for feature_set in FEATURE_SETS:
                subset = [
                    row
                    for row in rows
                    if row["feature_set"] == feature_set
                    and float(row["observation_duration_s"]) == duration
                    and row["noise_regime"] == noise
                ]
                feature_summary.append(
                    {
                        "observation_duration_s": duration,
                        "noise_regime": noise,
                        "feature_set": feature_set,
                        **_classification_metrics(subset),
                    }
                )
    return tuple(duration_summary), tuple(feature_summary)


def _quantitative_summary(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for duration in [float(value) for value in config["passive_observation"]["windows_s"]]:
        for noise in [str(row["name"]) for row in config["sensing"]["noise_regimes"]]:
            for feature_set in FEATURE_SETS:
                subset = [
                    row
                    for row in rows
                    if row["feature_set"] == feature_set
                    and row["noise_regime"] == noise
                    and float(row["observation_duration_s"]) == duration
                ]
                for outcome in config["predictor"]["predicted_outcomes"]:
                    actual = np.asarray([row[f"actual_{outcome}"] for row in subset], dtype=float)
                    estimate = np.asarray([row[f"predicted_{outcome}"] for row in subset], dtype=float)
                    upper = np.asarray([row[f"upper_{outcome}"] for row in subset], dtype=float)
                    absolute = np.abs(estimate - actual)
                    relative = absolute / np.maximum(np.abs(actual), 1.0e-9)
                    output.append(
                        {
                            "observation_duration_s": duration,
                            "noise_regime": noise,
                            "feature_set": feature_set,
                            "outcome": outcome,
                            "trial_count": len(subset),
                            "median_absolute_error": float(np.median(absolute)),
                            "p95_absolute_error": float(np.quantile(absolute, 0.95)),
                            "median_abs_relative_error": float(np.median(relative)),
                            "p95_abs_relative_error": float(np.quantile(relative, 0.95)),
                            "upper_bound_coverage": float(np.mean(actual <= upper)),
                        }
                    )
    return tuple(output)


def _early_stop(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    settings = config["early_stopping"]
    primary = str(config["predictor"]["primary_feature_set"])
    candidates = [row for row in rows if row["feature_set"] == primary]
    output: list[Mapping[str, Any]] = []
    keys = sorted({(str(row["target_id"]), str(row["noise_regime"])) for row in candidates})
    for target_id, noise in keys:
        sequence = sorted(
            [
                row
                for row in candidates
                if row["target_id"] == target_id and row["noise_regime"] == noise
            ],
            key=lambda row: float(row["observation_duration_s"]),
        )
        chosen = sequence[-1]
        reason = "maximum_observation"
        required_count = int(settings["required_consecutive_identical_decisions"])
        for index, current in enumerate(sequence):
            duration = float(current["observation_duration_s"])
            label = str(current["predicted_risk_class"])
            minimum = float(settings[f"minimum_{label.lower()}_observation_s"])
            if duration < minimum or index + 1 < required_count:
                continue
            recent = sequence[index - required_count + 1 : index + 1]
            stable = all(row["predicted_risk_class"] == label for row in recent)
            margin = float(current["decision_margin"]) >= float(settings["minimum_decision_margin"])
            previous = recent[-2]
            old_bound = float(previous["upper_hold_settling_time_s"])
            new_bound = float(current["upper_hold_settling_time_s"])
            negligible = abs(new_bound - old_bound) / max(abs(old_bound), 0.1) <= float(
                settings["maximum_relative_settling_bound_change"]
            )
            decay_certain = float(current["rd_decay_fit_r2"]) >= float(
                settings["minimum_decay_fit_r2"]
            )
            evidence = margin and (
                label == "UNSAFE" or decay_certain or negligible
            )
            if stable and evidence:
                chosen = current
                reason = (
                    "stable_margin_decay"
                    if decay_certain
                    else ("stable_unsafe_margin" if label == "UNSAFE" else "stable_negligible_change")
                )
                break
        output.append(
            {
                **dict(chosen),
                "early_stop_reason": reason,
                "early_stop_duration_s": float(chosen["observation_duration_s"]),
            }
        )
    summary: list[Mapping[str, Any]] = []
    for noise in [str(row["name"]) for row in config["sensing"]["noise_regimes"]]:
        selected = [row for row in output if row["noise_regime"] == noise]
        if not selected:
            continue
        durations = np.asarray([row["early_stop_duration_s"] for row in selected], dtype=float)
        summary.append(
            {
                "noise_regime": noise,
                **_classification_metrics(selected),
                "median_observation_duration_s": float(np.median(durations)),
                "p95_observation_duration_s": float(np.quantile(durations, 0.95)),
                "maximum_observation_duration_s": float(np.max(durations)),
                "early_stop_fraction": float(
                    np.mean(durations < float(settings["maximum_observation_s"]))
                ),
            }
        )
    return tuple(output), tuple(summary)


def _legacy_cases(config: Mapping[str, Any]) -> tuple[TargetCase, ...]:
    ids = [str(value) for value in config["legacy_exp_0005_false_safe_audit"]["target_ids"]]
    seeds = sorted({int(target_id.split("_s", 1)[1].split("_", 1)[0]) for target_id in ids})
    cases = generate_target_cases(
        seeds,
        int(config["seed_partitions"]["cases_per_seed"]),
        config["target_population"],
        partition="validation",
    )
    by_id = {case.target_id: case for case in cases}
    return tuple(by_id[target_id] for target_id in ids)


def _predict_features(
    features: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[float, str], OutcomeModels],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    pending: list[Mapping[str, Any]] = []
    for feature in features:
        duration = float(feature["observation_duration_s"])
        for feature_set, names in FEATURE_SETS.items():
            prediction = _predict(models[(duration, feature_set)], feature, names, config)
            pending.append(_prediction_row(feature, feature_set, prediction, config))
    return tuple(pending)


def _feature_importance(
    models: Mapping[tuple[float, str], OutcomeModels],
    reporting_duration: float,
) -> tuple[Mapping[str, Any], ...]:
    model = models[(reporting_duration, "chirp_ringdown")].regressors[
        "hold_settling_time_s"
    ]
    coefficient = np.asarray(model.coefficients[1:], dtype=float)
    rows = [
        {
            "feature": name,
            "feature_group": "ringdown" if name in RINGDOWN_FEATURES else "chirp",
            "standardized_settling_coefficient": float(value),
            "absolute_standardized_settling_coefficient": float(abs(value)),
        }
        for name, value in zip(FEATURE_SETS["chirp_ringdown"], coefficient)
    ]
    rows.sort(key=lambda row: row["absolute_standardized_settling_coefficient"], reverse=True)
    return tuple(rows)


def _class_distribution(
    outcomes: Sequence[Mapping[str, Any]], partition: str
) -> tuple[Mapping[str, Any], ...]:
    total = len(outcomes)
    return tuple(
        {
            "partition": partition,
            "risk_class": label,
            "case_count": int(sum(row["risk_class"] == label for row in outcomes)),
            "fraction": float(sum(row["risk_class"] == label for row in outcomes) / total),
        }
        for label in RISK_LABELS
    )


def _raw_signals(
    cases: Sequence[TargetCase],
    outcomes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    select_all: bool,
) -> Mapping[str, NDArray[Any]]:
    truth_map = _outcome_map(outcomes)
    if select_all:
        selected = list(cases)
    else:
        selected = [
            next(case for case in cases if truth_map[case.target_id]["risk_class"] == label)
            for label in RISK_LABELS
        ]
    time, command = _extended_probe_signal(config)
    population = simulate_population(selected, time, command, contact_mode="unilateral")
    regime_index = next(
        index
        for index, row in enumerate(config["sensing"]["noise_regimes"])
        if row["name"] == config["representative_raw"]["noise_regime"]
    )
    regime = config["sensing"]["noise_regimes"][regime_index]
    parts: dict[str, list[NDArray[Any]]] = {}
    for index, case in enumerate(selected):
        truth = _case_simulation(population, case, index, "unilateral")
        sensing = process_causal_sensing(
            truth,
            pipeline=str(config["sensing"]["primary_pipeline"]),
            sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
            noise=_noise(config, float(regime["multiplier"])),
            pipeline_settings=config["sensing"]["pipeline_settings"],
            random_seed=_measurement_seed(case, regime_index),
            timestamp_offsets_s={
                "displacement": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "force": 0.0,
            },
        )
        count = sensing.measurements.time_s.size
        values = {
            "target_id": np.full(count, case.target_id),
            "risk_class": np.full(count, truth_map[case.target_id]["risk_class"]),
            "time_s": sensing.measurements.time_s,
            "true_displacement_m": sensing.true_displacement_m,
            "measured_displacement_m": sensing.measurements.displacement_m,
            "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
            "estimated_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
            "true_force_n": sensing.true_contact_force_n,
            "measured_force_n": sensing.measurements.contact_force_n,
            "commanded_force_n": sensing.commanded_force_n,
        }
        for field, values_array in values.items():
            parts.setdefault(field, []).append(np.asarray(values_array))
    return {field: np.concatenate(values) for field, values in parts.items()}


def _summary(
    duration_summary: Sequence[Mapping[str, Any]],
    feature_summary: Sequence[Mapping[str, Any]],
    early_summary: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    validation_safety: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    criteria = config["stage1_safety_criterion"]
    practical_limit = float(criteria["maximum_fixed_observation_s"])
    durations = sorted(
        {
            float(row["observation_duration_s"])
            for row in duration_summary
            if float(row["observation_duration_s"]) <= practical_limit
        }
    )

    def duration_row(duration: float, noise: str) -> Mapping[str, Any]:
        return next(
            row
            for row in duration_summary
            if float(row["observation_duration_s"]) == duration
            and row["noise_regime"] == noise
        )

    nominal_legacy_ids = set(
        str(value)
        for value in config["legacy_exp_0005_false_safe_audit"][
            "nominal_false_safe_target_ids"
        ]
    )
    qualifying: list[float] = []
    duration_checks: list[Mapping[str, Any]] = []
    baseline_dose = float(duration_row(0.0, "nominal")["median_probe_force_squared_dose_n2_s"])
    for duration in durations:
        nominal = duration_row(duration, "nominal")
        high = duration_row(duration, "high")
        legacy = [
            row
            for row in legacy_rows
            if row["noise_regime"] == "nominal"
            and row["target_id"] in nominal_legacy_ids
            and float(row["observation_duration_s"]) == duration
            and row["feature_set"] == config["predictor"]["primary_feature_set"]
        ]
        legacy_detected = bool(legacy) and all(
            row["predicted_risk_class"] != "SAFE" for row in legacy
        )
        dose_change = abs(
            float(nominal["median_probe_force_squared_dose_n2_s"]) - baseline_dose
        ) / max(abs(baseline_dose), 1.0e-12)
        checks = {
            "observation_duration_s": duration,
            "nominal_false_safe": float(nominal["false_safe_rate"])
            <= float(criteria["maximum_nominal_false_safe_rate"]),
            "high_false_safe": float(high["false_safe_rate"])
            <= float(criteria["maximum_high_noise_false_safe_rate"]),
            "nominal_accuracy": float(nominal["accuracy"])
            >= float(criteria["minimum_nominal_accuracy"]),
            "high_accuracy": float(high["accuracy"])
            >= float(criteria["minimum_high_noise_accuracy"]),
            "unsafe_recall": min(float(nominal["unsafe_recall"]), float(high["unsafe_recall"]))
            >= float(criteria["minimum_unsafe_recall"]),
            "no_added_probe_energy": dose_change
            <= float(criteria["maximum_relative_probe_energy_increase"]),
            "legacy_nominal_false_safe_targets_detected": legacy_detected,
        }
        duration_checks.append(checks)
        if all(value for key, value in checks.items() if key != "observation_duration_s"):
            qualifying.append(duration)
    selected_duration = min(qualifying) if qualifying else min(
        durations,
        key=lambda duration: (
            float(duration_row(duration, "high")["false_safe_rate"]),
            float(duration_row(duration, "nominal")["false_safe_rate"]),
            duration,
        ),
    )
    baseline_nominal = duration_row(0.0, "nominal")
    baseline_high = duration_row(0.0, "high")
    selected_nominal = duration_row(selected_duration, "nominal")
    selected_high = duration_row(selected_duration, "high")
    selected_feature_comparison = tuple(
        dict(row)
        for row in feature_summary
        if float(row["observation_duration_s"]) == selected_duration
        and row["noise_regime"] == "nominal"
    )
    relative_reduction = {
        "nominal": float(
            (float(baseline_nominal["false_safe_rate"]) - float(selected_nominal["false_safe_rate"]))
            / max(float(baseline_nominal["false_safe_rate"]), 1.0e-12)
        ),
        "high": float(
            (float(baseline_high["false_safe_rate"]) - float(selected_high["false_safe_rate"]))
            / max(float(baseline_high["false_safe_rate"]), 1.0e-12)
        ),
    }
    stage_pass = bool(qualifying and not validation_safety)
    stage_decision = (
        "READY_FOR_INDEPENDENT_MATLAB_SIMULINK_VALIDATION"
        if stage_pass
        else "CONTINUE_STAGE_1"
    )
    summary = {
        "qualifying_fixed_durations_s": qualifying,
        "selected_reporting_duration_s": selected_duration,
        "duration_gate_checks": duration_checks,
        "baseline_chirp_only_primary": {
            "nominal": dict(baseline_nominal),
            "high": dict(baseline_high),
        },
        "selected_duration_primary": {
            "nominal": dict(selected_nominal),
            "high": dict(selected_high),
        },
        "selected_duration_feature_comparison_nominal": selected_feature_comparison,
        "false_safe_relative_reduction": relative_reduction,
        "early_stopping": {row["noise_regime"]: dict(row) for row in early_summary},
        "passive_ringdown_safety_criterion_pass": stage_pass,
        "stage1_decision": stage_decision,
        "validation_predictions_created_before_future_maneuver": True,
        "legacy_audit_was_evaluation_only": True,
    }
    metrics = {
        "selected_reporting_duration_s": selected_duration,
        "qualifying_duration_count": len(qualifying),
        "baseline_nominal_false_safe_rate": float(baseline_nominal["false_safe_rate"]),
        "selected_nominal_false_safe_rate": float(selected_nominal["false_safe_rate"]),
        "baseline_high_false_safe_rate": float(baseline_high["false_safe_rate"]),
        "selected_high_false_safe_rate": float(selected_high["false_safe_rate"]),
        "selected_nominal_accuracy": float(selected_nominal["accuracy"]),
        "selected_high_accuracy": float(selected_high["accuracy"]),
        "probe_safety_event_count": len(validation_safety),
        "stage1_gate_pass": stage_pass,
    }
    return summary, metrics, stage_decision


def run_passive_ringdown(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PassiveRingdownResult:
    """Fit, freeze, and evaluate EXP-0006 passive observation."""

    partitions = config["seed_partitions"]
    count = int(partitions["cases_per_seed"])
    training_cases = generate_target_cases(
        partitions["training"], count, config["target_population"], partition="training"
    )
    calibration_cases = generate_target_cases(
        partitions["calibration"], count, config["target_population"], partition="calibration"
    )
    validation_cases = generate_target_cases(
        partitions["validation"], count, config["target_population"], partition="validation"
    )
    regime_count = len(config["sensing"]["noise_regimes"])
    progress_total = regime_count * (
        len(training_cases) + len(calibration_cases) + len(validation_cases)
    )
    state = [0]
    training_features, training_safety, _ = _extract_partition_features(
        training_cases,
        config,
        progress_callback=progress_callback,
        progress_state=state,
        progress_total=progress_total,
    )
    training_outcomes = _simulate_outcomes(training_cases, config)
    calibration_features, calibration_safety, _ = _extract_partition_features(
        calibration_cases,
        config,
        progress_callback=progress_callback,
        progress_state=state,
        progress_total=progress_total,
    )
    calibration_outcomes = _simulate_outcomes(calibration_cases, config)
    models = _fit_models(
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
        progress_state=state,
        progress_total=progress_total,
    )
    # Critical separation: every validation prediction is frozen before the
    # hidden sustained-contact maneuver is simulated.
    pending_validation = _predict_features(validation_features, models, config)
    validation_outcomes = _simulate_outcomes(validation_cases, config)
    validation_rows = _attach_actual(pending_validation, validation_outcomes, config)
    duration_summary, feature_summary = _duration_summaries(validation_rows, config)
    quantitative = _quantitative_summary(validation_rows, config)
    early_rows, early_summary = _early_stop(validation_rows, config)

    # Locked EXP-0005 failures are reproduced only after EXP-0006 fitting and
    # validation predictions are complete; they never select or tune a model.
    legacy_cases = _legacy_cases(config)
    legacy_state = [0]
    legacy_features, legacy_safety, _ = _extract_partition_features(
        legacy_cases,
        config,
        progress_callback=None,
        progress_state=legacy_state,
        progress_total=regime_count * len(legacy_cases),
    )
    legacy_pending = _predict_features(legacy_features, models, config)
    legacy_outcomes = _simulate_outcomes(legacy_cases, config)
    legacy_rows = _attach_actual(legacy_pending, legacy_outcomes, config)

    summary, metrics, stage_decision = _summary(
        duration_summary,
        feature_summary,
        early_summary,
        legacy_rows,
        validation_safety,
        config,
    )
    importance = _feature_importance(
        models, float(summary["selected_reporting_duration_s"])
    )
    top_ringdown = next(
        row for row in importance if row["feature_group"] == "ringdown"
    )
    summary = {**summary, "top_ringdown_settling_feature": dict(top_ringdown)}
    class_distribution = _class_distribution(validation_outcomes, "validation")
    representative = _raw_signals(
        validation_cases, validation_outcomes, config, select_all=False
    )
    legacy_raw = _raw_signals(legacy_cases, legacy_outcomes, config, select_all=True)
    all_safety = tuple((*training_safety, *calibration_safety, *validation_safety, *legacy_safety))
    minimum_fraction = float(config["integrity_acceptance"]["minimum_fraction_per_risk_class"])
    validation_seeds = [int(seed) for seed in partitions["validation"]]
    doses = {
        float(row["observation_duration_s"]): float(row["median_probe_force_squared_dose_n2_s"])
        for row in duration_summary
        if row["noise_regime"] == "nominal"
    }
    acceptance = {
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
        "primary_pipeline_is_causal": config["sensing"]["primary_pipeline"]
        == "causal_low_pass",
        "no_noncausal_primary_result": bool(config["sensing"]["no_noncausal_primary_result"]),
        "passive_force_is_zero": float(config["passive_observation"]["force_n"]) == 0.0,
        "no_additional_probe_energy": max(doses.values()) - min(doses.values()) <= 1.0e-12,
        "no_validation_probe_safety_events": not validation_safety,
        "legacy_audit_evaluation_only": bool(
            config["legacy_exp_0005_false_safe_audit"]["evaluation_only"]
        ),
        "representative_all_classes_present": len(
            set(str(value) for value in representative["risk_class"])
        )
        == 3,
    }
    metrics = {
        **metrics,
        "validation_case_count": len(validation_cases),
        "validation_prediction_count": len(validation_rows),
        "all_partition_safety_event_count": len(all_safety),
    }
    return PassiveRingdownResult(
        validation_rows=validation_rows,
        duration_summary=duration_summary,
        feature_set_summary=feature_summary,
        quantitative_summary=quantitative,
        early_stop_rows=early_rows,
        early_stop_summary=early_summary,
        legacy_audit_rows=legacy_rows,
        feature_importance=importance,
        class_distribution=class_distribution,
        representative_raw=representative,
        legacy_raw=legacy_raw,
        safety_events=all_safety,
        summary=summary,
        metrics=metrics,
        acceptance_checks=acceptance,
        success=bool(all(acceptance.values())),
        stage1_decision=stage_decision,
    )
