"""EXP-0002: realistic sensing and practical-identifiability stress test."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from probeing.estimators import (
    batch_least_squares,
    dynamic_ratio_least_squares,
    recursive_least_squares,
    regression_diagnostics,
)
from probeing.measurements import (
    MeasurementNoise,
    SyntheticMeasurements,
    process_practical_sensing,
)
from probeing.metrics import (
    aggregate_trials,
    normalized_information_metrics,
    physical_disturbance_metrics,
    probe_spectrum_metrics,
    rms,
    true_modal_parameters,
)
from probeing.models import (
    ContactInteractionSimulation,
    InteractionParameters,
    MassSpringDamperModel,
    simulate_contact_interaction,
)
from probeing.probing import make_probe


@dataclass(frozen=True)
class PracticalIdentifiabilityResult:
    trial_metrics: tuple[Mapping[str, Any], ...]
    aggregate_metrics: tuple[Mapping[str, Any], ...]
    go_candidates: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    ramp_failure_analysis: Mapping[str, Any]
    metrics: Mapping[str, Any]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool
    stage1_decision: str
    best_candidate: Mapping[str, Any]


def _truth_parameters(target: Mapping[str, Any]) -> InteractionParameters:
    return InteractionParameters(
        stiffness_n_per_m=float(target["stiffness_n_per_m"]),
        damping_n_s_per_m=float(target["damping_n_s_per_m"]),
        effective_mass_kg=float(target["effective_mass_kg"]),
    )


def _parameter_vector(parameters: InteractionParameters) -> NDArray[np.float64]:
    return np.asarray(
        [
            parameters.stiffness_n_per_m,
            parameters.damping_n_s_per_m,
            parameters.effective_mass_kg,
        ],
        dtype=float,
    )


def _truth_measurements_from_sensing(sensing: Any) -> SyntheticMeasurements:
    measurements = sensing.measurements
    return SyntheticMeasurements(
        time_s=measurements.time_s,
        displacement_m=sensing.true_displacement_m,
        velocity_m_per_s=sensing.true_velocity_m_per_s,
        acceleration_m_per_s2=sensing.true_acceleration_m_per_s2,
        contact_force_n=measurements.contact_force_n,
        noise=MeasurementNoise(),
        random_seed=measurements.random_seed,
    )


def _rls_convergence(
    time_s: NDArray[np.float64],
    history: NDArray[np.float64],
    truth: NDArray[np.float64],
    tolerance: float,
) -> tuple[bool, float]:
    relative = np.max(np.abs((history - truth[None, :]) / np.abs(truth)[None, :]), axis=1)
    tail_worst = np.maximum.accumulate(relative[::-1])[::-1]
    matches = np.flatnonzero(tail_worst <= tolerance)
    if matches.size == 0:
        return False, -1.0
    return True, float(time_s[int(matches[0])])


def _safe_relative(value: float, truth: float) -> float:
    return float((value - truth) / abs(truth))


def _append_raw(
    parts: dict[str, list[NDArray[Any]]], values: Mapping[str, NDArray[Any]]
) -> None:
    for name, array in values.items():
        parts.setdefault(name, []).append(np.asarray(array))


def _representative_selected(
    target: str,
    probe: str,
    seed: int,
    config: Mapping[str, Any],
) -> bool:
    if seed != int(config["validation_seeds"][0]):
        return False
    pairs = {
        (str(entry["target"]), str(entry["probe"]))
        for entry in config["representative_raw"]["target_probe_pairs"]
    }
    return (target, probe) in pairs


def _case_id(
    *,
    contact_mode: str,
    target: str,
    probe: str,
    regime: str,
    pipeline: str,
    severity: str,
    seed: int,
) -> str:
    return "__".join(
        (contact_mode, target, probe, regime, pipeline, severity, f"s{seed}")
    )


def _go_candidates(
    trials: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    criteria = config["go_no_go"]
    nominal = str(criteria["severity"])
    contact_mode = str(criteria["contact_mode"])
    allowed_regimes = set(criteria["practical_regimes"])
    filtered = [
        row
        for row in trials
        if row["severity"] == nominal
        and row["contact_mode"] == contact_mode
        and row["sensing_regime"] in allowed_regimes
        and not bool(row["force_is_inferred"])
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in filtered:
        key = (str(row["sensing_regime"]), str(row["pipeline"]), str(row["probe"]))
        grouped.setdefault(key, []).append(row)

    candidates: list[Mapping[str, Any]] = []
    target_names = [str(target["name"]) for target in config["targets"]]
    for (regime, pipeline, probe), rows in grouped.items():
        output: dict[str, Any] = {
            "sensing_regime": regime,
            "pipeline": pipeline,
            "probe": probe,
            "severity": nominal,
            "contact_mode": contact_mode,
            "trial_count": len(rows),
        }
        for label in ("stiffness", "damping", "effective_mass"):
            target_p95 = []
            target_bias = []
            for target_name in target_names:
                target_rows = [row for row in rows if row["target"] == target_name]
                errors = np.asarray(
                    [row[f"{label}_relative_error"] for row in target_rows], dtype=float
                )
                target_p95.append(float(np.percentile(np.abs(errors), 95.0)))
                target_bias.append(abs(float(np.mean(errors))))
            output[f"worst_target_{label}_p95_abs_relative_error"] = max(target_p95)
            output[f"worst_target_{label}_abs_relative_bias"] = max(target_bias)

        for label in ("natural_frequency", "damping_ratio"):
            target_p95 = []
            for target_name in target_names:
                values = [
                    row[f"{label}_relative_error"]
                    for row in rows
                    if row["target"] == target_name and bool(row["ratio_estimate_valid"])
                ]
                target_p95.append(
                    float(np.percentile(np.abs(values), 95.0)) if values else float("inf")
                )
            output[f"worst_target_{label}_p95_abs_relative_error"] = max(target_p95)

        output["full_rank_fraction"] = float(
            np.mean([int(row["rank"]) == 3 for row in rows])
        )
        output["ratio_estimate_valid_fraction"] = float(
            np.mean([bool(row["ratio_estimate_valid"]) for row in rows])
        )
        output["maximum_peak_force_n"] = float(max(row["peak_force_n"] for row in rows))
        output["maximum_peak_target_displacement_m"] = float(
            max(row["peak_target_displacement_m"] for row in rows)
        )
        parameter_tail = np.asarray(
            [
                output["worst_target_stiffness_p95_abs_relative_error"],
                output["worst_target_damping_p95_abs_relative_error"],
                output["worst_target_effective_mass_p95_abs_relative_error"],
            ]
        )
        output["candidate_score"] = rms(parameter_tail)
        output["passes_parameter_tail"] = bool(
            output["worst_target_stiffness_p95_abs_relative_error"]
            <= float(criteria["max_stiffness_p95_abs_relative_error"])
            and output["worst_target_damping_p95_abs_relative_error"]
            <= float(criteria["max_damping_p95_abs_relative_error"])
            and output["worst_target_effective_mass_p95_abs_relative_error"]
            <= float(criteria["max_effective_mass_p95_abs_relative_error"])
        )
        output["passes_bias"] = bool(
            output["worst_target_stiffness_abs_relative_bias"]
            <= float(criteria["max_stiffness_abs_relative_bias"])
            and output["worst_target_damping_abs_relative_bias"]
            <= float(criteria["max_damping_abs_relative_bias"])
            and output["worst_target_effective_mass_abs_relative_bias"]
            <= float(criteria["max_effective_mass_abs_relative_bias"])
        )
        output["passes_observability"] = bool(
            output["full_rank_fraction"] >= float(criteria["min_full_rank_fraction"])
        )
        output["passes_disturbance"] = bool(
            output["maximum_peak_force_n"] <= float(criteria["max_peak_force_n"])
            and output["maximum_peak_target_displacement_m"]
            <= float(criteria["max_peak_target_displacement_m"])
        )
        output["stage1_go_candidate"] = bool(
            output["passes_parameter_tail"]
            and output["passes_bias"]
            and output["passes_observability"]
            and output["passes_disturbance"]
        )
        candidates.append(output)
    return tuple(sorted(candidates, key=lambda row: float(row["candidate_score"])))


def analyze_exp0001_ramp_failure(
    repository_root: Path, reference: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Quantify the recorded EXP-0001 ramp failure without changing its artifacts."""

    raw_path = repository_root / str(reference["raw_npz"])
    case_path = repository_root / str(reference["case_metrics_csv"])
    with case_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    archive = np.load(raw_path)
    near_rows = [row for row in rows if row["scenario"] == "near_ideal"]
    ramp_rows = [row for row in near_rows if row["probe"] == "ramp"]
    worst = max(ramp_rows, key=lambda row: float(row["batch_relative_error_rms"]))
    target = str(worst["target"])
    comparisons: dict[str, Any] = {}
    for probe in ("ramp", "chirp", "multisine"):
        mask = (
            (archive["scenario"] == "near_ideal")
            & (archive["target"] == target)
            & (archive["probe"] == probe)
        )
        time = archive["time_s"][mask].astype(float)
        displacement = archive["measured_displacement_m"][mask].astype(float)
        velocity = archive["measured_velocity_m_per_s"][mask].astype(float)
        acceleration = archive["measured_acceleration_m_per_s2"][mask].astype(float)
        force = archive["true_contact_force_n"][mask].astype(float)
        design = np.column_stack((displacement, velocity, acceleration))
        diagnostics = regression_diagnostics(design)
        row = next(
            item for item in near_rows if item["target"] == target and item["probe"] == probe
        )
        stiffness = float(row["true_stiffness_n_per_m"])
        damping = float(row["true_damping_n_s_per_m"])
        mass = float(row["true_effective_mass_kg"])
        natural_hz = np.sqrt(stiffness / mass) / (2.0 * np.pi)
        spectrum = probe_spectrum_metrics(
            time, force, natural_frequency_hz=float(natural_hz)
        )
        force_rms = max(rms(force), np.finfo(float).eps)
        comparisons[probe] = {
            "parameter_relative_error_rms": float(row["batch_relative_error_rms"]),
            "stiffness_relative_error": float(row["batch_stiffness_relative_error"]),
            "damping_relative_error": float(row["batch_damping_relative_error"]),
            "effective_mass_relative_error": float(
                row["batch_effective_mass_relative_error"]
            ),
            "displacement_feature_rms_m": rms(displacement),
            "velocity_feature_rms_m_per_s": rms(velocity),
            "acceleration_feature_rms_m_per_s2": rms(acceleration),
            "elastic_contribution_rms_fraction": rms(stiffness * displacement)
            / force_rms,
            "damping_contribution_rms_fraction": rms(damping * velocity) / force_rms,
            "inertial_contribution_rms_fraction": rms(mass * acceleration) / force_rms,
            "normalized_singular_values": diagnostics.normalized_singular_values.tolist(),
            "normalized_condition_number": diagnostics.normalized_condition_number,
            "maximum_abs_parameter_correlation": diagnostics.maximum_abs_parameter_correlation,
            **spectrum,
        }
    return {
        "reference_run_id": str(reference["run_id"]),
        "near_ideal_ramp_mean_relative_error_rms": float(
            np.mean([float(row["batch_relative_error_rms"]) for row in ramp_rows])
        ),
        "worst_case_id": str(worst["case_id"]),
        "worst_case_effective_mass_relative_error": float(
            worst["batch_effective_mass_relative_error"]
        ),
        "target_compared": target,
        "probe_diagnostics": comparisons,
        "interpretation": (
            "The ramp is dominated by a static/DC hold, so stiffness explains most of the "
            "force after the short transition. Its acceleration column is weak relative to "
            "the elastic contribution, leaving effective mass sensitive to small regressor "
            "noise. Broadband chirp and multisine inputs sustain acceleration excitation and "
            "reduce this errors-in-variables sensitivity."
        ),
    }


def run_practical_identifiability(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PracticalIdentifiabilityResult:
    """Run the complete EXP-0002 validation-seed Monte Carlo matrix."""

    if config["experiment"]["id"] != "EXP-0002":
        raise ValueError("this runner only implements EXP-0002")
    seeds = [int(seed) for seed in config["validation_seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("validation seeds must be unique")
    simulation = config["simulation"]
    sample_period = float(simulation["sample_period_s"])
    duration = float(simulation["duration_s"])
    target_configs = config["targets"]
    probe_configs = config["probes"]
    contact_modes = [str(mode) for mode in config["contact_modes"]]
    severities = config["imperfection_severities"]
    regimes = config["sensing_regimes"]
    pipeline_settings = config["pipeline_settings"]
    pipeline_count = sum(len(regime["pipelines"]) for regime in regimes)
    expected_trials = (
        len(target_configs)
        * len(probe_configs)
        * len(contact_modes)
        * len(severities)
        * pipeline_count
        * len(seeds)
    )

    truth_cache: dict[tuple[str, str, str], tuple[ContactInteractionSimulation, InteractionParameters]] = {}
    disturbance_cache: dict[tuple[str, str, str], Mapping[str, float | int]] = {}
    spectrum_cache: dict[tuple[str, str, str], Mapping[str, float]] = {}
    safety_events: list[Mapping[str, Any]] = []
    safety_seen: set[tuple[str, str, str, str]] = set()
    for target_config in target_configs:
        target_name = str(target_config["name"])
        parameters = _truth_parameters(target_config)
        model = MassSpringDamperModel(parameters)
        natural_frequency_hz = (
            np.sqrt(parameters.stiffness_n_per_m / parameters.effective_mass_kg)
            / (2.0 * np.pi)
        )
        for probe_config in probe_configs:
            probe = make_probe(dict(probe_config), sample_period, duration)
            for contact_mode in contact_modes:
                key = (target_name, probe.name, contact_mode)
                contact_truth = simulate_contact_interaction(
                    model,
                    probe.time_s,
                    probe.force_n,
                    contact_mode=contact_mode,
                )
                truth_cache[key] = (contact_truth, parameters)
                disturbance = physical_disturbance_metrics(contact_truth)
                disturbance_cache[key] = disturbance
                spectrum_cache[key] = probe_spectrum_metrics(
                    contact_truth.response.time_s,
                    contact_truth.contact_force_n,
                    natural_frequency_hz=float(natural_frequency_hz),
                )
                limits = config["safety_limits"]
                comparisons = (
                    ("peak_force_n", "max_peak_force_n", "N"),
                    (
                        "peak_target_displacement_m",
                        "max_peak_target_displacement_m",
                        "m",
                    ),
                    (
                        "peak_target_velocity_m_per_s",
                        "max_peak_target_velocity_m_per_s",
                        "m/s",
                    ),
                    (
                        "peak_target_acceleration_m_per_s2",
                        "max_peak_target_acceleration_m_per_s2",
                        "m/s^2",
                    ),
                )
                for metric, limit_name, units in comparisons:
                    if float(disturbance[metric]) > float(limits[limit_name]):
                        event_key = (target_name, probe.name, contact_mode, metric)
                        if event_key not in safety_seen:
                            safety_seen.add(event_key)
                            safety_events.append(
                                {
                                    "type": "limit_exceeded",
                                    "target": target_name,
                                    "probe": probe.name,
                                    "contact_mode": contact_mode,
                                    "metric": metric,
                                    "observed": float(disturbance[metric]),
                                    "limit": float(limits[limit_name]),
                                    "units": units,
                                }
                            )

    trial_rows: list[Mapping[str, Any]] = []
    representative_parts: dict[str, list[NDArray[Any]]] = {}
    completed = 0
    convergence_tolerance = float(config["estimators"]["rls_convergence_tolerance"])
    information_config = config["information_metrics"]

    for (target_name, probe_name, contact_mode), (contact_truth, parameters) in truth_cache.items():
        truth_vector = _parameter_vector(parameters)
        true_natural_frequency, true_damping_ratio = true_modal_parameters(parameters)
        disturbance = disturbance_cache[(target_name, probe_name, contact_mode)]
        spectrum = spectrum_cache[(target_name, probe_name, contact_mode)]
        for severity in severities:
            severity_name = str(severity["name"])
            for regime_config in regimes:
                regime = str(regime_config["name"])
                for pipeline in regime_config["pipelines"]:
                    pipeline_name = str(pipeline)
                    for seed in seeds:
                        sensing = process_practical_sensing(
                            contact_truth,
                            regime=regime,
                            pipeline=pipeline_name,
                            imperfection=severity,
                            pipeline_settings=pipeline_settings,
                            random_seed=seed,
                        )
                        batch = batch_least_squares(sensing.measurements)
                        oracle = batch_least_squares(
                            _truth_measurements_from_sensing(sensing)
                        )
                        recursive = recursive_least_squares(
                            sensing.measurements,
                            forgetting_factor=float(
                                config["estimators"]["forgetting_factor"]
                            ),
                            initial_covariance=float(
                                config["estimators"]["initial_covariance"]
                            ),
                            initial_parameters=config["estimators"]["initial_parameters"],
                        )
                        ratio = dynamic_ratio_least_squares(sensing.measurements)
                        relative_error = (batch.parameters - truth_vector) / np.abs(
                            truth_vector
                        )
                        error_rms = rms(relative_error)
                        eiv_shift = (batch.parameters - oracle.parameters) / np.abs(
                            truth_vector
                        )
                        rls_converged, convergence_time = _rls_convergence(
                            sensing.measurements.time_s,
                            recursive.parameter_history,
                            truth_vector,
                            convergence_tolerance,
                        )
                        diagnostics = batch.diagnostics
                        info = normalized_information_metrics(
                            diagnostics,
                            peak_displacement_m=float(
                                disturbance["peak_target_displacement_m"]
                            ),
                            absolute_input_energy_j=float(
                                disturbance["absolute_input_energy_j"]
                            ),
                            relative_estimation_error_rms=error_rms,
                            displacement_reference_m=float(
                                information_config["displacement_reference_m"]
                            ),
                            energy_reference_j=float(
                                information_config["energy_reference_j"]
                            ),
                        )
                        ratio_valid = bool(ratio.valid_physical_parameters)
                        estimated_natural = (
                            ratio.natural_frequency_rad_per_s if ratio_valid else -1.0
                        )
                        estimated_zeta = ratio.damping_ratio if ratio_valid else -1.0
                        case_id = _case_id(
                            contact_mode=contact_mode,
                            target=target_name,
                            probe=probe_name,
                            regime=regime,
                            pipeline=pipeline_name,
                            severity=severity_name,
                            seed=seed,
                        )
                        correlation = diagnostics.parameter_correlation
                        row: dict[str, Any] = {
                            "case_id": case_id,
                            "validation_seed": seed,
                            "contact_mode": contact_mode,
                            "target": target_name,
                            "probe": probe_name,
                            "sensing_regime": regime,
                            "pipeline": pipeline_name,
                            "severity": severity_name,
                            "severity_index": int(severity["severity_index"]),
                            "sample_rate_hz": float(severity["sample_rate_hz"]),
                            "position_sample_rate_hz": sensing.position_sample_rate_hz,
                            "displacement_std_m": float(severity["displacement_std_m"]),
                            "acceleration_std_m_per_s2": float(
                                severity["acceleration_std_m_per_s2"]
                            ),
                            "force_std_n": float(severity["force_std_n"]),
                            "position_latency_s": float(severity["position_latency_s"]),
                            "acceleration_latency_s": float(
                                severity["acceleration_latency_s"]
                            ),
                            "force_latency_s": float(severity["force_latency_s"]),
                            "timestamp_mismatch_std_s": float(
                                severity["timestamp_mismatch_std_s"]
                            ),
                            "filter_delay_s": sensing.filter_delay_s,
                            "force_is_inferred": sensing.force_is_inferred,
                            "sample_count": sensing.measurements.time_s.size,
                            "true_stiffness_n_per_m": truth_vector[0],
                            "true_damping_n_s_per_m": truth_vector[1],
                            "true_effective_mass_kg": truth_vector[2],
                            "estimated_stiffness_n_per_m": batch.parameters[0],
                            "estimated_damping_n_s_per_m": batch.parameters[1],
                            "estimated_effective_mass_kg": batch.parameters[2],
                            "stiffness_error": batch.parameters[0] - truth_vector[0],
                            "damping_error": batch.parameters[1] - truth_vector[1],
                            "effective_mass_error": batch.parameters[2] - truth_vector[2],
                            "stiffness_relative_error": relative_error[0],
                            "damping_relative_error": relative_error[1],
                            "effective_mass_relative_error": relative_error[2],
                            "parameter_relative_error_rms": error_rms,
                            "stiffness_eiv_relative_shift": eiv_shift[0],
                            "damping_eiv_relative_shift": eiv_shift[1],
                            "effective_mass_eiv_relative_shift": eiv_shift[2],
                            "oracle_stiffness_n_per_m": oracle.parameters[0],
                            "oracle_damping_n_s_per_m": oracle.parameters[1],
                            "oracle_effective_mass_kg": oracle.parameters[2],
                            "force_fit_rmse_n": batch.force_rmse_n,
                            "rank": diagnostics.rank,
                            "normalized_singular_value_1": diagnostics.normalized_singular_values[
                                0
                            ],
                            "normalized_singular_value_2": diagnostics.normalized_singular_values[
                                1
                            ],
                            "normalized_singular_value_3": diagnostics.normalized_singular_values[
                                2
                            ],
                            "normalized_condition_number": diagnostics.normalized_condition_number,
                            "parameter_correlation_k_c": correlation[0, 1],
                            "parameter_correlation_k_m": correlation[0, 2],
                            "parameter_correlation_c_m": correlation[1, 2],
                            "maximum_abs_parameter_correlation": diagnostics.maximum_abs_parameter_correlation,
                            "rls_final_stiffness_n_per_m": recursive.parameters[0],
                            "rls_final_damping_n_s_per_m": recursive.parameters[1],
                            "rls_final_effective_mass_kg": recursive.parameters[2],
                            "rls_converged": rls_converged,
                            "rls_convergence_time_s": convergence_time,
                            "ratio_estimate_valid": ratio_valid,
                            "estimated_natural_frequency_rad_per_s": estimated_natural,
                            "estimated_damping_ratio": estimated_zeta,
                            "natural_frequency_relative_error": (
                                _safe_relative(estimated_natural, true_natural_frequency)
                                if ratio_valid
                                else -1.0
                            ),
                            "damping_ratio_relative_error": (
                                _safe_relative(estimated_zeta, true_damping_ratio)
                                if ratio_valid
                                else -1.0
                            ),
                            **disturbance,
                            **spectrum,
                            **info,
                            "position_timestamp_offset_s": sensing.timestamp_offsets_s[
                                "position"
                            ],
                            "acceleration_timestamp_offset_s": sensing.timestamp_offsets_s[
                                "acceleration"
                            ],
                            "force_timestamp_offset_s": sensing.timestamp_offsets_s["force"],
                        }
                        trial_rows.append(row)

                        if _representative_selected(
                            target_name, probe_name, seed, config
                        ):
                            count = sensing.measurements.time_s.size
                            _append_raw(
                                representative_parts,
                                {
                                    "case_id": np.full(count, case_id),
                                    "contact_mode": np.full(count, contact_mode),
                                    "target": np.full(count, target_name),
                                    "probe": np.full(count, probe_name),
                                    "sensing_regime": np.full(count, regime),
                                    "pipeline": np.full(count, pipeline_name),
                                    "severity": np.full(count, severity_name),
                                    "validation_seed": np.full(count, seed, dtype=np.int64),
                                    "time_s": sensing.measurements.time_s,
                                    "commanded_force_n": sensing.commanded_force_n,
                                    "true_contact_force_n": sensing.true_contact_force_n,
                                    "raw_force_n": sensing.raw_force_n,
                                    "processed_force_n": sensing.measurements.contact_force_n,
                                    "true_displacement_m": sensing.true_displacement_m,
                                    "raw_displacement_m": sensing.raw_displacement_m,
                                    "processed_displacement_m": sensing.measurements.displacement_m,
                                    "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
                                    "processed_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
                                    "true_acceleration_m_per_s2": sensing.true_acceleration_m_per_s2,
                                    "raw_acceleration_m_per_s2": sensing.raw_acceleration_m_per_s2,
                                    "processed_acceleration_m_per_s2": sensing.measurements.acceleration_m_per_s2,
                                },
                            )
                        completed += 1
                        if progress_callback is not None and (
                            completed % 1000 == 0 or completed == expected_trials
                        ):
                            progress_callback(completed, expected_trials)

    aggregate = aggregate_trials(
        trial_rows,
        group_keys=(
            "contact_mode",
            "target",
            "probe",
            "sensing_regime",
            "pipeline",
            "severity",
        ),
    )
    candidates = _go_candidates(trial_rows, config)
    if not candidates:
        raise ValueError("GO/NO-GO configuration produced no practical candidates")
    best_candidate = candidates[0]
    stage1_go = any(bool(row["stage1_go_candidate"]) for row in candidates)
    stage1_decision = "GO" if stage1_go else "CONTINUE_STAGE_1"
    representative_raw = {
        name: np.concatenate(parts) for name, parts in representative_parts.items()
    }
    ramp_analysis = analyze_exp0001_ramp_failure(
        repository_root, config["exp_0001_reference"]
    )

    sensorless = [
        row
        for row in trial_rows
        if row["sensing_regime"] == "sensorless_force_exploratory"
    ]
    nominal_practical = [
        row
        for row in trial_rows
        if row["severity"] == config["go_no_go"]["severity"]
        and row["sensing_regime"] in config["go_no_go"]["practical_regimes"]
    ]
    summary: dict[str, Any] = {
        "trial_count": len(trial_rows),
        "expected_trial_count": expected_trials,
        "aggregate_group_count": len(aggregate),
        "validation_seed_count": len(seeds),
        "minimum_validation_seed": min(seeds),
        "maximum_validation_seed": max(seeds),
        "target_count": len(target_configs),
        "probe_count": len(probe_configs),
        "contact_mode_count": len(contact_modes),
        "severity_count": len(severities),
        "sensing_pipeline_combination_count": pipeline_count,
        "stage1_go": stage1_go,
        "stage1_decision": stage1_decision,
        "go_candidate_count": len(candidates),
        "passing_go_candidate_count": sum(
            bool(row["stage1_go_candidate"]) for row in candidates
        ),
        "best_practical_probe": best_candidate["probe"],
        "best_practical_sensing_regime": best_candidate["sensing_regime"],
        "best_practical_pipeline": best_candidate["pipeline"],
        "best_candidate_score": best_candidate["candidate_score"],
        "best_candidate_stiffness_p95": best_candidate[
            "worst_target_stiffness_p95_abs_relative_error"
        ],
        "best_candidate_damping_p95": best_candidate[
            "worst_target_damping_p95_abs_relative_error"
        ],
        "best_candidate_effective_mass_p95": best_candidate[
            "worst_target_effective_mass_p95_abs_relative_error"
        ],
        "nominal_practical_parameter_error_median": float(
            np.median([row["parameter_relative_error_rms"] for row in nominal_practical])
        ),
        "nominal_practical_parameter_error_p95": float(
            np.percentile(
                [row["parameter_relative_error_rms"] for row in nominal_practical], 95.0
            )
        ),
        "sensorless_parameter_error_median": float(
            np.median([row["parameter_relative_error_rms"] for row in sensorless])
        ),
        "sensorless_parameter_error_p95": float(
            np.percentile([row["parameter_relative_error_rms"] for row in sensorless], 95.0)
        ),
        "maximum_peak_force_n": float(max(row["peak_force_n"] for row in trial_rows)),
        "maximum_peak_target_displacement_m": float(
            max(row["peak_target_displacement_m"] for row in trial_rows)
        ),
        "safety_event_count": len(safety_events),
        "representative_sample_count": int(representative_raw["time_s"].size),
    }

    finite_fields = (
        "parameter_relative_error_rms",
        "normalized_condition_number",
        "maximum_abs_parameter_correlation",
    )
    checks = {
        "complete_trial_matrix": len(trial_rows) == expected_trials,
        "untouched_validation_seed_partition": bool(
            seeds and all(2000 <= seed <= 2999 and seed != 1101 for seed in seeds)
        ),
        "monte_carlo_seed_count_sufficient": len(seeds)
        >= int(config["integrity_acceptance"]["minimum_monte_carlo_seeds"]),
        "both_contact_modes_present": set(contact_modes) == {"bilateral", "unilateral"},
        "all_core_metrics_finite": all(
            np.isfinite(float(row[field])) for row in trial_rows for field in finite_fields
        ),
        "safety_limits_respected": not safety_events,
        "representative_raw_saved": bool(representative_raw),
        "go_no_go_evaluated": stage1_decision in {"GO", "CONTINUE_STAGE_1"},
    }
    return PracticalIdentifiabilityResult(
        trial_metrics=tuple(trial_rows),
        aggregate_metrics=aggregate,
        go_candidates=candidates,
        representative_raw=representative_raw,
        ramp_failure_analysis=ramp_analysis,
        metrics=summary,
        safety_events=tuple(safety_events),
        acceptance_checks=checks,
        success=all(checks.values()),
        stage1_decision=stage1_decision,
        best_candidate=best_candidate,
    )
