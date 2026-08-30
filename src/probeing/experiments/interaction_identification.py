"""EXP-0001: controlled Milestone A interaction-identification matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from probeing.estimators import batch_least_squares, recursive_least_squares
from probeing.measurements import MeasurementNoise, generate_measurements
from probeing.models import InteractionParameters, MassSpringDamperModel
from probeing.probing import make_probe


@dataclass(frozen=True)
class MilestoneAExperimentResult:
    raw: Mapping[str, NDArray[Any]]
    case_metrics: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float | int]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool
    representative_case_id: str


def _relative_error(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> NDArray[np.float64]:
    return (estimate - truth) / np.abs(truth)


def _rms(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _measurement_noise(config: Mapping[str, Any]) -> MeasurementNoise:
    return MeasurementNoise(
        displacement_std_m=float(config.get("displacement_std_m", 0.0)),
        velocity_std_m_per_s=float(config.get("velocity_std_m_per_s", 0.0)),
        acceleration_std_m_per_s2=float(config.get("acceleration_std_m_per_s2", 0.0)),
        force_std_n=float(config.get("force_std_n", 0.0)),
    )


def _safety_event(
    *, case_id: str, metric: str, observed: float, limit: float, units: str
) -> Mapping[str, Any]:
    return {
        "type": "limit_exceeded",
        "case_id": case_id,
        "metric": metric,
        "observed": observed,
        "limit": limit,
        "units": units,
    }


def run_interaction_identification(
    config: Mapping[str, Any],
) -> MilestoneAExperimentResult:
    """Run all configured target, probe, and measurement-scenario combinations."""

    experiment = config["experiment"]
    if experiment["id"] != "EXP-0001":
        raise ValueError("this runner only implements EXP-0001")
    simulation_config = config["simulation"]
    sample_period_s = float(simulation_config["sample_period_s"])
    duration_s = float(simulation_config["duration_s"])
    if simulation_config.get("integrator", "rk4") != "rk4":
        raise ValueError("Milestone A currently supports only the validated RK4 integrator")

    targets = config["targets"]
    probes = config["probes"]
    scenarios = config["measurement_scenarios"]
    if not targets or not probes or not scenarios:
        raise ValueError("targets, probes, and measurement_scenarios cannot be empty")

    base_seed = int(experiment["random_seed"])
    rls_config = config["estimators"]["recursive_least_squares"]
    safety_limits = config["safety_limits"]
    raw_parts: dict[str, list[NDArray[Any]]] = {}
    case_metrics: list[Mapping[str, Any]] = []
    safety_events: list[Mapping[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios):
        scenario_name = str(scenario["name"])
        noise = _measurement_noise(scenario["noise"])
        seed_offset = int(scenario.get("seed_offset", 10_000 * scenario_index))
        for target_index, target in enumerate(targets):
            target_name = str(target["name"])
            parameters = InteractionParameters(
                stiffness_n_per_m=float(target["stiffness_n_per_m"]),
                damping_n_s_per_m=float(target["damping_n_s_per_m"]),
                effective_mass_kg=float(target["effective_mass_kg"]),
            )
            true_parameters = np.asarray(
                [
                    parameters.stiffness_n_per_m,
                    parameters.damping_n_s_per_m,
                    parameters.effective_mass_kg,
                ],
                dtype=float,
            )
            model = MassSpringDamperModel(parameters)
            for probe_index, probe_config in enumerate(probes):
                probe = make_probe(dict(probe_config), sample_period_s, duration_s)
                probe_name = probe.name
                case_seed = base_seed + seed_offset + 100 * target_index + probe_index
                case_id = f"{scenario_name}__{target_name}__{probe_name}"
                truth = model.simulate(probe.time_s, probe.force_n)
                measurements = generate_measurements(
                    truth, noise, random_seed=case_seed
                )
                batch = batch_least_squares(measurements)
                recursive = recursive_least_squares(
                    measurements,
                    forgetting_factor=float(rls_config["forgetting_factor"]),
                    initial_covariance=float(rls_config["initial_covariance"]),
                    initial_parameters=rls_config["initial_parameters"],
                )
                batch_relative = _relative_error(batch.parameters, true_parameters)
                recursive_relative = _relative_error(recursive.parameters, true_parameters)
                recursive_history_relative = (
                    recursive.parameter_history - true_parameters[None, :]
                ) / np.abs(true_parameters[None, :])
                diagnostics = batch.diagnostics
                correlation = diagnostics.parameter_correlation

                peak_values = {
                    "peak_abs_probe_force_n": float(np.max(np.abs(probe.force_n))),
                    "peak_abs_displacement_m": float(np.max(np.abs(truth.displacement_m))),
                    "peak_abs_velocity_m_per_s": float(
                        np.max(np.abs(truth.velocity_m_per_s))
                    ),
                    "peak_abs_acceleration_m_per_s2": float(
                        np.max(np.abs(truth.acceleration_m_per_s2))
                    ),
                }
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "scenario": scenario_name,
                    "target": target_name,
                    "probe": probe_name,
                    "random_seed": case_seed,
                    "true_stiffness_n_per_m": true_parameters[0],
                    "true_damping_n_s_per_m": true_parameters[1],
                    "true_effective_mass_kg": true_parameters[2],
                    "batch_stiffness_n_per_m": batch.parameters[0],
                    "batch_damping_n_s_per_m": batch.parameters[1],
                    "batch_effective_mass_kg": batch.parameters[2],
                    "batch_stiffness_relative_error": batch_relative[0],
                    "batch_damping_relative_error": batch_relative[1],
                    "batch_effective_mass_relative_error": batch_relative[2],
                    "batch_relative_error_rms": _rms(batch_relative),
                    "batch_force_rmse_n": batch.force_rmse_n,
                    "rls_stiffness_n_per_m": recursive.parameters[0],
                    "rls_damping_n_s_per_m": recursive.parameters[1],
                    "rls_effective_mass_kg": recursive.parameters[2],
                    "rls_stiffness_relative_error": recursive_relative[0],
                    "rls_damping_relative_error": recursive_relative[1],
                    "rls_effective_mass_relative_error": recursive_relative[2],
                    "rls_relative_error_rms": _rms(recursive_relative),
                    "regression_rank": diagnostics.rank,
                    "normalized_condition_number": diagnostics.normalized_condition_number,
                    "normalized_min_singular_value": diagnostics.normalized_singular_values[-1],
                    "parameter_correlation_k_c": correlation[0, 1],
                    "parameter_correlation_k_m": correlation[0, 2],
                    "parameter_correlation_c_m": correlation[1, 2],
                    "maximum_abs_parameter_correlation": diagnostics.maximum_abs_parameter_correlation,
                    **peak_values,
                }
                case_metrics.append(row)

                limits = (
                    ("peak_abs_probe_force_n", "max_abs_probe_force_n", "N"),
                    ("peak_abs_displacement_m", "max_abs_displacement_m", "m"),
                    ("peak_abs_velocity_m_per_s", "max_abs_velocity_m_per_s", "m/s"),
                    (
                        "peak_abs_acceleration_m_per_s2",
                        "max_abs_acceleration_m_per_s2",
                        "m/s^2",
                    ),
                )
                for metric_name, limit_name, units in limits:
                    observed = peak_values[metric_name]
                    limit = float(safety_limits[limit_name])
                    if observed > limit:
                        safety_events.append(
                            _safety_event(
                                case_id=case_id,
                                metric=metric_name,
                                observed=observed,
                                limit=limit,
                                units=units,
                            )
                        )

                sample_count = truth.time_s.size
                raw_case: Mapping[str, NDArray[Any]] = {
                    "case_id": np.full(sample_count, case_id),
                    "scenario": np.full(sample_count, scenario_name),
                    "target": np.full(sample_count, target_name),
                    "probe": np.full(sample_count, probe_name),
                    "random_seed": np.full(sample_count, case_seed, dtype=np.int64),
                    "time_s": truth.time_s,
                    "probe_force_n": probe.force_n,
                    "true_displacement_m": truth.displacement_m,
                    "measured_displacement_m": measurements.displacement_m,
                    "true_velocity_m_per_s": truth.velocity_m_per_s,
                    "measured_velocity_m_per_s": measurements.velocity_m_per_s,
                    "true_acceleration_m_per_s2": truth.acceleration_m_per_s2,
                    "measured_acceleration_m_per_s2": measurements.acceleration_m_per_s2,
                    "true_contact_force_n": truth.applied_force_n,
                    "measured_contact_force_n": measurements.contact_force_n,
                    "rls_stiffness_n_per_m": recursive.parameter_history[:, 0],
                    "rls_damping_n_s_per_m": recursive.parameter_history[:, 1],
                    "rls_effective_mass_kg": recursive.parameter_history[:, 2],
                    "rls_stiffness_relative_error": recursive_history_relative[:, 0],
                    "rls_damping_relative_error": recursive_history_relative[:, 1],
                    "rls_effective_mass_relative_error": recursive_history_relative[:, 2],
                }
                for name, values in raw_case.items():
                    raw_parts.setdefault(name, []).append(np.asarray(values))

    raw = {name: np.concatenate(parts) for name, parts in raw_parts.items()}
    perfect_name = str(config["acceptance"]["perfect_scenario"])
    near_ideal_name = str(config["acceptance"]["near_ideal_scenario"])
    perfect_rows = [row for row in case_metrics if row["scenario"] == perfect_name]
    near_ideal_rows = [row for row in case_metrics if row["scenario"] == near_ideal_name]
    if not perfect_rows or not near_ideal_rows:
        raise ValueError("acceptance scenarios must both be represented")

    metrics: dict[str, float | int] = {
        "case_count": len(case_metrics),
        "sample_count": int(raw["time_s"].size),
        "perfect_case_count": len(perfect_rows),
        "near_ideal_case_count": len(near_ideal_rows),
        "minimum_regression_rank_perfect": min(
            int(row["regression_rank"]) for row in perfect_rows
        ),
        "maximum_normalized_condition_number_perfect": max(
            float(row["normalized_condition_number"]) for row in perfect_rows
        ),
        "maximum_abs_parameter_correlation_perfect": max(
            float(row["maximum_abs_parameter_correlation"]) for row in perfect_rows
        ),
        "maximum_perfect_batch_relative_error_rms": max(
            float(row["batch_relative_error_rms"]) for row in perfect_rows
        ),
        "maximum_perfect_rls_relative_error_rms": max(
            float(row["rls_relative_error_rms"]) for row in perfect_rows
        ),
        "median_near_ideal_batch_relative_error_rms": float(
            np.median([row["batch_relative_error_rms"] for row in near_ideal_rows])
        ),
        "maximum_near_ideal_batch_relative_error_rms": max(
            float(row["batch_relative_error_rms"]) for row in near_ideal_rows
        ),
        "median_near_ideal_rls_relative_error_rms": float(
            np.median([row["rls_relative_error_rms"] for row in near_ideal_rows])
        ),
        "maximum_near_ideal_rls_relative_error_rms": max(
            float(row["rls_relative_error_rms"]) for row in near_ideal_rows
        ),
        "maximum_abs_probe_force_n": max(
            float(row["peak_abs_probe_force_n"]) for row in case_metrics
        ),
        "maximum_abs_displacement_m": max(
            float(row["peak_abs_displacement_m"]) for row in case_metrics
        ),
        "maximum_abs_velocity_m_per_s": max(
            float(row["peak_abs_velocity_m_per_s"]) for row in case_metrics
        ),
        "maximum_abs_acceleration_m_per_s2": max(
            float(row["peak_abs_acceleration_m_per_s2"]) for row in case_metrics
        ),
        "safety_event_count": len(safety_events),
    }

    acceptance = config["acceptance"]
    checks = {
        "all_perfect_cases_full_rank": int(metrics["minimum_regression_rank_perfect"]) == 3,
        "normalized_condition_within_limit": float(
            metrics["maximum_normalized_condition_number_perfect"]
        )
        <= float(acceptance["max_normalized_condition_number"]),
        "perfect_batch_error_within_tolerance": float(
            metrics["maximum_perfect_batch_relative_error_rms"]
        )
        <= float(acceptance["max_perfect_batch_relative_error_rms"]),
        "perfect_rls_error_within_tolerance": float(
            metrics["maximum_perfect_rls_relative_error_rms"]
        )
        <= float(acceptance["max_perfect_rls_relative_error_rms"]),
        "near_ideal_batch_median_within_tolerance": float(
            metrics["median_near_ideal_batch_relative_error_rms"]
        )
        <= float(acceptance["max_near_ideal_batch_median_relative_error_rms"]),
        "near_ideal_rls_median_within_tolerance": float(
            metrics["median_near_ideal_rls_relative_error_rms"]
        )
        <= float(acceptance["max_near_ideal_rls_median_relative_error_rms"]),
        "safety_limits_respected": not safety_events,
    }
    representative = config["plotting"]["representative_case"]
    representative_case_id = (
        f"{representative['scenario']}__{representative['target']}__{representative['probe']}"
    )
    if representative_case_id not in set(raw["case_id"]):
        raise ValueError("configured representative case does not exist")
    return MilestoneAExperimentResult(
        raw=raw,
        case_metrics=tuple(case_metrics),
        metrics=metrics,
        safety_events=tuple(safety_events),
        acceptance_checks=checks,
        success=all(checks.values()),
        representative_case_id=representative_case_id,
    )
