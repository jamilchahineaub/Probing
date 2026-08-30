"""EXP-0003 causal timing and errors-in-variables identifiability study."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from probeing.estimators import (
    delayed_input_instruments,
    dynamic_ratio_least_squares,
    estimate_eiv,
    regression_matrix,
)
from probeing.measurements import (
    MeasurementNoise,
    SyntheticMeasurements,
    backward_difference,
    estimate_signal_delay,
    process_causal_sensing,
)
from probeing.metrics import physical_disturbance_metrics, rms, true_modal_parameters
from probeing.models import (
    ContactInteractionSimulation,
    InteractionParameters,
    MassSpringDamperModel,
    simulate_contact_interaction,
)
from probeing.probing import chirp


PARAMETERS = ("stiffness", "damping", "effective_mass")
MODAL_PARAMETERS = ("natural_frequency", "damping_ratio", "inverse_mass")


@dataclass(frozen=True)
class CausalEIVIdentifiabilityResult:
    timing_trials: tuple[Mapping[str, Any], ...]
    timing_aggregate: tuple[Mapping[str, Any], ...]
    timing_profile_trials: tuple[Mapping[str, Any], ...]
    timing_profile_aggregate: tuple[Mapping[str, Any], ...]
    identification_trials: tuple[Mapping[str, Any], ...]
    identification_aggregate: tuple[Mapping[str, Any], ...]
    parameter_classifications: tuple[Mapping[str, Any], ...]
    timing_thresholds: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool
    stage1_decision: str


def _parameters(target: Mapping[str, Any]) -> InteractionParameters:
    return InteractionParameters(
        stiffness_n_per_m=float(target["stiffness_n_per_m"]),
        damping_n_s_per_m=float(target["damping_n_s_per_m"]),
        effective_mass_kg=float(target["effective_mass_kg"]),
    )


def _truth_vector(parameters: InteractionParameters) -> NDArray[np.float64]:
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
    factor = 0.0 if zero else 1.0
    return MeasurementNoise(
        displacement_std_m=factor * float(values["displacement_std_m"]),
        velocity_std_m_per_s=factor * float(values["velocity_std_m_per_s"]),
        acceleration_std_m_per_s2=factor * float(values["acceleration_std_m_per_s2"]),
        force_std_n=factor * float(values["force_std_n"]),
    )


def _trim_measurements(
    measurements: SyntheticMeasurements, trim_s: float
) -> SyntheticMeasurements:
    time = measurements.time_s
    mask = (time >= time[0] + trim_s) & (time <= time[-1] - trim_s)
    if np.count_nonzero(mask) < 20:
        raise ValueError("analysis trim leaves too few samples")
    return SyntheticMeasurements(
        time_s=time[mask],
        displacement_m=measurements.displacement_m[mask],
        velocity_m_per_s=measurements.velocity_m_per_s[mask],
        acceleration_m_per_s2=measurements.acceleration_m_per_s2[mask],
        contact_force_n=measurements.contact_force_n[mask],
        noise=measurements.noise,
        random_seed=measurements.random_seed,
    )


def _trim_array(time: NDArray[np.float64], values: NDArray[np.float64], trim_s: float) -> NDArray[np.float64]:
    mask = (time >= time[0] + trim_s) & (time <= time[-1] - trim_s)
    return np.asarray(values[mask], dtype=float)


def _simulate_truth(
    target: Mapping[str, Any], band: Mapping[str, Any], config: Mapping[str, Any]
) -> ContactInteractionSimulation:
    simulation = config["simulation"]
    probe = config["probe"]
    signal = chirp(
        float(probe["amplitude_n"]),
        float(band["start_frequency_hz"]),
        float(band["end_frequency_hz"]),
        float(simulation["duration_s"]),
        float(simulation["sample_period_s"]),
    )
    return simulate_contact_interaction(
        MassSpringDamperModel(_parameters(target)),
        signal.time_s,
        signal.force_n,
        contact_mode=str(simulation["contact_mode"]),
    )


def _safe_modal(parameters: NDArray[np.float64]) -> tuple[bool, float, float, float]:
    k, c, mass = (float(value) for value in parameters)
    valid = bool(np.all(np.isfinite(parameters)) and k > 0.0 and mass > 0.0)
    if not valid:
        return False, float("nan"), float("nan"), float("nan")
    natural = float(np.sqrt(k / mass))
    damping_ratio = float(c / (2.0 * np.sqrt(k * mass)))
    return True, natural, damping_ratio, 1.0 / mass


def _relative(estimate: float, truth: float) -> float:
    return float((estimate - truth) / abs(truth))


def _geometry(design: NDArray[np.float64], force_noise_std_n: float, truth: NDArray[np.float64]) -> Mapping[str, float]:
    variance = np.var(design, axis=0)
    correlation = np.corrcoef(design, rowvar=False)
    sigma_squared = max(force_noise_std_n**2, np.finfo(float).eps)
    relative_information = []
    for index in range(3):
        other = np.delete(design, index, axis=1)
        coefficients, _, _, _ = np.linalg.lstsq(other, design[:, index], rcond=None)
        residual = design[:, index] - other @ coefficients
        relative_information.append(float(truth[index] ** 2 * np.sum(residual**2) / sigma_squared))
    return {
        "displacement_regressor_variance": float(variance[0]),
        "velocity_regressor_variance": float(variance[1]),
        "acceleration_regressor_variance": float(variance[2]),
        "displacement_regressor_rms": rms(design[:, 0]),
        "velocity_regressor_rms": rms(design[:, 1]),
        "acceleration_regressor_rms": rms(design[:, 2]),
        "cross_correlation_x_v": float(correlation[0, 1]),
        "cross_correlation_x_a": float(correlation[0, 2]),
        "cross_correlation_v_a": float(correlation[1, 2]),
        "relative_fisher_information_stiffness": relative_information[0],
        "relative_fisher_information_damping": relative_information[1],
        "relative_fisher_information_effective_mass": relative_information[2],
        "relative_fisher_information_geometric_mean": float(
            np.prod(np.maximum(relative_information, np.finfo(float).tiny)) ** (1.0 / 3.0)
        ),
    }


def _safety(
    disturbance: Mapping[str, float], limits: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    comparisons = {
        "peak_force": ("peak_force_n", "max_peak_force_n"),
        "peak_displacement": (
            "peak_target_displacement_m",
            "max_peak_target_displacement_m",
        ),
        "peak_velocity": (
            "peak_target_velocity_m_per_s",
            "max_peak_target_velocity_m_per_s",
        ),
        "peak_acceleration": (
            "peak_target_acceleration_m_per_s2",
            "max_peak_target_acceleration_m_per_s2",
        ),
    }
    violations = [
        label
        for label, (metric, limit) in comparisons.items()
        if float(disturbance[metric]) > float(limits[limit])
    ]
    return not violations, violations


def _estimation_fields(
    estimated: NDArray[np.float64],
    truth: NDArray[np.float64],
    true_natural: float,
    true_zeta: float,
) -> Mapping[str, Any]:
    valid_modal, natural, zeta, inverse_mass = _safe_modal(estimated)
    return {
        "estimated_stiffness_n_per_m": float(estimated[0]),
        "estimated_damping_n_s_per_m": float(estimated[1]),
        "estimated_effective_mass_kg": float(estimated[2]),
        "stiffness_relative_error": _relative(float(estimated[0]), float(truth[0])),
        "damping_relative_error": _relative(float(estimated[1]), float(truth[1])),
        "effective_mass_relative_error": _relative(float(estimated[2]), float(truth[2])),
        "direct_modal_valid": valid_modal,
        "direct_natural_frequency_rad_per_s": natural,
        "direct_damping_ratio": zeta,
        "direct_inverse_mass_per_kg": inverse_mass,
        "direct_natural_frequency_relative_error": _relative(natural, true_natural)
        if valid_modal
        else float("nan"),
        "direct_damping_ratio_relative_error": _relative(zeta, true_zeta)
        if valid_modal
        else float("nan"),
        "direct_inverse_mass_relative_error": _relative(inverse_mass, 1.0 / truth[2])
        if valid_modal
        else float("nan"),
    }


def _timing_conditions(config: Mapping[str, Any]) -> list[tuple[str, float, Mapping[str, float], float]]:
    output: list[tuple[str, float, Mapping[str, float], float]] = []
    offsets = [float(value) / 1000.0 for value in config["timing_study"]["offset_values_ms"]]
    for channel in ("displacement", "acceleration", "force"):
        for value in offsets:
            channel_offsets = {channel: value}
            if channel == "displacement":
                channel_offsets["velocity"] = value
            output.append((channel, value, channel_offsets, 0.0))
    for value_ms in config["timing_study"]["group_delay_values_ms"]:
        value = float(value_ms) / 1000.0
        output.append(("filter_group_delay", value, {}, value))
    return output


def _timing_row(
    *,
    target: str,
    seed: int,
    pipeline: str,
    sweep_channel: str,
    offset_s: float,
    profile: str,
    estimated: NDArray[np.float64],
    truth: NDArray[np.float64],
    valid: bool,
) -> Mapping[str, Any]:
    return {
        "target": target,
        "validation_seed": seed,
        "pipeline": pipeline,
        "sweep_channel": sweep_channel,
        "timing_offset_s": offset_s,
        "timing_offset_ms": 1000.0 * offset_s,
        "timing_profile": profile,
        "estimator": "ols",
        "valid": valid,
        "stiffness_relative_error": _relative(float(estimated[0]), float(truth[0])),
        "damping_relative_error": _relative(float(estimated[1]), float(truth[1])),
        "effective_mass_relative_error": _relative(float(estimated[2]), float(truth[2])),
    }


def _aggregate_timing(
    rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for values, group in grouped.items():
        result: dict[str, Any] = dict(zip(group_keys, values))
        result["trial_count"] = len(group)
        result["failure_fraction"] = float(np.mean([not bool(row["valid"]) for row in group]))
        for label in PARAMETERS:
            errors = np.asarray([row[f"{label}_relative_error"] for row in group], dtype=float)
            result[f"{label}_relative_bias"] = float(np.mean(errors))
            result[f"{label}_median_abs_relative_error"] = float(np.median(np.abs(errors)))
            result[f"{label}_p95_abs_relative_error"] = float(np.percentile(np.abs(errors), 95.0))
        output.append(result)
    return tuple(output)


def _aggregate_identification(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    keys = ("target", "frequency_band", "pipeline", "estimator", "pipeline_is_causal", "safety_pass")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    error_fields = {
        **{label: f"{label}_relative_error" for label in PARAMETERS},
        "direct_natural_frequency": "direct_natural_frequency_relative_error",
        "direct_damping_ratio": "direct_damping_ratio_relative_error",
        "direct_inverse_mass": "direct_inverse_mass_relative_error",
        "ratio_natural_frequency": "ratio_natural_frequency_relative_error",
        "ratio_damping_ratio": "ratio_damping_ratio_relative_error",
        "ratio_inverse_mass": "ratio_inverse_mass_relative_error",
    }
    for values, group in grouped.items():
        result: dict[str, Any] = dict(zip(keys, values))
        result["trial_count"] = len(group)
        result["numerical_failure_fraction"] = float(np.mean([not bool(row["estimator_valid"]) for row in group]))
        result["nonphysical_fraction"] = float(np.mean([not bool(row["physical_estimate_valid"]) for row in group]))
        for label, field in error_fields.items():
            errors = np.asarray([row[field] for row in group], dtype=float)
            errors = errors[np.isfinite(errors)]
            if errors.size:
                result[f"{label}_relative_bias"] = float(np.mean(errors))
                result[f"{label}_relative_error_median"] = float(np.median(np.abs(errors)))
                result[f"{label}_p95_abs_relative_error"] = float(np.percentile(np.abs(errors), 95.0))
                result[f"{label}_worst_abs_relative_error"] = float(np.max(np.abs(errors)))
            else:
                result[f"{label}_relative_bias"] = 1.0e9
                result[f"{label}_relative_error_median"] = 1.0e9
                result[f"{label}_p95_abs_relative_error"] = 1.0e9
                result[f"{label}_worst_abs_relative_error"] = 1.0e9
        median_fields = (
            "force_fit_rmse_n",
            "normalized_condition_number",
            "maximum_abs_parameter_correlation",
            "displacement_regressor_variance",
            "velocity_regressor_variance",
            "acceleration_regressor_variance",
            "displacement_regressor_rms",
            "velocity_regressor_rms",
            "acceleration_regressor_rms",
            "cross_correlation_x_v",
            "cross_correlation_x_a",
            "cross_correlation_v_a",
            "relative_fisher_information_stiffness",
            "relative_fisher_information_damping",
            "relative_fisher_information_effective_mass",
            "relative_fisher_information_geometric_mean",
            "effective_acceleration_delay_s",
            "acceleration_noise_attenuation_db",
            "computational_cost_units_per_sample",
            "peak_force_n",
            "absolute_input_energy_j",
            "peak_target_displacement_m",
            "peak_target_velocity_m_per_s",
            "peak_target_acceleration_m_per_s2",
        )
        for field in median_fields:
            result[field] = float(np.median([row[field] for row in group]))
        result["minimum_instrument_strength"] = float(np.median([row["minimum_instrument_strength"] for row in group]))
        output.append(result)
    return tuple(output)


def _parameter_classifications(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    criteria = config["classification"]
    target_names = [str(target["name"]) for target in config["targets"]]
    mappings = {
        "stiffness": "stiffness_relative_error",
        "damping": "damping_relative_error",
        "effective_mass": "effective_mass_relative_error",
        "natural_frequency": "ratio_natural_frequency_relative_error",
        "damping_ratio": "ratio_damping_ratio_relative_error",
        "inverse_mass": "ratio_inverse_mass_relative_error",
    }
    candidates: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["pipeline_is_causal"]) and bool(row["safety_pass"]):
            candidates[(str(row["frequency_band"]), str(row["pipeline"]), str(row["estimator"]))].append(row)

    classifications = []
    for parameter, field in mappings.items():
        candidate_rows = []
        for (band, pipeline, estimator), group in candidates.items():
            if parameter in MODAL_PARAMETERS and estimator != "ols":
                continue
            passing_targets = 0
            target_p95 = []
            target_bias = []
            target_failure = []
            for target in target_names:
                subset = [row for row in group if row["target"] == target]
                values = np.asarray([row[field] for row in subset], dtype=float)
                finite = values[np.isfinite(values)]
                failure = 1.0 - finite.size / max(values.size, 1)
                p95 = float(np.percentile(np.abs(finite), 95.0)) if finite.size else 1.0e9
                bias = abs(float(np.mean(finite))) if finite.size else 1.0e9
                target_p95.append(p95)
                target_bias.append(bias)
                target_failure.append(failure)
                if (
                    p95 <= float(criteria["max_p95_abs_relative_error"][parameter])
                    and bias <= float(criteria["max_abs_relative_bias"][parameter])
                    and failure <= float(criteria["max_failure_fraction"])
                ):
                    passing_targets += 1
            candidate_rows.append(
                {
                    "frequency_band": band,
                    "pipeline": pipeline,
                    "estimator": estimator,
                    "passing_target_count": passing_targets,
                    "passing_target_fraction": passing_targets / len(target_names),
                    "worst_target_p95_abs_relative_error": max(target_p95),
                    "worst_target_abs_relative_bias": max(target_bias),
                    "worst_target_failure_fraction": max(target_failure),
                }
            )
        candidate_rows.sort(
            key=lambda row: (
                -int(row["passing_target_count"]),
                float(row["worst_target_p95_abs_relative_error"]),
                float(row["worst_target_abs_relative_bias"]),
            )
        )
        best = candidate_rows[0]
        if int(best["passing_target_count"]) == len(target_names):
            category = "A_practically_identifiable"
        elif int(best["passing_target_count"]) >= int(criteria["restricted_min_target_count"]):
            category = "B_restricted_conditions"
        else:
            category = "C_not_practically_identifiable"
        classifications.append({"parameter": parameter, "category": category, **best})
    return tuple(classifications)


def _timing_thresholds(
    aggregate: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    thresholds = config["classification"]["practically_significant_timing_bias"]
    output = []
    for channel in ("displacement", "acceleration", "force", "filter_group_delay"):
        channel_rows = [row for row in aggregate if row["sweep_channel"] == channel]
        for parameter in PARAMETERS:
            baseline = {
                row["target"]: float(row[f"{parameter}_relative_bias"])
                for row in channel_rows
                if np.isclose(float(row["timing_offset_s"]), 0.0)
            }
            values = sorted({abs(float(row["timing_offset_s"])) for row in channel_rows if not np.isclose(float(row["timing_offset_s"]), 0.0)})
            crossing = None
            crossing_bias = None
            for value in values:
                at_value = [row for row in channel_rows if np.isclose(abs(float(row["timing_offset_s"])), value)]
                worst = max(
                    abs(float(row[f"{parameter}_relative_bias"]) - baseline[str(row["target"])])
                    for row in at_value
                )
                if worst >= float(thresholds[parameter]):
                    crossing = value
                    crossing_bias = worst
                    break
            output.append(
                {
                    "sweep_channel": channel,
                    "parameter": parameter,
                    "significant_bias_threshold": float(thresholds[parameter]),
                    "minimum_significant_offset_ms": -1.0 if crossing is None else 1000.0 * crossing,
                    "worst_incremental_bias_at_crossing": -1.0 if crossing_bias is None else crossing_bias,
                }
            )
    return tuple(output)


def _stage_decision(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any]]:
    criteria = config["classification"]
    targets = [str(target["name"]) for target in config["targets"]]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["pipeline_is_causal"]) and bool(row["safety_pass"]):
            grouped[(str(row["frequency_band"]), str(row["pipeline"]), str(row["estimator"]))].append(row)

    def passes(group: Sequence[Mapping[str, Any]], target: str, parameter: str, field: str) -> bool:
        subset = [row for row in group if row["target"] == target]
        values = np.asarray([row[field] for row in subset], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size / max(values.size, 1) < 1.0 - float(criteria["max_failure_fraction"]):
            return False
        return bool(
            np.percentile(np.abs(finite), 95.0) <= float(criteria["max_p95_abs_relative_error"][parameter])
            and abs(np.mean(finite)) <= float(criteria["max_abs_relative_bias"][parameter])
        )

    passing = []
    for key, group in grouped.items():
        physical_ok = all(
            passes(group, target, "stiffness", "stiffness_relative_error")
            and passes(group, target, "damping", "damping_relative_error")
            for target in targets
        )
        mass_ok = all(
            passes(group, target, "effective_mass", "effective_mass_relative_error")
            for target in targets
        )
        modal_ok = all(
            passes(group, target, "natural_frequency", "ratio_natural_frequency_relative_error")
            and passes(group, target, "damping_ratio", "ratio_damping_ratio_relative_error")
            for target in targets
        )
        if physical_ok and (mass_ok or modal_ok):
            passing.append(
                {
                    "frequency_band": key[0],
                    "pipeline": key[1],
                    "estimator": key[2],
                    "mass_pass": mass_ok,
                    "modal_pass": modal_ok,
                }
            )
    decision = "READY_FOR_INDEPENDENT_MATLAB_VALIDATION" if passing else "CONTINUE_STAGE_1"
    return decision, {"passing_candidates": passing, "passing_candidate_count": len(passing)}


def run_causal_eiv_identifiability(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CausalEIVIdentifiabilityResult:
    """Run EXP-0003 without writing artifacts."""

    seeds = [int(seed) for seed in config["validation_seeds"]]
    targets = list(config["targets"])
    bands = list(config["chirp_frequency_bands"])
    pipelines = [str(entry["name"]) for entry in config["pipelines"]]
    estimators = [str(name) for name in config["estimators"]["names"]]
    baseline_name = str(config["probe"]["baseline_band"])
    baseline_band = next(band for band in bands if band["name"] == baseline_name)
    sample_rate = float(config["sensing"]["sample_rate_hz"])
    trim_s = float(config["sensing"]["analysis_trim_s"])
    settings = config["pipeline_settings"]
    noisy = _noise(config)
    clean_noise = _noise(config, zero=True)

    truth_cache: dict[tuple[str, str], ContactInteractionSimulation] = {}
    disturbance_cache: dict[tuple[str, str], Mapping[str, float]] = {}
    safety_cache: dict[tuple[str, str], bool] = {}
    safety_events: list[Mapping[str, Any]] = []
    for target in targets:
        for band in bands:
            key = (str(target["name"]), str(band["name"]))
            truth = _simulate_truth(target, band, config)
            truth_cache[key] = truth
            disturbance = physical_disturbance_metrics(truth)
            disturbance_cache[key] = disturbance
            safe, violations = _safety(disturbance, config["safety_limits"])
            safety_cache[key] = safe
            if not safe:
                safety_events.append(
                    {"target": key[0], "frequency_band": key[1], "violations": violations, **disturbance}
                )

    timing_conditions = _timing_conditions(config)
    expected_timing = len(targets) * len(seeds) * len(timing_conditions)
    expected_profiles = len(targets) * len(seeds) * len(pipelines) * len(config["timing_study"]["profiles"])
    expected_identification = len(targets) * len(bands) * len(seeds) * len(pipelines) * len(estimators)
    total = expected_timing + expected_profiles + expected_identification
    completed = 0

    timing_rows: list[Mapping[str, Any]] = []
    for target in targets:
        target_name = str(target["name"])
        truth = truth_cache[(target_name, baseline_name)]
        truth_vector = _truth_vector(_parameters(target))
        for seed in seeds:
            for channel, offset, offsets, group_delay in timing_conditions:
                sensing = process_causal_sensing(
                    truth,
                    pipeline="direct",
                    sample_rate_hz=sample_rate,
                    noise=noisy,
                    pipeline_settings=settings,
                    random_seed=seed,
                    timestamp_offsets_s=offsets,
                    kinematic_group_delay_s=group_delay,
                )
                measurement = _trim_measurements(sensing.measurements, trim_s)
                estimate = estimate_eiv("ols", measurement)
                timing_rows.append(
                    _timing_row(
                        target=target_name,
                        seed=seed,
                        pipeline="direct",
                        sweep_channel=channel,
                        offset_s=offset,
                        profile="one_factor",
                        estimated=estimate.parameters,
                        truth=truth_vector,
                        valid=estimate.valid,
                    )
                )
                completed += 1
                if progress_callback and completed % 1000 == 0:
                    progress_callback(completed, total)

    profile_rows: list[Mapping[str, Any]] = []
    for target in targets:
        target_name = str(target["name"])
        truth = truth_cache[(target_name, baseline_name)]
        truth_vector = _truth_vector(_parameters(target))
        for seed in seeds:
            for profile in config["timing_study"]["profiles"]:
                generator = np.random.default_rng(seed + 7919)
                mismatch = float(profile["mismatch_std_s"])
                offsets = {
                    channel: float(profile[f"{channel}_offset_s"]) + (float(generator.normal(0.0, mismatch)) if mismatch else 0.0)
                    for channel in ("displacement", "velocity", "acceleration", "force")
                }
                for pipeline in pipelines:
                    sensing = process_causal_sensing(
                        truth,
                        pipeline=pipeline,
                        sample_rate_hz=sample_rate,
                        noise=noisy,
                        pipeline_settings=settings,
                        random_seed=seed,
                        timestamp_offsets_s=offsets,
                    )
                    estimate = estimate_eiv("ols", _trim_measurements(sensing.measurements, trim_s))
                    profile_rows.append(
                        _timing_row(
                            target=target_name,
                            seed=seed,
                            pipeline=pipeline,
                            sweep_channel="profile",
                            offset_s=0.0,
                            profile=str(profile["name"]),
                            estimated=estimate.parameters,
                            truth=truth_vector,
                            valid=estimate.valid,
                        )
                    )
                    completed += 1
                    if progress_callback and completed % 1000 == 0:
                        progress_callback(completed, total)

    identification_rows: list[Mapping[str, Any]] = []
    representative_parts: dict[str, list[NDArray[Any]]] = defaultdict(list)
    clean_cache: dict[tuple[str, str, str], Any] = {}
    for target in targets:
        target_name = str(target["name"])
        parameters = _parameters(target)
        truth_vector = _truth_vector(parameters)
        true_natural, true_zeta = true_modal_parameters(parameters)
        for band in bands:
            band_name = str(band["name"])
            key = (target_name, band_name)
            truth = truth_cache[key]
            disturbance = disturbance_cache[key]
            for pipeline in pipelines:
                clean_key = (target_name, band_name, pipeline)
                clean_sensing = process_causal_sensing(
                    truth,
                    pipeline=pipeline,
                    sample_rate_hz=sample_rate,
                    noise=clean_noise,
                    pipeline_settings=settings,
                    random_seed=1600,
                )
                clean_cache[clean_key] = clean_sensing
                clean_measurement = _trim_measurements(clean_sensing.measurements, trim_s)
                truth_design = np.column_stack(
                    (
                        _trim_array(clean_sensing.measurements.time_s, clean_sensing.true_displacement_m, trim_s),
                        _trim_array(clean_sensing.measurements.time_s, clean_sensing.true_velocity_m_per_s, trim_s),
                        _trim_array(clean_sensing.measurements.time_s, clean_sensing.true_acceleration_m_per_s2, trim_s),
                    )
                )
                geometry = _geometry(truth_design, noisy.force_std_n, truth_vector)
                clean_acceleration = clean_measurement.acceleration_m_per_s2
                true_acceleration = _trim_array(
                    clean_sensing.measurements.time_s,
                    clean_sensing.true_acceleration_m_per_s2,
                    trim_s,
                )
                effective_delay = estimate_signal_delay(
                    true_acceleration,
                    clean_acceleration,
                    1.0 / sample_rate,
                    maximum_delay_s=0.20,
                )
                for seed in seeds:
                    sensing = process_causal_sensing(
                        truth,
                        pipeline=pipeline,
                        sample_rate_hz=sample_rate,
                        noise=noisy,
                        pipeline_settings=settings,
                        random_seed=seed,
                    )
                    measurement = _trim_measurements(sensing.measurements, trim_s)
                    command_instrument = np.maximum(
                        _trim_array(sensing.measurements.time_s, sensing.commanded_force_n, trim_s),
                        0.0,
                    )
                    instruments = delayed_input_instruments(
                        command_instrument,
                        measurement.time_s,
                        config["estimators"]["iv_input_delays_s"],
                    )
                    ratio = dynamic_ratio_least_squares(measurement)
                    ratio_inverse_mass = float(ratio.ratios[2])
                    ratio_fields = {
                        "ratio_estimate_valid": bool(ratio.valid_physical_parameters),
                        "ratio_natural_frequency_rad_per_s": float(ratio.natural_frequency_rad_per_s),
                        "ratio_damping_ratio": float(ratio.damping_ratio),
                        "ratio_inverse_mass_per_kg": ratio_inverse_mass,
                        "ratio_natural_frequency_relative_error": _relative(
                            float(ratio.natural_frequency_rad_per_s), true_natural
                        )
                        if ratio.valid_physical_parameters
                        else float("nan"),
                        "ratio_damping_ratio_relative_error": _relative(float(ratio.damping_ratio), true_zeta)
                        if ratio.valid_physical_parameters
                        else float("nan"),
                        "ratio_inverse_mass_relative_error": _relative(
                            ratio_inverse_mass, 1.0 / truth_vector[2]
                        )
                        if ratio.valid_physical_parameters
                        else float("nan"),
                    }
                    noisy_acceleration = measurement.acceleration_m_per_s2
                    output_noise_rms = rms(noisy_acceleration - clean_acceleration)
                    noisy_raw = _trim_array(
                        sensing.measurements.time_s,
                        backward_difference(sensing.raw_displacement_m, 1.0 / sample_rate, derivative_order=2),
                        trim_s,
                    )
                    clean_raw = _trim_array(
                        clean_sensing.measurements.time_s,
                        backward_difference(clean_sensing.raw_displacement_m, 1.0 / sample_rate, derivative_order=2),
                        trim_s,
                    )
                    raw_noise_rms = rms(noisy_raw - clean_raw)
                    attenuation = 20.0 * np.log10(
                        max(raw_noise_rms, np.finfo(float).tiny)
                        / max(output_noise_rms, np.finfo(float).tiny)
                    )
                    for estimator_name in estimators:
                        estimate = estimate_eiv(
                            estimator_name,
                            measurement,
                            instruments=instruments if estimator_name == "iv" else None,
                        )
                        physical_valid, _, _, _ = _safe_modal(estimate.parameters)
                        diagnostics = estimate.diagnostics
                        estimator_fields = estimate.estimator_diagnostics
                        row = {
                            "target": target_name,
                            "frequency_band": band_name,
                            "start_frequency_hz": float(band["start_frequency_hz"]),
                            "end_frequency_hz": float(band["end_frequency_hz"]),
                            "validation_seed": seed,
                            "pipeline": pipeline,
                            "pipeline_is_causal": sensing.is_causal,
                            "estimator": estimator_name,
                            "sample_rate_hz": sample_rate,
                            "sample_count": measurement.time_s.size,
                            "estimator_valid": estimate.valid,
                            "physical_estimate_valid": physical_valid,
                            "true_stiffness_n_per_m": truth_vector[0],
                            "true_damping_n_s_per_m": truth_vector[1],
                            "true_effective_mass_kg": truth_vector[2],
                            "true_natural_frequency_rad_per_s": true_natural,
                            "true_damping_ratio": true_zeta,
                            **_estimation_fields(
                                estimate.parameters, truth_vector, true_natural, true_zeta
                            ),
                            **ratio_fields,
                            "force_fit_rmse_n": estimate.force_rmse_n,
                            "rank": diagnostics.rank,
                            "normalized_singular_value_1": float(diagnostics.normalized_singular_values[0]),
                            "normalized_singular_value_2": float(diagnostics.normalized_singular_values[1]),
                            "normalized_singular_value_3": float(diagnostics.normalized_singular_values[2]),
                            "normalized_condition_number": diagnostics.normalized_condition_number,
                            "parameter_correlation_k_c": float(diagnostics.parameter_correlation[0, 1]),
                            "parameter_correlation_k_m": float(diagnostics.parameter_correlation[0, 2]),
                            "parameter_correlation_c_m": float(diagnostics.parameter_correlation[1, 2]),
                            "maximum_abs_parameter_correlation": diagnostics.maximum_abs_parameter_correlation,
                            **geometry,
                            "effective_acceleration_delay_s": effective_delay,
                            "nominal_pipeline_delay_s": sensing.nominal_delay_s,
                            "required_lookahead_s": sensing.required_lookahead_s,
                            "acceleration_noise_attenuation_db": attenuation,
                            "computational_cost_units_per_sample": sensing.computational_cost_units_per_sample,
                            "minimum_instrument_strength": float(estimator_fields.get("minimum_instrument_strength", -1.0)),
                            "mean_instrument_strength": float(estimator_fields.get("mean_instrument_strength", -1.0)),
                            "safety_pass": safety_cache[key],
                            **disturbance,
                            "information_per_peak_force": geometry["relative_fisher_information_geometric_mean"]
                            / max(float(disturbance["peak_force_n"]), np.finfo(float).eps),
                            "information_per_peak_displacement": geometry["relative_fisher_information_geometric_mean"]
                            / max(float(disturbance["peak_target_displacement_m"]), np.finfo(float).eps),
                            "information_per_input_energy": geometry["relative_fisher_information_geometric_mean"]
                            / max(float(disturbance["absolute_input_energy_j"]), np.finfo(float).eps),
                            "information_per_peak_acceleration": geometry["relative_fisher_information_geometric_mean"]
                            / max(float(disturbance["peak_target_acceleration_m_per_s2"]), np.finfo(float).eps),
                        }
                        identification_rows.append(row)
                        completed += 1
                        if progress_callback and completed % 1000 == 0:
                            progress_callback(completed, total)

                    representative = config["representative_raw"]
                    if (
                        target_name == str(representative["target"])
                        and band_name == str(representative["band"])
                        and seed == int(representative["seed"])
                    ):
                        count = sensing.measurements.time_s.size
                        values = {
                            "case_id": np.full(count, f"{target_name}__{band_name}__{pipeline}__s{seed}"),
                            "pipeline": np.full(count, pipeline),
                            "time_s": sensing.measurements.time_s,
                            "true_displacement_m": sensing.true_displacement_m,
                            "raw_displacement_m": sensing.raw_displacement_m,
                            "processed_displacement_m": sensing.measurements.displacement_m,
                            "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
                            "processed_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
                            "true_acceleration_m_per_s2": sensing.true_acceleration_m_per_s2,
                            "processed_acceleration_m_per_s2": sensing.measurements.acceleration_m_per_s2,
                            "true_force_n": sensing.true_contact_force_n,
                            "processed_force_n": sensing.measurements.contact_force_n,
                        }
                        for field, array in values.items():
                            representative_parts[field].append(np.asarray(array))

    if progress_callback:
        progress_callback(completed, total)

    timing_aggregate = _aggregate_timing(
        timing_rows, ("target", "pipeline", "sweep_channel", "timing_offset_s", "timing_offset_ms")
    )
    profile_aggregate = _aggregate_timing(
        profile_rows, ("target", "pipeline", "timing_profile")
    )
    identification_aggregate = _aggregate_identification(identification_rows)
    classifications = _parameter_classifications(identification_rows, config)
    thresholds = _timing_thresholds(timing_aggregate, config)
    stage_decision, gate_details = _stage_decision(identification_rows, config)

    classification_map = {row["parameter"]: row for row in classifications}
    causal_rows = [row for row in identification_aggregate if bool(row["pipeline_is_causal"]) and bool(row["safety_pass"])]
    best_ols = min(
        [row for row in causal_rows if row["estimator"] == "ols"],
        key=lambda row: rms(
            [
                row["stiffness_p95_abs_relative_error"],
                row["damping_p95_abs_relative_error"],
                row["effective_mass_p95_abs_relative_error"],
            ]
        ),
    )
    estimator_scores = {}
    for estimator in estimators:
        rows_for_estimator = [row for row in causal_rows if row["estimator"] == estimator]
        estimator_scores[estimator] = float(
            np.median(
                [
                    rms(
                        [
                            row["stiffness_p95_abs_relative_error"],
                            row["damping_p95_abs_relative_error"],
                            row["effective_mass_p95_abs_relative_error"],
                        ]
                    )
                    for row in rows_for_estimator
                ]
            )
        )
    best_estimator = min(estimator_scores, key=estimator_scores.get)
    best_causal_pipeline = min(
        {str(row["pipeline"]) for row in causal_rows},
        key=lambda pipeline: np.median(
            [
                rms(
                    [
                        row["stiffness_p95_abs_relative_error"],
                        row["damping_p95_abs_relative_error"],
                        row["effective_mass_p95_abs_relative_error"],
                    ]
                )
                for row in causal_rows
                if row["pipeline"] == pipeline
            ]
        ),
    )
    timing_profile_summary = {}
    def median_or_nan(values: Sequence[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    for pipeline in pipelines:
        timing_profile_summary[pipeline] = {}
        for parameter in PARAMETERS:
            synchronized = [
                row[f"{parameter}_median_abs_relative_error"]
                for row in profile_aggregate
                if row["pipeline"] == pipeline and row["timing_profile"] == "synchronized"
            ]
            exp2_like = [
                row[f"{parameter}_median_abs_relative_error"]
                for row in profile_aggregate
                if row["pipeline"] == pipeline and row["timing_profile"] == "exp_0002_nominal_plus_mismatch"
            ]
            timing_profile_summary[pipeline][parameter] = {
                "synchronized_median_abs_relative_error": median_or_nan(synchronized),
                "exp_0002_like_median_abs_relative_error": median_or_nan(exp2_like),
                "absolute_error_increase": (
                    median_or_nan(exp2_like) - median_or_nan(synchronized)
                    if exp2_like and synchronized
                    else float("nan")
                ),
            }

    ratio_best = {
        parameter: classification_map[parameter]
        for parameter in MODAL_PARAMETERS
    }
    parameterization_change_supported = bool(
        ratio_best["natural_frequency"]["category"] != "C_not_practically_identifiable"
        and ratio_best["damping_ratio"]["category"] != "C_not_practically_identifiable"
        and classification_map["effective_mass"]["category"] == "C_not_practically_identifiable"
    )
    summary = {
        "best_causal_pipeline": best_causal_pipeline,
        "best_estimator_by_median_cross_case_score": best_estimator,
        "estimator_scores": estimator_scores,
        "best_ols_target_band_pipeline": {
            key: best_ols[key]
            for key in (
                "target",
                "frequency_band",
                "pipeline",
                "stiffness_p95_abs_relative_error",
                "damping_p95_abs_relative_error",
                "effective_mass_p95_abs_relative_error",
            )
        },
        "timing_profile_comparison": timing_profile_summary,
        "parameterization_change_supported": parameterization_change_supported,
        "parameterization_recommendation": (
            "adopt_modal_scale_parameterization"
            if parameterization_change_supported
            else "retain_direct_parameters_and_modal_diagnostics"
        ),
        "stage1_gate": gate_details,
    }

    expected_seed_set = set(range(3101, 3121))
    acceptance = {
        "complete_timing_matrix": len(timing_rows) == expected_timing,
        "complete_timing_profile_matrix": len(profile_rows) == expected_profiles,
        "complete_identification_matrix": len(identification_rows) == expected_identification,
        "new_untouched_validation_seed_partition": set(seeds) == expected_seed_set,
        "monte_carlo_seed_count_sufficient": len(seeds) >= int(config["integrity_acceptance"]["minimum_monte_carlo_seeds"]),
        "chirp_only": str(config["probe"]["type"]) == "chirp",
        "all_estimators_present": {row["estimator"] for row in identification_rows} == set(estimators),
        "all_pipelines_present": {row["pipeline"] for row in identification_rows} == set(pipelines),
        "sensorless_force_excluded": "sensorless" not in " ".join(pipelines).lower(),
        "all_frequency_bands_respect_safety_limits": not safety_events,
        "representative_raw_saved": bool(representative_parts),
    }
    metrics = {
        "timing_trial_count": len(timing_rows),
        "timing_profile_trial_count": len(profile_rows),
        "identification_trial_count": len(identification_rows),
        "validation_seed_count": len(seeds),
        "target_count": len(targets),
        "frequency_band_count": len(bands),
        "pipeline_count": len(pipelines),
        "estimator_count": len(estimators),
        "safety_event_count": len(safety_events),
        "stage1_ready_candidate_count": int(gate_details["passing_candidate_count"]),
    }
    representative_raw = {
        field: np.concatenate(parts) for field, parts in representative_parts.items()
    }
    success = all(acceptance.values())
    return CausalEIVIdentifiabilityResult(
        timing_trials=tuple(timing_rows),
        timing_aggregate=timing_aggregate,
        timing_profile_trials=tuple(profile_rows),
        timing_profile_aggregate=profile_aggregate,
        identification_trials=tuple(identification_rows),
        identification_aggregate=identification_aggregate,
        parameter_classifications=classifications,
        timing_thresholds=thresholds,
        representative_raw=representative_raw,
        summary=summary,
        metrics=metrics,
        safety_events=tuple(safety_events),
        acceptance_checks=acceptance,
        success=success,
        stage1_decision=stage_decision,
    )
