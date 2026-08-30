"""Metrics and aggregation for realistic Stage 1 identifiability tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from probeing.estimators import RegressionDiagnostics
from probeing.models import ContactInteractionSimulation, InteractionParameters


PARAMETER_LABELS = ("stiffness", "damping", "effective_mass")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def physical_disturbance_metrics(
    simulation: ContactInteractionSimulation,
) -> Mapping[str, float | int]:
    """Summarize force, work, motion, and contact loss from simulation truth."""

    response = simulation.response
    time = response.time_s
    force = simulation.contact_force_n
    velocity = response.velocity_m_per_s
    power = force * velocity
    return {
        "peak_force_n": float(np.max(np.abs(force))),
        "net_impulse_n_s": float(np.trapz(force, time)),
        "absolute_impulse_n_s": float(np.trapz(np.abs(force), time)),
        "net_input_work_j": float(np.trapz(power, time)),
        "absolute_input_energy_j": float(np.trapz(np.abs(power), time)),
        "peak_target_displacement_m": float(
            np.max(np.abs(response.displacement_m))
        ),
        "rms_target_displacement_m": rms(response.displacement_m),
        "peak_target_velocity_m_per_s": float(
            np.max(np.abs(response.velocity_m_per_s))
        ),
        "peak_target_acceleration_m_per_s2": float(
            np.max(np.abs(response.acceleration_m_per_s2))
        ),
        "contact_active_fraction": float(np.mean(simulation.contact_active)),
        "contact_loss_count": simulation.contact_loss_count,
    }


def probe_spectrum_metrics(
    time_s: np.ndarray,
    force_n: np.ndarray,
    *,
    natural_frequency_hz: float,
) -> Mapping[str, float]:
    """Return simple frequency-content diagnostics, explicitly including DC."""

    time = np.asarray(time_s, dtype=float)
    force = np.asarray(force_n, dtype=float)
    step = float(time[1] - time[0])
    spectrum = np.fft.rfft(force)
    frequencies = np.fft.rfftfreq(force.size, step)
    energy = np.abs(spectrum) ** 2
    total = max(float(np.sum(energy)), np.finfo(float).tiny)
    non_dc = energy.copy()
    non_dc[0] = 0.0
    non_dc_total = max(float(np.sum(non_dc)), np.finfo(float).tiny)
    centroid = float(np.sum(frequencies * non_dc) / non_dc_total)
    high_frequency_fraction = float(
        np.sum(energy[frequencies >= natural_frequency_hz]) / total
    )
    return {
        "force_dc_energy_fraction": float(energy[0] / total),
        "force_spectral_centroid_hz": centroid,
        "force_energy_above_natural_frequency_fraction": high_frequency_fraction,
    }


def normalized_information_metrics(
    diagnostics: RegressionDiagnostics,
    *,
    peak_displacement_m: float,
    absolute_input_energy_j: float,
    relative_estimation_error_rms: float,
    displacement_reference_m: float,
    energy_reference_j: float,
) -> Mapping[str, float]:
    """Heuristic dimensionless information/disturbance scores.

    These metrics are comparison aids, not claims of an optimal information
    measure. The information score is the geometric mean of the singular values
    of the column-normalized regression matrix.
    """

    singular_values = np.maximum(
        np.asarray(diagnostics.normalized_singular_values, dtype=float), 0.0
    )
    information_score = float(np.prod(singular_values) ** (1.0 / 3.0))
    normalized_peak = max(
        peak_displacement_m / displacement_reference_m, np.finfo(float).eps
    )
    normalized_energy = max(
        absolute_input_energy_j / energy_reference_j, np.finfo(float).eps
    )
    return {
        "normalized_information_score": information_score,
        "information_per_peak_displacement": information_score / normalized_peak,
        "information_per_input_energy": information_score / normalized_energy,
        "estimation_error_per_peak_displacement": relative_estimation_error_rms
        / normalized_peak,
        "estimation_error_per_input_energy": relative_estimation_error_rms
        / normalized_energy,
    }


def true_modal_parameters(parameters: InteractionParameters) -> tuple[float, float]:
    natural_frequency = float(
        np.sqrt(parameters.stiffness_n_per_m / parameters.effective_mass_kg)
    )
    damping_ratio = float(
        parameters.damping_n_s_per_m
        / (
            2.0
            * np.sqrt(
                parameters.stiffness_n_per_m * parameters.effective_mass_kg
            )
        )
    )
    return natural_frequency, damping_ratio


def aggregate_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Aggregate Monte Carlo bias, RMSE, tail error, observability, and disturbance."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    aggregate: list[Mapping[str, Any]] = []
    for group_values, rows in grouped.items():
        output: dict[str, Any] = dict(zip(group_keys, group_values))
        output["trial_count"] = len(rows)
        for label in PARAMETER_LABELS:
            errors = np.asarray([row[f"{label}_error"] for row in rows], dtype=float)
            relative = np.asarray(
                [row[f"{label}_relative_error"] for row in rows], dtype=float
            )
            eiv = np.asarray(
                [row[f"{label}_eiv_relative_shift"] for row in rows], dtype=float
            )
            output[f"{label}_bias"] = float(np.mean(errors))
            output[f"{label}_rmse"] = rms(errors)
            output[f"{label}_relative_bias"] = float(np.mean(relative))
            output[f"{label}_relative_rmse"] = rms(relative)
            output[f"{label}_p95_abs_relative_error"] = float(
                np.percentile(np.abs(relative), 95.0)
            )
            output[f"{label}_worst_abs_relative_error"] = float(
                np.max(np.abs(relative))
            )
            output[f"{label}_eiv_relative_bias"] = float(np.mean(eiv))
            output[f"{label}_eiv_relative_rmse"] = rms(eiv)

        combined = np.asarray(
            [row["parameter_relative_error_rms"] for row in rows], dtype=float
        )
        output["parameter_error_median"] = float(np.median(combined))
        output["parameter_error_p95"] = float(np.percentile(combined, 95.0))
        output["parameter_error_worst"] = float(np.max(combined))
        output["rank_minimum"] = int(min(int(row["rank"]) for row in rows))
        output["rank_deficient_fraction"] = float(
            np.mean([int(row["rank"]) < 3 for row in rows])
        )
        for index in range(3):
            singular = np.asarray(
                [row[f"normalized_singular_value_{index + 1}"] for row in rows],
                dtype=float,
            )
            output[f"normalized_singular_value_{index + 1}_median"] = float(
                np.median(singular)
            )
        condition = np.asarray(
            [row["normalized_condition_number"] for row in rows], dtype=float
        )
        correlation = np.asarray(
            [row["maximum_abs_parameter_correlation"] for row in rows], dtype=float
        )
        output["condition_number_median"] = float(np.median(condition))
        output["condition_number_p95"] = float(np.percentile(condition, 95.0))
        output["parameter_correlation_median"] = float(np.median(correlation))
        output["parameter_correlation_p95"] = float(
            np.percentile(correlation, 95.0)
        )

        converged = [row for row in rows if bool(row["rls_converged"])]
        output["rls_convergence_fraction"] = len(converged) / len(rows)
        if converged:
            convergence = np.asarray(
                [row["rls_convergence_time_s"] for row in converged], dtype=float
            )
            output["rls_convergence_time_median_s"] = float(np.median(convergence))
            output["rls_convergence_time_p95_s"] = float(
                np.percentile(convergence, 95.0)
            )
        else:
            output["rls_convergence_time_median_s"] = -1.0
            output["rls_convergence_time_p95_s"] = -1.0

        valid_modal = [row for row in rows if bool(row["ratio_estimate_valid"])]
        output["ratio_estimate_valid_fraction"] = len(valid_modal) / len(rows)
        for label in ("natural_frequency", "damping_ratio"):
            if valid_modal:
                values = np.asarray(
                    [row[f"{label}_relative_error"] for row in valid_modal], dtype=float
                )
                output[f"{label}_relative_bias"] = float(np.mean(values))
                output[f"{label}_relative_rmse"] = rms(values)
                output[f"{label}_p95_abs_relative_error"] = float(
                    np.percentile(np.abs(values), 95.0)
                )
            else:
                output[f"{label}_relative_bias"] = -1.0
                output[f"{label}_relative_rmse"] = -1.0
                output[f"{label}_p95_abs_relative_error"] = -1.0

        median_fields = (
            "peak_force_n",
            "absolute_impulse_n_s",
            "absolute_input_energy_j",
            "peak_target_displacement_m",
            "rms_target_displacement_m",
            "peak_target_velocity_m_per_s",
            "peak_target_acceleration_m_per_s2",
            "normalized_information_score",
            "information_per_peak_displacement",
            "information_per_input_energy",
            "estimation_error_per_peak_displacement",
            "estimation_error_per_input_energy",
            "filter_delay_s",
        )
        for field in median_fields:
            output[field] = float(np.median([row[field] for row in rows]))
        output["maximum_peak_force_n"] = float(
            max(row["peak_force_n"] for row in rows)
        )
        output["maximum_peak_target_displacement_m"] = float(
            max(row["peak_target_displacement_m"] for row in rows)
        )
        aggregate.append(output)
    return tuple(aggregate)


def group_rows(
    rows: Iterable[Mapping[str, Any]], **conditions: Any
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(name) == value for name, value in conditions.items())
    ]
