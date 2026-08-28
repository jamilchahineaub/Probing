"""EXP-0001: analytical validation of the Kelvin-Voigt evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from probeing.metrics import kelvin_voigt_validation_metrics
from probeing.models import KelvinVoigtModel, KelvinVoigtParameters
from probeing.probing import raised_cosine_ramp


@dataclass(frozen=True)
class ReducedExperimentResult:
    raw: Mapping[str, NDArray[np.float64]]
    metrics: Mapping[str, float | int]
    safety_events: tuple[Mapping[str, Any], ...]
    acceptance_checks: Mapping[str, bool]
    success: bool


def _safety_event(
    *, metric: str, observed: float, limit: float, units: str
) -> Mapping[str, Any]:
    return {
        "type": "limit_exceeded",
        "metric": metric,
        "observed": observed,
        "limit": limit,
        "units": units,
    }


def run_kelvin_voigt_validation(
    config: Mapping[str, Any],
) -> ReducedExperimentResult:
    """Run the deterministic reduced-model validation defined by ``config``."""

    experiment = config["experiment"]
    if experiment["id"] != "EXP-0001":
        raise ValueError("this runner only implements EXP-0001")

    target = config["target"]
    probe = config["probe"]
    parameters = KelvinVoigtParameters(
        stiffness_n_per_m=float(target["stiffness_n_per_m"]),
        damping_n_s_per_m=float(target["damping_n_s_per_m"]),
    )
    model = KelvinVoigtModel(parameters, contact_mode=target["contact_mode"])
    trajectory = raised_cosine_ramp(
        amplitude_m=float(probe["amplitude_m"]),
        rise_duration_s=float(probe["rise_duration_s"]),
        hold_duration_s=float(probe["hold_duration_s"]),
        sample_period_s=float(probe["sample_period_s"]),
    )
    response = model.response(
        trajectory.displacement_m, trajectory.velocity_m_per_s
    )

    analytical_force = (
        parameters.stiffness_n_per_m * trajectory.displacement_m
        + parameters.damping_n_s_per_m * trajectory.velocity_m_per_s
    )
    metrics = dict(
        kelvin_voigt_validation_metrics(
            time_s=trajectory.time_s,
            displacement_m=trajectory.displacement_m,
            velocity_m_per_s=trajectory.velocity_m_per_s,
            force_n=response.force_n,
            analytical_force_n=analytical_force,
            stiffness_n_per_m=parameters.stiffness_n_per_m,
            damping_n_s_per_m=parameters.damping_n_s_per_m,
        )
    )
    displacement_overshoot = max(
        0.0,
        float(metrics["peak_displacement_m"]) - float(probe["amplitude_m"]),
    )
    metrics["displacement_overshoot_m"] = displacement_overshoot

    safety = config["safety_limits"]
    safety_events: list[Mapping[str, Any]] = []
    safety_observations = (
        (
            "peak_abs_force_n",
            float(metrics["peak_abs_force_n"]),
            float(safety["max_abs_force_n"]),
            "N",
        ),
        (
            "peak_displacement_m",
            float(metrics["peak_displacement_m"]),
            float(safety["max_displacement_m"]),
            "m",
        ),
        (
            "peak_abs_velocity_m_per_s",
            float(metrics["peak_abs_velocity_m_per_s"]),
            float(safety["max_abs_velocity_m_per_s"]),
            "m/s",
        ),
    )
    for metric_name, observed, limit, units in safety_observations:
        if observed > limit:
            safety_events.append(
                _safety_event(
                    metric=metric_name, observed=observed, limit=limit, units=units
                )
            )

    acceptance = config["acceptance"]
    checks = {
        "force_rmse_within_tolerance": float(metrics["force_rmse_n"])
        <= float(acceptance["max_force_rmse_n"]),
        "force_max_error_within_tolerance": float(metrics["max_force_error_n"])
        <= float(acceptance["max_force_error_n"]),
        "energy_balance_within_tolerance": float(
            metrics["relative_energy_balance_error"]
        )
        <= float(acceptance["max_relative_energy_balance_error"]),
        "displacement_bound_respected": displacement_overshoot
        <= float(acceptance["max_displacement_overshoot_m"]),
        "safety_limits_respected": (
            not bool(safety_events)
            if bool(acceptance["require_no_safety_violations"])
            else True
        ),
    }

    raw = {
        "time_s": trajectory.time_s,
        "displacement_m": trajectory.displacement_m,
        "velocity_m_per_s": trajectory.velocity_m_per_s,
        "acceleration_m_per_s2": trajectory.acceleration_m_per_s2,
        "elastic_force_n": response.elastic_force_n,
        "damping_force_n": response.damping_force_n,
        "raw_force_n": response.raw_force_n,
        "force_n": response.force_n,
        "analytical_force_n": analytical_force,
        "contact_active": response.contact_active.astype(float),
    }
    return ReducedExperimentResult(
        raw=raw,
        metrics=metrics,
        safety_events=tuple(safety_events),
        acceptance_checks=checks,
        success=all(checks.values()),
    )

