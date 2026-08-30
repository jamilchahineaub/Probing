"""EXP-0007: validation-only replication of the locked EXP-0006 policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .decision_sufficiency import (
    _actual_severity,
    _case_simulation,
    _classification_metrics,
    _measurement_seed,
    _noise,
    evaluate_future_response,
    generate_target_cases,
    simulate_population,
)
from .passive_ringdown import (
    FEATURE_SETS,
    OutcomeModels,
    _attach_actual,
    _extract_partition_features,
    _predict,
    _prediction_row,
    _extended_probe_signal,
)
from ..measurements import process_causal_sensing
from .decision_sufficiency import RidgeModel


RISK_LABELS = ("SAFE", "CAUTION", "UNSAFE")
OUTCOMES = (
    "peak_displacement_m",
    "peak_velocity_m_per_s",
    "late_hold_oscillation_rms_m",
    "hold_settling_time_s",
)


@dataclass(frozen=True)
class LockedReplicationResult:
    rows: tuple[Mapping[str, Any], ...]
    binary_summary: tuple[Mapping[str, Any], ...]
    secondary_summary: tuple[Mapping[str, Any], ...]
    confidence_summary: tuple[Mapping[str, Any], ...]
    boundary_summary: tuple[Mapping[str, Any], ...]
    quantitative_summary: tuple[Mapping[str, Any], ...]
    comparison_summary: tuple[Mapping[str, Any], ...]
    margin_summary: tuple[Mapping[str, Any], ...]
    parameter_rows: tuple[Mapping[str, Any], ...]
    representative_raw: Mapping[str, NDArray[Any]]
    false_safe_raw: Mapping[str, NDArray[Any]]
    boundary_cases: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    acceptance_checks: Mapping[str, bool]
    safety_events: tuple[Mapping[str, Any], ...]
    success: bool
    stage1_decision: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inclusive_seeds(spec: Mapping[str, Any]) -> tuple[int, ...]:
    first, last = int(spec["first"]), int(spec["last"])
    if first > last:
        raise ValueError("seed range must be increasing")
    return tuple(range(first, last + 1))


def _load_locked_policy(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[Mapping[str, Any], OutcomeModels, Mapping[str, Any]]:
    lock = config["locked_policy"]
    bundle_path = repository_root / str(lock["bundle_path"])
    if _sha256(bundle_path) != str(lock["expected_bundle_sha256"]):
        raise RuntimeError("locked EXP-0006 policy bundle hash mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    source_config = repository_root / str(bundle["locked_config_path"])
    source_module = repository_root / "src/probeing/experiments/passive_ringdown.py"
    expected_reference = config["frozen_references"]["exp_0006"]
    if _sha256(source_config) != str(expected_reference["expected_config_sha256"]):
        raise RuntimeError("EXP-0006 source configuration changed")
    if _sha256(source_module) != str(expected_reference["expected_policy_source_sha256"]):
        raise RuntimeError("EXP-0006 policy implementation changed")
    import yaml

    locked_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    locked_config["passive_observation"]["windows_s"] = [0.5]
    if bundle["policy_id"] != str(lock["policy_id"]):
        raise RuntimeError("locked policy identifier mismatch")
    if bundle["feature_set"] != str(lock["source_feature_set"]):
        raise RuntimeError("locked feature set mismatch")
    feature_names = tuple(str(value) for value in bundle["feature_names"])
    if feature_names != tuple(FEATURE_SETS["chirp_ringdown"]):
        raise RuntimeError("locked feature definitions do not match EXP-0006")
    if float(bundle["observation_duration_s"]) != float(lock["observation_duration_s"]):
        raise RuntimeError("locked observation duration mismatch")
    normalization = bundle["normalization"]
    mean = np.asarray(normalization["mean"], dtype=float)
    scale = np.asarray(normalization["scale"], dtype=float)
    if mean.size != len(feature_names) or scale.size != len(feature_names):
        raise RuntimeError("locked normalization dimension mismatch")
    regressors: dict[str, RidgeModel] = {}
    upper: dict[str, float] = {}
    for outcome, values in bundle["regressors"].items():
        regressors[str(outcome)] = RidgeModel(
            mean=mean.copy(),
            scale=scale.copy(),
            coefficients=np.asarray(values["coefficients"], dtype=float),
        )
        upper[str(outcome)] = float(values["upper_residual"])
    model = OutcomeModels(regressors=regressors, upper_residuals=upper)
    return locked_config, model, bundle


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [
        math.lgamma(n + 1.0)
        - math.lgamma(j + 1.0)
        - math.lgamma(n - j + 1.0)
        + j * math.log(p)
        + (n - j) * math.log1p(-p)
        for j in range(k + 1)
    ]
    maximum = max(logs)
    return float(min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)))


def _cp_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    if k <= 0:
        lower = 0.0
    else:
        lo, hi = 0.0, float(k) / n
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            survival = 1.0 - _binomial_cdf(k - 1, n, mid)
            if survival > alpha / 2.0:
                hi = mid
            else:
                lo = mid
        lower = hi
    if k >= n:
        upper = 1.0
    else:
        lo, hi = float(k) / n, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _binomial_cdf(k, n, mid) > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = hi
    return float(lower), float(upper)


def _one_sided_upper(k: int, n: int, confidence: float) -> float:
    if n <= 0 or k >= n:
        return 1.0
    alpha = 1.0 - confidence
    if k == 0:
        return float(1.0 - alpha ** (1.0 / n))
    lo, hi = float(k) / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _binary_metrics(rows: Sequence[Mapping[str, Any]], *, stratum: str) -> Mapping[str, Any]:
    actual_safe = np.asarray([str(row["actual_risk_class"]) == "SAFE" for row in rows], dtype=bool)
    predicted_safe = np.asarray([str(row["predicted_risk_class"]) == "SAFE" for row in rows], dtype=bool)
    safe_count = int(np.sum(actual_safe))
    non_safe_count = int(np.sum(~actual_safe))
    true_safe = int(np.sum(actual_safe & predicted_safe))
    false_positive = int(np.sum(actual_safe & ~predicted_safe))
    false_safe = int(np.sum(~actual_safe & predicted_safe))
    true_non_safe = int(np.sum(~actual_safe & ~predicted_safe))
    rate = false_safe / max(non_safe_count, 1)
    lower, upper = _cp_interval(false_safe, non_safe_count, 0.95)
    one_sided = _one_sided_upper(false_safe, non_safe_count, 0.95)
    return {
        "stratum": stratum,
        "trial_count": len(rows),
        "safe_count": safe_count,
        "non_safe_count": non_safe_count,
        "true_safe_count": true_safe,
        "true_non_safe_count": true_non_safe,
        "false_safe_count": false_safe,
        "false_positive_count": false_positive,
        "false_negative_count": false_safe,
        "false_safe_rate": float(rate),
        "false_safe_ci95_lower": lower,
        "false_safe_ci95_upper": upper,
        "false_safe_one_sided95_upper": one_sided,
        "binary_accuracy": float(np.mean(actual_safe == predicted_safe)),
        "safe_precision": float(true_safe / max(true_safe + false_safe, 1)),
        "non_safe_recall": float(true_non_safe / max(non_safe_count, 1)),
        "sensitivity_non_safe": float(true_non_safe / max(non_safe_count, 1)),
        "specificity_safe": float(true_safe / max(safe_count, 1)),
        "false_positive_rate": float(false_positive / max(safe_count, 1)),
        "false_negative_rate": float(false_safe / max(non_safe_count, 1)),
    }


def _summary_for_strata(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for noise in ("low", "nominal", "high"):
        selected = [row for row in rows if row["noise_regime"] == noise]
        for stratum, subset in (
            ("overall", selected),
            ("broad", [row for row in selected if row["population_stratum"] == "broad"]),
            ("boundary", [row for row in selected if row["population_stratum"] == "boundary"]),
        ):
            if subset:
                output.append({"noise_regime": noise, **_binary_metrics(subset, stratum=stratum)})
    return tuple(output)


def _confidence_rows(binary_summary: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "noise_regime": row["noise_regime"],
            "stratum": row["stratum"],
            "false_safe_count": row["false_safe_count"],
            "non_safe_count": row["non_safe_count"],
            "confidence_level": 0.95,
            "two_sided_lower": row["false_safe_ci95_lower"],
            "two_sided_upper": row["false_safe_ci95_upper"],
            "one_sided_upper": row["false_safe_one_sided95_upper"],
        }
        for row in binary_summary
    )


def _secondary(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for noise in ("low", "nominal", "high"):
        selected = [row for row in rows if row["noise_regime"] == noise]
        if selected:
            output.append({"noise_regime": noise, **_classification_metrics(selected, False)})
    return tuple(output)


def _quantitative(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for noise in ("low", "nominal", "high"):
        selected = [row for row in rows if row["noise_regime"] == noise]
        for outcome in OUTCOMES:
            actual = np.asarray([row[f"actual_{outcome}"] for row in selected], dtype=float)
            predicted = np.asarray([row[f"predicted_{outcome}"] for row in selected], dtype=float)
            upper = np.asarray([row[f"upper_{outcome}"] for row in selected], dtype=float)
            absolute = np.abs(predicted - actual)
            relative = absolute / np.maximum(np.abs(actual), 1.0e-9)
            output.append(
                {
                    "noise_regime": noise,
                    "outcome": outcome,
                    "trial_count": len(selected),
                    "median_absolute_error": float(np.median(absolute)),
                    "p95_absolute_error": float(np.quantile(absolute, 0.95)),
                    "median_abs_relative_error": float(np.median(relative)),
                    "p95_abs_relative_error": float(np.quantile(relative, 0.95)),
                    "upper_bound_coverage": float(np.mean(actual <= upper)),
                }
            )
    return tuple(output)


def _margin_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    edges = (0.0, 0.25, 0.5, 0.75, 1.0, float("inf"))
    output: list[Mapping[str, Any]] = []
    for noise in ("low", "nominal", "high"):
        for low, high in zip(edges[:-1], edges[1:]):
            selected = [
                row for row in rows
                if row["noise_regime"] == noise and low <= float(row["decision_margin"]) < high
            ]
            if selected:
                output.append({
                    "noise_regime": noise,
                    "margin_low": low,
                    "margin_high": high,
                    "trial_count": len(selected),
                    "binary_accuracy": float(np.mean([
                        (row["actual_risk_class"] == "SAFE") == (row["predicted_risk_class"] == "SAFE")
                        for row in selected
                    ])),
                    "false_safe_rate": float(np.mean([
                        row["actual_risk_class"] != "SAFE" and row["predicted_risk_class"] == "SAFE"
                        for row in selected
                    ])),
                })
    return tuple(output)


def _boundary_selection(
    candidates: Sequence[Any], config: Mapping[str, Any]
) -> tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    outcomes = evaluate_future_response(
        simulate_population(
            candidates,
            *_maneuver_signal_for_config(config),
            contact_mode="unilateral",
        ),
        candidates,
        config,
    )
    by_id = {str(row["target_id"]): row for row in outcomes}
    ranked: dict[str, list[Any]] = {"SAFE": [], "NON_SAFE": []}
    for case in candidates:
        outcome = by_id[case.target_id]
        group = "SAFE" if outcome["risk_class"] == "SAFE" else "NON_SAFE"
        severity = max(float(_actual_severity(outcome, config)), 1.0e-12)
        ranked[group].append((abs(math.log(severity)), case, outcome, severity))
    for group in ranked:
        ranked[group].sort(key=lambda value: (value[0], value[1].seed, value[1].case_index))
    settings = config["replication_population"]["boundary_enrichment"]
    selected: list[Any] = []
    records: list[Mapping[str, Any]] = []
    for group, count in (("SAFE", int(settings["safe_count"])), ("NON_SAFE", int(settings["non_safe_count"]))):
        if len(ranked[group]) < count:
            raise RuntimeError(f"boundary candidate pool lacks {group} cases")
        for rank, (distance, case, outcome, severity) in enumerate(ranked[group][:count]):
            selected_case = replace(
                case,
                target_id=f"boundary_s{case.seed}_c{case.case_index:02d}",
                partition="boundary",
            )
            selected.append(selected_case)
            records.append({
                "target_id": selected_case.target_id,
                "source_seed": case.seed,
                "source_case_index": case.case_index,
                "selected_group": group,
                "selection_rank": rank,
                "true_severity_ratio": severity,
                "absolute_log_distance_to_safe_boundary": distance,
                "actual_risk_class": outcome["risk_class"],
            })
    return tuple(selected), tuple(records)


def _maneuver_signal_for_config(config: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def _raw_for_rows(
    cases: Sequence[Any], rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], noise_name: str,
    *, maximum_cases: int, label: str,
) -> Mapping[str, NDArray[Any]]:
    selected_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        target_id = str(row["target_id"])
        if target_id not in seen:
            selected_ids.append(target_id)
            seen.add(target_id)
        if len(selected_ids) >= maximum_cases:
            break
    selected = [case for case in cases if case.target_id in set(selected_ids)]
    if not selected:
        selected = list(cases[:maximum_cases])
    time, command = _extended_probe_signal(config)
    population = simulate_population(selected, time, command, contact_mode="unilateral")
    regimes = {str(item["name"]): (index, item) for index, item in enumerate(config["sensing"]["noise_regimes"])}
    regime_index, regime = regimes[noise_name]
    outcomes = evaluate_future_response(
        simulate_population(selected, *_maneuver_signal_for_config(config), contact_mode="unilateral"),
        selected,
        config,
    )
    outcome_map = {row["target_id"]: row for row in outcomes}
    pred_map = {row["target_id"]: row for row in rows}
    chunks: dict[str, list[NDArray[Any]]] = {}
    for index, case in enumerate(selected):
        truth = _case_simulation(population, case, index, "unilateral")
        sensing = process_causal_sensing(
            truth,
            pipeline=str(config["sensing"]["primary_pipeline"]),
            sample_rate_hz=float(config["sensing"]["sample_rate_hz"]),
            noise=_noise(config, float(regime["multiplier"])),
            pipeline_settings=config["sensing"]["pipeline_settings"],
            random_seed=_measurement_seed(case, regime_index),
            timestamp_offsets_s={"displacement": 0.0, "velocity": 0.0, "acceleration": 0.0, "force": 0.0},
        )
        count = sensing.measurements.time_s.size
        prediction = pred_map.get(case.target_id, {"predicted_risk_class": "UNKNOWN", "predicted_binary_class": "UNKNOWN"})
        values = {
            "target_id": np.full(count, case.target_id),
            "audit_label": np.full(count, label),
            "actual_risk_class": np.full(count, outcome_map[case.target_id]["risk_class"]),
            "predicted_risk_class": np.full(count, prediction["predicted_risk_class"]),
            "predicted_binary_class": np.full(count, prediction.get("predicted_binary_class", "UNKNOWN")),
            "time_s": sensing.measurements.time_s,
            "true_displacement_m": sensing.true_displacement_m,
            "measured_displacement_m": sensing.measurements.displacement_m,
            "true_velocity_m_per_s": sensing.true_velocity_m_per_s,
            "estimated_velocity_m_per_s": sensing.measurements.velocity_m_per_s,
            "true_force_n": sensing.true_contact_force_n,
            "measured_force_n": sensing.measurements.contact_force_n,
            "commanded_force_n": sensing.commanded_force_n,
        }
        for name, array in values.items():
            chunks.setdefault(name, []).append(np.asarray(array))
    return {name: np.concatenate(values) for name, values in chunks.items()}


def _add_binary_fields(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for row in rows:
        value = dict(row)
        actual_binary = "SAFE" if row["actual_risk_class"] == "SAFE" else "NON_SAFE"
        predicted_binary = "SAFE" if row["predicted_risk_class"] == "SAFE" else "NON_SAFE"
        value.update(
            {
                "population_stratum": "boundary" if str(row["target_id"]).startswith("boundary_") else "broad",
                "actual_binary_class": actual_binary,
                "predicted_binary_class": predicted_binary,
                "binary_correct": actual_binary == predicted_binary,
                "false_safe": actual_binary == "NON_SAFE" and predicted_binary == "SAFE",
                "false_positive": actual_binary == "SAFE" and predicted_binary == "NON_SAFE",
            }
        )
        output.append(value)
    return tuple(output)


def run_locked_policy_replication(
    config: Mapping[str, Any],
    *, repository_root: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> LockedReplicationResult:
    """Evaluate a serialized EXP-0006 policy; never fits on EXP-0007 data."""

    locked, model, bundle = _load_locked_policy(config, repository_root)
    broad_seeds = _inclusive_seeds(config["replication_population"]["broad_seed_range"])
    boundary_seeds = _inclusive_seeds(config["replication_population"]["boundary_candidate_seed_range"])
    count = int(config["replication_population"]["cases_per_seed"])
    bounds = config["replication_population"]["target_bounds"]
    broad = generate_target_cases(broad_seeds, count, bounds, partition="broad")
    candidates = generate_target_cases(boundary_seeds, count, bounds, partition="boundary_candidate")
    boundary, boundary_records = _boundary_selection(candidates, config)
    cases = tuple((*broad, *boundary))
    if len(set(case.target_id for case in cases)) != len(cases):
        raise RuntimeError("replication target identifiers are not unique")
    # This is the only sensing pass. It creates regressors/features, never outcomes.
    state = [0]
    features, safety_events, _ = _extract_partition_features(
        cases,
        locked,
        progress_callback=progress_callback,
        progress_state=state,
        progress_total=3 * len(cases),
    )
    pending = []
    names = FEATURE_SETS["chirp_ringdown"]
    for feature in features:
        prediction = _predict(model, feature, names, locked)
        pending.append(_prediction_row(feature, "chirp_ringdown", prediction, locked))
    # Critical replication separation: labels are generated only after all predictions.
    outcomes = evaluate_future_response(
        simulate_population(cases, *_maneuver_signal_for_config(locked), contact_mode="unilateral"),
        cases,
        locked,
    )
    rows = _add_binary_fields(_attach_actual(pending, outcomes, locked))
    binary_summary = _summary_for_strata(rows)
    confidence = _confidence_rows(binary_summary)
    secondary = _secondary(rows)
    quantitative = _quantitative(rows)
    margin_summary = _margin_rows(rows)
    by_id_outcome = {row["target_id"]: row for row in outcomes}
    parameter_rows = tuple(
        {
            "target_id": case.target_id,
            "population_stratum": "boundary" if case.partition == "boundary" else "broad",
            "stiffness_n_per_m": case.stiffness_n_per_m,
            "damping_n_s_per_m": case.damping_n_s_per_m,
            "effective_mass_kg": case.effective_mass_kg,
            "actual_risk_class": by_id_outcome[case.target_id]["risk_class"],
            "actual_severity_ratio": _actual_severity(by_id_outcome[case.target_id], locked),
        }
        for case in cases
    )
    boundary_summary = tuple(row for row in binary_summary if row["stratum"] == "boundary")
    comparison: list[Mapping[str, Any]] = []
    references = config["exp_0006_reference_metrics"]
    for noise in ("nominal", "high"):
        current = next(row for row in binary_summary if row["noise_regime"] == noise and row["stratum"] == "overall")
        reference = references[noise]
        comparison.append({
            "experiment": "EXP-0006",
            "noise_regime": noise,
            **reference,
        })
        comparison.append({
            "experiment": "EXP-0007",
            "noise_regime": noise,
            "binary_accuracy": current["binary_accuracy"],
            "false_safe_rate": current["false_safe_rate"],
            "non_safe_recall": current["non_safe_recall"],
            "safe_precision": current["safe_precision"],
        })
    comparison_summary = tuple(comparison)
    high_rows = [row for row in rows if row["noise_regime"] == "high"]
    false_safe_rows = [row for row in high_rows if row["false_safe"]]
    nearest = sorted(
        [row for row in high_rows if row["actual_binary_class"] == "NON_SAFE" and not row["false_safe"]],
        key=lambda row: float(row["predicted_risk_score"]),
    )
    representative_ids = [
        row["target_id"] for row in high_rows if row["actual_risk_class"] != "SAFE" and row["predicted_binary_class"] == "NON_SAFE"
    ][:3]
    representative_rows = [row for row in high_rows if row["target_id"] in representative_ids]
    false_audit_rows = false_safe_rows or nearest[:3]
    representative_raw = _raw_for_rows(cases, representative_rows, locked, "high", maximum_cases=3, label="correctly_rejected")
    false_safe_raw = _raw_for_rows(cases, false_audit_rows, locked, "high", maximum_cases=3, label="false_safe" if false_safe_rows else "nearest_non_safe_rejection")
    overall = {(row["noise_regime"], row["stratum"]): row for row in binary_summary}
    nominal = overall[("nominal", "overall")]
    high = overall[("high", "overall")]
    boundary_nominal = overall[("nominal", "boundary")]
    boundary_high = overall[("high", "boundary")]
    gate = config["stage1_replication_criterion"]
    checks = {
        "locked_policy_hash": _sha256(repository_root / str(config["locked_policy"]["bundle_path"])) == str(config["locked_policy"]["expected_bundle_sha256"]),
        "no_exp_0007_fit_or_calibration": not bool(config["locked_policy"]["fitting_or_calibration_allowed_in_exp_0007"]),
        "predictions_precede_outcome_join": bool(config["integrity_acceptance"]["require_predictions_before_outcome_join"]),
        "minimum_total_cases": len(cases) >= int(config["integrity_acceptance"]["minimum_total_replication_cases"]),
        "minimum_broad_cases": len(broad) >= int(config["integrity_acceptance"]["minimum_broad_cases"]),
        "minimum_boundary_cases": len(boundary) >= int(config["integrity_acceptance"]["minimum_boundary_cases"]),
        "minimum_non_safe_cases": nominal["non_safe_count"] >= int(config["integrity_acceptance"]["minimum_non_safe_cases"]),
        "new_seed_set_disjoint": not (set(broad_seeds) | set(boundary_seeds)) & set(sum((list(range(a, b + 1)) for a, b in config["replication_population"]["prohibited_seed_ranges"].values()), [])),
        "primary_policy_is_causal": locked["sensing"]["primary_pipeline"] == "causal_low_pass" and bool(locked["sensing"]["synchronized"]),
        "zero_probe_safety_events": not safety_events,
        "finite_predictions": all(np.isfinite(float(row["predicted_risk_score"])) for row in rows),
    }
    stage_pass = bool(
        checks["locked_policy_hash"]
        and checks["new_seed_set_disjoint"]
        and checks["zero_probe_safety_events"]
        and nominal["false_safe_rate"] <= float(gate["maximum_nominal_false_safe_rate"])
        and high["false_safe_rate"] <= float(gate["maximum_high_noise_false_safe_rate"])
        and nominal["false_safe_one_sided95_upper"] <= float(gate["maximum_nominal_false_safe_one_sided_95_upper"])
        and high["false_safe_one_sided95_upper"] <= float(gate["maximum_high_noise_false_safe_one_sided_95_upper"])
        and nominal["binary_accuracy"] >= float(gate["minimum_nominal_binary_accuracy"])
        and high["binary_accuracy"] >= float(gate["minimum_high_noise_binary_accuracy"])
        and nominal["safe_precision"] >= float(gate["minimum_nominal_safe_precision"])
        and high["safe_precision"] >= float(gate["minimum_high_noise_safe_precision"])
        and nominal["non_safe_recall"] >= float(gate["minimum_nominal_non_safe_recall"])
        and high["non_safe_recall"] >= float(gate["minimum_high_noise_non_safe_recall"])
        and float(references["nominal"]["binary_accuracy"]) - nominal["binary_accuracy"] <= float(gate["maximum_binary_accuracy_degradation_from_exp_0006"])
        and float(references["high"]["binary_accuracy"]) - high["binary_accuracy"] <= float(gate["maximum_binary_accuracy_degradation_from_exp_0006"])
        and boundary_nominal["false_safe_rate"] <= float(gate["maximum_boundary_false_safe_rate"])
        and boundary_high["false_safe_rate"] <= float(gate["maximum_boundary_false_safe_rate"])
        and boundary_nominal["false_safe_one_sided95_upper"] <= float(gate["maximum_boundary_false_safe_one_sided_95_upper"])
        and boundary_high["false_safe_one_sided95_upper"] <= float(gate["maximum_boundary_false_safe_one_sided_95_upper"])
        and boundary_nominal["binary_accuracy"] >= float(gate["minimum_boundary_binary_accuracy"])
        and boundary_high["binary_accuracy"] >= float(gate["minimum_boundary_binary_accuracy"])
    )
    summary = {
        "policy_id": bundle["policy_id"],
        "validation_case_count": len(cases),
        "broad_case_count": len(broad),
        "boundary_case_count": len(boundary),
        "boundary_candidate_count": len(candidates),
        "boundary_selection_used_policy": False,
        "predictions_created_before_hidden_outcomes": True,
        "false_safe_cases_high_noise": len(false_safe_rows),
        "false_safe_cases_any_noise": int(sum(row["false_safe"] for row in rows)),
        "stage1_replication_criterion_pass": stage_pass,
        "stage1_decision": "PASS_TO_INDEPENDENT_MATLAB_SIMULINK" if stage_pass else "CONTINUE_STAGE_1",
        "exp_0006_reference_run_id": config["frozen_references"]["exp_0006"]["run_id"],
    }
    metrics = {
        "validation_case_count": len(cases),
        "prediction_count": len(rows),
        "stage1_gate_pass": stage_pass,
        "overall_nominal_false_safe_rate": nominal["false_safe_rate"],
        "overall_high_false_safe_rate": high["false_safe_rate"],
        "overall_nominal_false_safe_one_sided95_upper": nominal["false_safe_one_sided95_upper"],
        "overall_high_false_safe_one_sided95_upper": high["false_safe_one_sided95_upper"],
        "overall_nominal_binary_accuracy": nominal["binary_accuracy"],
        "overall_high_binary_accuracy": high["binary_accuracy"],
        "boundary_nominal_false_safe_rate": boundary_nominal["false_safe_rate"],
        "boundary_high_false_safe_rate": boundary_high["false_safe_rate"],
        "safety_event_count": len(safety_events),
        "false_safe_case_count": int(sum(row["false_safe"] for row in rows)),
    }
    return LockedReplicationResult(
        rows=rows,
        binary_summary=binary_summary,
        secondary_summary=secondary,
        confidence_summary=confidence,
        boundary_summary=boundary_summary,
        quantitative_summary=quantitative,
        comparison_summary=comparison_summary,
        margin_summary=margin_summary,
        parameter_rows=parameter_rows,
        representative_raw=representative_raw,
        false_safe_raw=false_safe_raw,
        boundary_cases=boundary_records,
        summary=summary,
        metrics=metrics,
        acceptance_checks=checks,
        safety_events=tuple(safety_events),
        success=bool(stage_pass and all(checks.values())),
        stage1_decision=str(summary["stage1_decision"]),
    )
