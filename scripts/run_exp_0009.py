#!/usr/bin/env python3
"""Run the frozen EXP-0009 held-out delivery and policy transfer experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from probeing.contact import TargetParameters
from probeing.controllers import StateTransition
from probeing.experiments.coupled_uav_contact import VehicleTrajectory
from probeing.experiments.decision_sufficiency import (
    TargetCase,
    _actual_severity,
    _maneuver_signal,
    evaluate_future_response,
    generate_target_cases,
    simulate_population,
)
from probeing.experiments.hybrid_contact_delivery import (
    HybridTrajectory,
    hybrid_delivery_metrics,
    hybrid_frozen_policy_prediction,
    hybrid_future_outcome,
    simulate_hybrid_contact,
)
from probeing.experiments.locked_policy_replication import _binary_metrics, _load_locked_policy


@dataclass(frozen=True)
class Trial:
    trial_id: str
    stratum: str
    seed: int
    case_index: int
    stiffness_n_per_m: float
    damping_n_s_per_m: float
    effective_mass_kg: float
    reduced_order_boundary_severity: float | None = None

    @property
    def target(self) -> TargetParameters:
        return TargetParameters(
            self.stiffness_n_per_m,
            self.damping_n_s_per_m,
            self.effective_mass_kg,
        )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _save_trajectory(path: Path, trajectory: HybridTrajectory) -> None:
    np.savez_compressed(
        path,
        time_s=trajectory.time_s,
        phase=trajectory.phase,
        phase_elapsed_s=trajectory.phase_elapsed_s,
        probe_variation_reference_n=trajectory.probe_variation_reference_n,
        total_contact_reference_n=trajectory.total_contact_reference_n,
        commanded_normal_force_n=trajectory.commanded_normal_force_n,
        realized_contact_force_n=trajectory.realized_contact_force_n,
        contact_active=trajectory.contact_active,
        contact_penetration_m=trajectory.contact_penetration_m,
        contact_force_world_n=trajectory.contact_force_world_n,
        contact_torque_body_nm=trajectory.contact_torque_body_nm,
        target_displacement_m=trajectory.target_displacement_m,
        target_velocity_m_per_s=trajectory.target_velocity_m_per_s,
        target_acceleration_m_per_s2=trajectory.target_acceleration_m_per_s2,
        near_loss_active=trajectory.near_loss_active,
        desired_tip_coordinate_m=trajectory.desired_tip_coordinate_m,
        vehicle_position_world_m=trajectory.vehicle.position_world_m,
        vehicle_velocity_world_m_per_s=trajectory.vehicle.velocity_world_m_per_s,
        vehicle_quaternion_wxyz=trajectory.vehicle.quaternion_wxyz,
        vehicle_euler_xyz_rad=trajectory.vehicle.euler_xyz_rad,
        vehicle_angular_velocity_body_rad_s=trajectory.vehicle.angular_velocity_body_rad_s,
        rotor_speed_rad_s=trajectory.vehicle.rotor_speed_rad_s,
        rotor_thrust_n=trajectory.vehicle.rotor_thrust_n,
        desired_position_world_m=trajectory.vehicle.desired_position_world_m,
        desired_force_world_n=trajectory.vehicle.desired_force_world_n,
        desired_torque_body_nm=trajectory.vehicle.desired_torque_body_nm,
        actuator_reserve=trajectory.vehicle.actuator_reserve,
        motor_saturated=trajectory.vehicle.motor_saturated,
    )
    transition_path = path.with_suffix(".transitions.json")
    transition_path.write_text(
        json.dumps([asdict(item) for item in trajectory.transitions], indent=2),
        encoding="utf-8",
    )


def _load_trajectory(path: Path) -> HybridTrajectory:
    values = np.load(path, allow_pickle=False)
    transitions = tuple(
        StateTransition(**item)
        for item in json.loads(path.with_suffix(".transitions.json").read_text(encoding="utf-8"))
    )
    vehicle = VehicleTrajectory(
        values["time_s"], values["vehicle_position_world_m"],
        values["vehicle_velocity_world_m_per_s"], values["vehicle_quaternion_wxyz"],
        values["vehicle_euler_xyz_rad"], values["vehicle_angular_velocity_body_rad_s"],
        values["rotor_speed_rad_s"], values["rotor_thrust_n"],
        values["desired_position_world_m"], values["desired_force_world_n"],
        values["desired_torque_body_nm"], values["actuator_reserve"],
        values["motor_saturated"].astype(bool),
    )
    return HybridTrajectory(
        values["time_s"], values["phase"], values["phase_elapsed_s"],
        values["probe_variation_reference_n"], values["total_contact_reference_n"],
        values["commanded_normal_force_n"], values["realized_contact_force_n"],
        values["contact_active"].astype(bool), values["contact_penetration_m"],
        values["contact_force_world_n"], values["contact_torque_body_nm"],
        values["target_displacement_m"], values["target_velocity_m_per_s"],
        values["target_acceleration_m_per_s2"], values["near_loss_active"].astype(bool),
        values["desired_tip_coordinate_m"], vehicle, transitions,
    )


def _boundary_trials(
    config: Mapping[str, Any], locked_config: Mapping[str, Any]
) -> list[Trial]:
    held = config["held_out_population"]
    count = int(held["boundary_cases_per_seed"])
    multiplier = int(held["boundary_candidate_multiplier"])
    bounds = {
        "stiffness_n_per_m": held["stiffness_n_per_m"],
        "damping_n_s_per_m": held["damping_n_s_per_m"],
        "effective_mass_kg": held["effective_mass_kg"],
    }
    output: list[Trial] = []
    maneuver_time, maneuver_force = _maneuver_signal(locked_config)
    for seed in held["boundary_seeds"]:
        cases = generate_target_cases(
            (int(seed),), count * multiplier, bounds,
            partition=f"exp0009_boundary_candidates_s{int(seed)}",
        )
        population = simulate_population(cases, maneuver_time, maneuver_force, contact_mode="unilateral")
        outcomes = evaluate_future_response(population, cases, locked_config)
        ranked = sorted(
            zip(cases, outcomes),
            key=lambda item: abs(np.log(max(_actual_severity(item[1], locked_config), 1e-12))),
        )[:count]
        for index, (case, outcome) in enumerate(ranked):
            severity = _actual_severity(outcome, locked_config)
            output.append(Trial(
                trial_id=f"exp0009_boundary_s{int(seed)}_c{index:02d}",
                stratum="boundary", seed=int(seed), case_index=index,
                stiffness_n_per_m=case.stiffness_n_per_m,
                damping_n_s_per_m=case.damping_n_s_per_m,
                effective_mass_kg=case.effective_mass_kg,
                reduced_order_boundary_severity=float(severity),
            ))
    return output


def _population(config: Mapping[str, Any], locked_config: Mapping[str, Any]) -> list[Trial]:
    held = config["held_out_population"]
    bounds = {
        "stiffness_n_per_m": held["stiffness_n_per_m"],
        "damping_n_s_per_m": held["damping_n_s_per_m"],
        "effective_mass_kg": held["effective_mass_kg"],
    }
    broad = generate_target_cases(
        tuple(int(value) for value in held["broad_seeds"]),
        int(held["broad_cases_per_seed"]), bounds, partition="exp0009_broad",
    )
    output = [
        Trial(
            case.target_id, "broad", case.seed, case.case_index,
            case.stiffness_n_per_m, case.damping_n_s_per_m, case.effective_mass_kg,
        )
        for case in broad
    ]
    output.extend(_boundary_trials(config, locked_config))
    if len(output) != int(held["target_count"]):
        raise RuntimeError("held-out target count does not match prospective configuration")
    return output


def _terminal_reason(trajectory: HybridTrajectory) -> str:
    if trajectory.transitions:
        final = trajectory.transitions[-1]
        if final.to_phase in {"ABORT", "DECISION"}:
            return final.reason
    return str(trajectory.phase[-1])


def _trial_worker(payload: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]) -> Mapping[str, Any]:
    config, locked_config, trial_values, raw_root_value = payload
    trial = Trial(**trial_values)
    raw_root = Path(raw_root_value)
    probe = simulate_hybrid_contact(config, trial.target)
    delivery = dict(hybrid_delivery_metrics(probe, config))
    actual, future = hybrid_future_outcome(config, trial.target, locked_config)
    probe_path = raw_root / f"{trial.trial_id}__probe.npz"
    future_path = raw_root / f"{trial.trial_id}__sustained_contact.npz"
    _save_trajectory(probe_path, probe)
    _save_trajectory(future_path, future)
    row: dict[str, Any] = {
        **asdict(trial),
        "terminal_phase": str(probe.phase[-1]),
        "terminal_reason": _terminal_reason(probe),
        "actual_risk_class": actual["risk_class"],
        "future_terminal_phase": str(future.phase[-1]),
        "future_terminal_reason": _terminal_reason(future),
        **delivery,
        **{f"future_{key}": value for key, value in actual.items() if key != "risk_class"},
        "probe_raw_path": probe_path.relative_to(raw_root.parent.parent).as_posix(),
        "future_raw_path": future_path.relative_to(raw_root.parent.parent).as_posix(),
    }
    result_path = raw_root / f"{trial.trial_id}__result.json"
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def _finite(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _quantile(rows: Sequence[Mapping[str, Any]], key: str, q: float, *, finite_only: bool = False) -> float:
    raw = [row[key] for row in rows]
    values = _finite(raw) if finite_only else np.asarray(raw, dtype=float)
    return float(np.quantile(values, q)) if values.size else float("inf")


def _delivery_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = config["prospective_delivery_gates"]
    complete = [row for row in rows if bool(row["completed"])]
    metrics = {
        "trial_count": len(rows),
        "completed_count": len(complete),
        "completion_fraction": len(complete) / max(len(rows), 1),
        "abort_count": int(sum(bool(row["aborted"]) for row in rows)),
        "zero_separation_fraction_all_trials": float(np.mean([bool(row["zero_separation_during_probe"]) for row in rows])),
        "zero_separation_fraction_completed": float(np.mean([bool(row["zero_separation_during_probe"]) for row in complete])) if complete else 0.0,
        "peak_force_p99_n": _quantile(rows, "peak_contact_force_n", 0.99),
        "absolute_peak_force_n": float(max(float(row["peak_contact_force_n"]) for row in rows)),
        "median_probe_rms_error_n_all_trials": _quantile(rows, "probe_rms_tracking_error_n", 0.5),
        "median_probe_rms_error_n_completed": _quantile(complete, "probe_rms_tracking_error_n", 0.5, finite_only=True),
        "p95_probe_rms_error_n_completed": _quantile(complete, "probe_rms_tracking_error_n", 0.95, finite_only=True),
        "median_normalized_rms_error_completed": _quantile(complete, "probe_normalized_rms_tracking_error", 0.5, finite_only=True),
        "median_absolute_lag_s_completed": _quantile([dict(row, absolute_lag=abs(float(row["probe_cross_correlation_lag_s"]))) for row in complete], "absolute_lag", 0.5, finite_only=True),
        "p95_absolute_lag_s_completed": _quantile([dict(row, absolute_lag=abs(float(row["probe_cross_correlation_lag_s"]))) for row in complete], "absolute_lag", 0.95, finite_only=True),
        "median_faithful_bandwidth_hz_completed": _quantile(complete, "probe_delivery_bandwidth_hz", 0.5, finite_only=True),
        "no_faithful_bin_fraction_completed": float(np.mean([float(row["probe_delivery_bandwidth_hz"]) <= 0.0 for row in complete])) if complete else 1.0,
        "passive_recontact_fraction_all_trials": float(np.mean([bool(row["passive_recontact"]) for row in rows])),
        "maximum_first_contact_peak_force_n": float(max(float(row["first_contact_peak_force_n"]) for row in rows)),
        "maximum_first_contact_impulse_n_s": float(max(float(row["first_contact_impulse_n_s"]) for row in rows)),
        "maximum_recontact_peak_force_n": float(max(float(row["recontact_peak_force_n"]) for row in rows)),
        "maximum_unload_impulse_n_s_completed": float(max((float(row["unload_impulse_n_s"]) for row in complete), default=float("inf"))),
        "maximum_attitude_deg": float(max(float(row["peak_attitude_deg"]) for row in rows)),
        "motor_saturation_sample_fraction": float(np.mean([float(row["motor_saturation_fraction"]) for row in rows])),
        "minimum_actuator_reserve": float(min(float(row["minimum_actuator_reserve"]) for row in rows)),
        "maximum_passive_residual_force_rms_n_completed": float(max((float(row["passive_residual_force_rms_n"]) for row in complete), default=float("inf"))),
        "maximum_passive_force_squared_energy_n2_s_completed": float(max((float(row["passive_force_squared_energy_n2_s"]) for row in complete), default=float("inf"))),
        "maximum_passive_commanded_probe_force_n_completed": float(max((float(row["passive_max_commanded_probe_force_n"]) for row in complete), default=float("inf"))),
    }
    checks = {
        "peak_force_p99": metrics["peak_force_p99_n"] <= float(gates["population_peak_force_p99_n"]),
        "absolute_peak_force": metrics["absolute_peak_force_n"] <= float(gates["absolute_peak_force_n"]),
        "zero_separation": metrics["zero_separation_fraction_all_trials"] >= float(gates["zero_separation_during_probe_fraction"]),
        "median_rms_error": metrics["median_probe_rms_error_n_all_trials"] <= float(gates["median_rms_force_error_n"]),
        "p95_rms_error_completed": metrics["p95_probe_rms_error_n_completed"] <= float(gates["p95_rms_force_error_n"]),
        "normalized_rms_error": metrics["median_normalized_rms_error_completed"] <= float(gates["median_normalized_rms_force_error"]),
        "median_lag": metrics["median_absolute_lag_s_completed"] <= float(gates["median_absolute_lag_s"]),
        "p95_lag": metrics["p95_absolute_lag_s_completed"] <= float(gates["p95_absolute_lag_s"]),
        "faithful_bandwidth": metrics["median_faithful_bandwidth_hz_completed"] >= float(gates["median_faithful_bandwidth_hz"]),
        "no_faithful_bin": metrics["no_faithful_bin_fraction_completed"] <= float(gates["no_faithful_bin_fraction"]),
        "passive_recontact": metrics["passive_recontact_fraction_all_trials"] <= float(gates["passive_recontact_trial_fraction"]),
        "first_contact_peak": metrics["maximum_first_contact_peak_force_n"] <= float(gates["first_contact_peak_force_n"]),
        "first_contact_impulse": metrics["maximum_first_contact_impulse_n_s"] <= float(gates["first_contact_impulse_n_s"]),
        "recontact_peak": metrics["maximum_recontact_peak_force_n"] <= float(gates["recontact_peak_force_n"]),
        "unload_impulse": metrics["maximum_unload_impulse_n_s_completed"] <= float(gates["absolute_unload_impulse_n_s"]),
        "actuator_reserve": metrics["minimum_actuator_reserve"] >= float(gates["minimum_actuator_reserve"]),
        "motor_saturation": metrics["motor_saturation_sample_fraction"] <= float(gates["motor_saturation_fraction"]),
        "attitude": metrics["maximum_attitude_deg"] <= float(gates["maximum_attitude_deg"]),
        "passive_force": metrics["maximum_passive_residual_force_rms_n_completed"] <= float(gates["passive_residual_force_rms_n"]),
        "passive_energy": metrics["maximum_passive_force_squared_energy_n2_s_completed"] <= float(gates["passive_force_squared_energy_n2_s"]),
        "passive_command": metrics["maximum_passive_commanded_probe_force_n_completed"] == 0.0,
    }
    return {**metrics, "checks": checks, "passed": bool(all(checks.values()))}


def _failure_mechanism(row: Mapping[str, Any], locked_config: Mapping[str, Any]) -> str:
    if str(row.get("terminal_phase")) == "ABORT":
        reason = str(row.get("terminal_reason", ""))
        if "acquisition" in reason:
            return "contact_acquisition_failure"
        if "force" in reason:
            return "impact_or_force_limit"
        return f"controller_abort:{reason}"
    if float(row.get("probe_separation_count", 0.0)) > 0.0:
        return "contact_loss_or_reimpact"
    if float(row.get("passive_residual_force_rms_n", 0.0)) > 0.03:
        return "passive_window_contamination"
    if float(row.get("probe_rms_tracking_error_n", 0.0)) > 0.35:
        return "force_tracking_distortion"
    if float(row.get("future_hold_settling_time_s", 0.0)) > float(locked_config["risk_envelope"]["safe"]["hold_settling_time_s"]):
        return "target_intrinsically_slow_settling"
    return "classifier_or_other"


def _policy_rows(
    physical_rows: Sequence[Mapping[str, Any]], run_dir: Path, root: Path,
    config: Mapping[str, Any], replication_config: Mapping[str, Any], policy_cache,
) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    regimes = config["sensing"]["noise_multipliers"]
    offset = int(config["held_out_population"]["measurement_seed_offset"])
    for trial_index, physical in enumerate(physical_rows):
        completed = bool(physical["completed"])
        trajectory = None
        if completed:
            trajectory = _load_trajectory(run_dir / str(physical["probe_raw_path"]))
        for regime_index, (noise_name, multiplier) in enumerate(regimes.items()):
            if completed and trajectory is not None:
                prediction = hybrid_frozen_policy_prediction(
                    trajectory,
                    TargetParameters(float(physical["stiffness_n_per_m"]), float(physical["damping_n_s_per_m"]), float(physical["effective_mass_kg"])),
                    config=config, target_id=str(physical["trial_id"]),
                    source_seed=int(physical["seed"]), case_index=int(physical["case_index"]),
                    noise_name=str(noise_name), noise_multiplier=float(multiplier),
                    measurement_seed=offset + int(physical["seed"]) * 1000 + int(physical["case_index"]) * 10 + regime_index,
                    repository_root=root, replication_config=replication_config,
                    policy_cache=policy_cache,
                )
                predicted = str(prediction["predicted_risk_class"])
                row = {
                    "decision_source": "frozen_policy",
                    "predicted_risk_score": prediction["predicted_risk_score"],
                    "decision_margin": prediction["decision_margin"],
                    "passive_persistence_veto_active": prediction["passive_persistence_veto_active"],
                    "rd_decay_rate_log": prediction["rd_decay_rate_log"],
                    "rd_threshold_dwell_fraction": prediction["rd_threshold_dwell_fraction"],
                    "rd_time_to_threshold_fraction": prediction["rd_time_to_threshold_fraction"],
                }
            else:
                # ABORT is a conservative operational NON-SAFE action, not a
                # classifier output. It is reported separately and never used
                # to claim successful frozen-policy transfer.
                predicted = "CAUTION"
                row = {
                    "decision_source": "controller_abort_non_safe",
                    "predicted_risk_score": float("nan"), "decision_margin": float("nan"),
                    "passive_persistence_veto_active": False,
                    "rd_decay_rate_log": float("nan"),
                    "rd_threshold_dwell_fraction": float("nan"),
                    "rd_time_to_threshold_fraction": float("nan"),
                }
            actual = str(physical["actual_risk_class"])
            output.append({
                "trial_id": physical["trial_id"], "stratum": physical["stratum"],
                "noise_regime": noise_name, "completed_delivery": completed,
                "predicted_risk_class": predicted, "actual_risk_class": actual,
                "predicted_safe": predicted == "SAFE", "actual_safe": actual == "SAFE",
                "false_safe": predicted == "SAFE" and actual != "SAFE",
                "binary_correct": (predicted == "SAFE") == (actual == "SAFE"),
                "failure_mechanism": _failure_mechanism(physical, policy_cache[0]),
                **row,
            })
        if (trial_index + 1) % 25 == 0 or trial_index + 1 == len(physical_rows):
            print(f"[policy {trial_index + 1:03d}/{len(physical_rows):03d}]", flush=True)
    return output


def _classification_summary(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for noise in ("low", "nominal", "high"):
        selected = [row for row in rows if row["noise_regime"] == noise]
        for scope, subset in (
            ("end_to_end_abort_is_non_safe", selected),
            ("frozen_policy_completed_delivery_only", [row for row in selected if row["decision_source"] == "frozen_policy"]),
        ):
            for stratum, values in (
                ("overall", subset),
                ("broad", [row for row in subset if row["stratum"] == "broad"]),
                ("boundary", [row for row in subset if row["stratum"] == "boundary"]),
            ):
                if values:
                    output.append({"noise_regime": noise, "scope": scope, **_binary_metrics(values, stratum=stratum)})
    return output


def run(root: Path, config_path: Path, *, run_dir: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["hybrid_controller"]["status"] != "frozen_before_held_out":
        raise RuntimeError("EXP-0009 controller is not frozen")
    freeze_tag = str(config["hybrid_controller"]["freeze_tag"])
    freeze_sha = _git(root, "rev-list", "-n", "1", freeze_tag)
    replication_path = root / "configs/experiments/exp_0007_locked_policy_replication.yaml"
    replication_config = yaml.safe_load(replication_path.read_text(encoding="utf-8"))
    policy_cache = _load_locked_policy(replication_config, root)
    locked_config = policy_cache[0]
    if run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"EXP-0009_{timestamp}_s14101_{freeze_sha[:8]}"
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for child in ("raw", "raw/held_out", "figures", "animations", "matlab"):
            (run_dir / child).mkdir(parents=True, exist_ok=False)
        (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    else:
        run_dir = run_dir.resolve()
    raw_root = run_dir / "raw/held_out"
    trials_path = run_dir / "held_out_targets.csv"
    if trials_path.exists():
        with trials_path.open(newline="", encoding="utf-8") as stream:
            trials = [Trial(
                trial_id=row["trial_id"], stratum=row["stratum"], seed=int(row["seed"]),
                case_index=int(row["case_index"]), stiffness_n_per_m=float(row["stiffness_n_per_m"]),
                damping_n_s_per_m=float(row["damping_n_s_per_m"]), effective_mass_kg=float(row["effective_mass_kg"]),
                reduced_order_boundary_severity=float(row["reduced_order_boundary_severity"]) if row["reduced_order_boundary_severity"] else None,
            ) for row in csv.DictReader(stream)]
    else:
        trials = _population(config, locked_config)
        _write_csv(trials_path, [asdict(item) for item in trials])

    print(f"EXP-0009 held-out physical delivery: {len(trials)} untouched targets", flush=True)
    existing: dict[str, Mapping[str, Any]] = {}
    for path in raw_root.glob("*__result.json"):
        item = json.loads(path.read_text(encoding="utf-8")); existing[str(item["trial_id"])] = item
    pending = [trial for trial in trials if trial.trial_id not in existing]
    payloads = [(config, locked_config, asdict(trial), str(raw_root)) for trial in pending]
    physical_rows = list(existing.values())
    workers = int(config["held_out_population"]["execution_workers"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_trial_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            physical_rows.append(future.result())
            print(f"[delivery {len(existing) + index:03d}/{len(trials):03d}]", flush=True)
    physical_rows.sort(key=lambda row: str(row["trial_id"]))
    _write_csv(run_dir / "physical_delivery_trials.csv", physical_rows)
    delivery = _delivery_summary(physical_rows, config)
    (run_dir / "delivery_summary.json").write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    print("PHYSICAL DELIVERY GATE COMPLETE — controller remains frozen", flush=True)
    print(json.dumps(delivery, indent=2), flush=True)

    policy_rows = _policy_rows(physical_rows, run_dir, root, config, replication_config, policy_cache)
    summaries = _classification_summary(policy_rows)
    false_safe = [row for row in policy_rows if bool(row["false_safe"])]
    _write_csv(run_dir / "policy_predictions.csv", policy_rows)
    _write_csv(run_dir / "classification_summary.csv", summaries)
    _write_csv(run_dir / "false_safe_audit.csv", false_safe)

    safety_events: list[Mapping[str, Any]] = []
    for row in physical_rows:
        for metric, limit in (
            ("peak_contact_force_n", 1.25),
            ("peak_attitude_deg", float(config["prospective_delivery_gates"]["maximum_attitude_deg"])),
        ):
            if float(row[metric]) > limit:
                safety_events.append({"trial_id": row["trial_id"], "metric": metric, "value": row[metric], "limit": limit})
    _write_csv(run_dir / "probe_safety_events.csv", safety_events)

    summary = {
        "experiment_id": "EXP-0009", "run_id": run_dir.name,
        "controller_freeze_tag": freeze_tag, "controller_freeze_sha": freeze_sha,
        "held_out_target_count": len(trials), "broad_count": sum(item.stratum == "broad" for item in trials),
        "boundary_count": sum(item.stratum == "boundary" for item in trials),
        "physical_delivery": delivery, "classification": summaries,
        "false_safe_audit_count": len(false_safe), "probe_safety_violation_count": len(safety_events),
        "controller_changed_after_held_out": False,
        "scientific_outcome": "C" if not delivery["passed"] else "pending_policy_interpretation",
    }
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "git_sha": _git(root, "rev-parse", "HEAD"), "controller_freeze_sha": freeze_sha,
        "ffmpeg": shutil.which("ffmpeg"),
    }
    manifest = {
        "experiment_id": "EXP-0009", "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": environment["git_sha"], "controller_freeze_sha": freeze_sha,
        "config_sha256": _sha(config_path), "locked_policy_sha256": _sha(root / str(config["frozen_references"]["locked_policy_bundle"])),
        "exp_0008_artifact_fingerprint": config["frozen_references"]["exp_0008_artifact_fingerprint"],
        "seeds": {"broad": config["held_out_population"]["broad_seeds"], "boundary": config["held_out_population"]["boundary_seeds"]},
        "artifact_index_pending": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(run_dir, flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp_0009_hybrid_contact_delivery.yaml")
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run(root, root / arguments.config, run_dir=arguments.resume)


if __name__ == "__main__":
    main()
