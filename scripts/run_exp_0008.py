#!/usr/bin/env python3
"""Execute immutable EXP-0008 Stage 3A coupled-UAV validation."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from probeing.contact import TargetParameters
from probeing.experiments.coupled_uav_contact import (
    CoupledTrajectory, contact_mechanics_validation, coupled_future_outcome,
    frozen_policy_prediction, no_contact_validation, probe_tracking_metrics,
    simulate_coupled_contact,
)
from probeing.experiments.decision_sufficiency import generate_target_cases
from probeing.experiments.locked_policy_replication import _binary_metrics, _load_locked_policy


@dataclass(frozen=True)
class Trial:
    trial_id: str
    stratum: str
    seed: int
    case_index: int
    target: TargetParameters
    config: Mapping[str, Any]
    perturbation: str = "nominal"
    perturbation_value: float = 1.0


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple, np.ndarray))})
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _target_from_row(row: Mapping[str, Any]) -> TargetParameters:
    return TargetParameters(float(row["stiffness_n_per_m"]), float(row["damping_n_s_per_m"]), float(row["effective_mass_kg"]))


def _trial_population(config: Mapping[str, Any]) -> list[Trial]:
    trials: list[Trial] = []
    for index, row in enumerate(config["target_population"]["representative"]):
        trials.append(Trial(f"representative_{row['name']}", "representative", 13000, index, _target_from_row(row), copy.deepcopy(config)))
    bounds = config["target_population"]
    levels = {
        "k": np.geomspace(float(bounds["stiffness_n_per_m"][0]), float(bounds["stiffness_n_per_m"][1]), 3),
        "c": np.geomspace(float(bounds["damping_n_s_per_m"][0]), float(bounds["damping_n_s_per_m"][1]), 3),
        "m": np.geomspace(float(bounds["effective_mass_kg"][0]), float(bounds["effective_mass_kg"][1]), 3),
    }
    counter = 0
    for k in levels["k"]:
        for c in levels["c"]:
            for mass in levels["m"]:
                trials.append(Trial(f"grid_{counter:03d}", "structured_grid", 13001, counter, TargetParameters(float(k), float(c), float(mass)), copy.deepcopy(config)))
                counter += 1
    cases = generate_target_cases(
        tuple(int(seed) for seed in config["sweep"]["monte_carlo_seeds"]),
        int(config["sweep"]["cases_per_seed"]),
        {
            "stiffness_n_per_m": bounds["stiffness_n_per_m"],
            "damping_n_s_per_m": bounds["damping_n_s_per_m"],
            "effective_mass_kg": bounds["effective_mass_kg"],
        }, partition="stage3a_mc",
    )
    for case in cases:
        trials.append(Trial(case.target_id, "monte_carlo", case.seed, case.case_index, TargetParameters(case.stiffness_n_per_m, case.damping_n_s_per_m, case.effective_mass_kg), copy.deepcopy(config)))

    reference = TargetParameters(450.0, 8.0, 1.0)
    variations: list[tuple[str, float, Any]] = []
    for value in config["sweep"]["vehicle_mass_multiplier"]:
        if float(value) != 1.0:
            variations.append(("vehicle_mass_multiplier", float(value), lambda cfg, v=float(value): cfg["vehicle"].__setitem__("mass_kg", float(cfg["vehicle"]["mass_kg"]) * v)))
    for value in config["sweep"]["inertia_multiplier"]:
        if float(value) != 1.0:
            variations.append(("inertia_multiplier", float(value), lambda cfg, v=float(value): cfg["vehicle"].__setitem__("inertia_body_kg_m2", (np.asarray(cfg["vehicle"]["inertia_body_kg_m2"]) * v).tolist())))
    for value in config["sweep"]["center_of_mass_error_m"]:
        if float(value) != 0.0:
            variations.append(("center_of_mass_error_m", float(value), lambda cfg, v=float(value): cfg["vehicle"].__setitem__("center_of_mass_body_m", [0.0, 0.0, v])))
    for value in config["sweep"]["probe_offset_multiplier"]:
        if float(value) != 1.0:
            variations.append(("probe_offset_multiplier", float(value), lambda cfg, v=float(value): cfg["probe_geometry"].__setitem__("tip_offset_body_m", (np.asarray(cfg["probe_geometry"]["tip_offset_body_m"]) * v).tolist())))
    for value in config["sweep"]["contact_normal_angle_deg"]:
        if float(value) != 0.0:
            radians = np.deg2rad(float(value)); variations.append(("contact_normal_angle_deg", float(value), lambda cfg, a=radians: cfg["contact"].__setitem__("normal_world", [float(np.cos(a)), float(np.sin(a)), 0.0])))
    for value in config["sweep"]["motor_lag_multiplier"]:
        if float(value) != 1.0:
            variations.append(("motor_lag_multiplier", float(value), lambda cfg, v=float(value): cfg["vehicle"].__setitem__("motor_time_constant_s", float(cfg["vehicle"]["motor_time_constant_s"]) * v)))
    for value in config["sweep"]["maximum_speed_multiplier"]:
        if float(value) != 1.0:
            variations.append(("maximum_speed_multiplier", float(value), lambda cfg, v=float(value): cfg["vehicle"].__setitem__("maximum_rotor_speed_rad_s", float(cfg["vehicle"]["maximum_rotor_speed_rad_s"]) * v)))
    for index, (name, value, mutate) in enumerate(variations):
        trial_config = copy.deepcopy(config); mutate(trial_config)
        trials.append(Trial(f"onefactor_{name}_{value:+.3f}", "one_factor", 13002, index, reference, trial_config, name, value))
    return trials


def _save_trajectory(path: Path, trajectory: CoupledTrajectory) -> None:
    np.savez_compressed(
        path, time_s=trajectory.time_s, desired_probe_force_n=trajectory.desired_probe_force_n,
        realized_contact_force_n=trajectory.realized_contact_force_n,
        contact_active=trajectory.contact_active, contact_penetration_m=trajectory.contact_penetration_m,
        contact_force_world_n=trajectory.contact_force_world_n,
        contact_torque_body_nm=trajectory.contact_torque_body_nm,
        target_displacement_m=trajectory.target_displacement_m,
        target_velocity_m_per_s=trajectory.target_velocity_m_per_s,
        target_acceleration_m_per_s2=trajectory.target_acceleration_m_per_s2,
        vehicle_position_world_m=trajectory.vehicle.position_world_m,
        vehicle_velocity_world_m_per_s=trajectory.vehicle.velocity_world_m_per_s,
        vehicle_quaternion_wxyz=trajectory.vehicle.quaternion_wxyz,
        vehicle_euler_xyz_rad=trajectory.vehicle.euler_xyz_rad,
        vehicle_angular_velocity_body_rad_s=trajectory.vehicle.angular_velocity_body_rad_s,
        rotor_speed_rad_s=trajectory.vehicle.rotor_speed_rad_s,
        rotor_thrust_n=trajectory.vehicle.rotor_thrust_n,
        actuator_reserve=trajectory.vehicle.actuator_reserve,
        motor_saturated=trajectory.vehicle.motor_saturated,
    )


def run(root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    replication_config = yaml.safe_load((root / "configs/experiments/exp_0007_locked_policy_replication.yaml").read_text(encoding="utf-8"))
    policy_cache = _load_locked_policy(replication_config, root); locked_config = policy_cache[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"EXP-0008_{timestamp}_s13101_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"
    run_dir = root / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(); (run_dir / "raw").mkdir()
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    no_contact_rows, no_contact_raw = no_contact_validation(config)
    contact_rows, contact_raw = contact_mechanics_validation(config)
    _write_csv(run_dir / "no_contact_validation.csv", no_contact_rows)
    _write_csv(run_dir / "contact_validation.csv", contact_rows)
    np.savez_compressed(run_dir / "raw/contact_validation.npz", **contact_raw)
    for name, trajectory in no_contact_raw.items():
        np.savez_compressed(run_dir / f"raw/no_contact_{name}.npz", **trajectory.__dict__)

    trials = _trial_population(config); physics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []; safety_events: list[dict[str, Any]] = []
    representative: dict[str, CoupledTrajectory] = {}; representative_future: dict[str, CoupledTrajectory] = {}
    noise_multipliers = config["sensing"]["noise_multipliers"]
    for trial_index, trial in enumerate(trials):
        probe = simulate_coupled_contact(trial.config, trial.target)
        tracking = dict(probe_tracking_metrics(probe, trial.config))
        actual, future = coupled_future_outcome(trial.config, trial.target, locked_config)
        physics_row = {
            "trial_id": trial.trial_id, "stratum": trial.stratum, "seed": trial.seed,
            "case_index": trial.case_index, "perturbation": trial.perturbation,
            "perturbation_value": trial.perturbation_value,
            "stiffness_n_per_m": trial.target.stiffness_n_per_m,
            "damping_n_s_per_m": trial.target.damping_n_s_per_m,
            "effective_mass_kg": trial.target.effective_mass_kg,
            "actual_risk_class": actual["risk_class"],
            **tracking, **{f"future_{key}": value for key, value in actual.items() if key != "risk_class"},
        }
        physics_rows.append(physics_row)
        limits = config["safety_limits"]
        for metric, limit_key in (
            ("peak_contact_force_n", "maximum_contact_force_n"),
            ("peak_target_displacement_m", "maximum_target_displacement_m"),
            ("peak_target_velocity_m_per_s", "maximum_target_velocity_m_per_s"),
            ("peak_target_acceleration_m_per_s2", "maximum_target_acceleration_m_per_s2"),
            ("peak_attitude_deg", "maximum_attitude_deg"),
        ):
            if float(tracking[metric]) > float(limits[limit_key]):
                safety_events.append({"trial_id": trial.trial_id, "metric": metric, "value": tracking[metric], "limit": limits[limit_key]})
        for regime_index, (noise_name, multiplier) in enumerate(noise_multipliers.items()):
            prediction = frozen_policy_prediction(
                probe, trial.target, target_id=trial.trial_id, source_seed=trial.seed,
                case_index=trial.case_index, noise_name=str(noise_name), noise_multiplier=float(multiplier),
                measurement_seed=trial.seed * 1000 + trial.case_index * 10 + regime_index,
                repository_root=root, replication_config=replication_config, policy_cache=policy_cache,
            )
            predicted = str(prediction["predicted_risk_class"]); actual_class = str(actual["risk_class"])
            prediction_rows.append({
                "trial_id": trial.trial_id, "stratum": trial.stratum, "noise_regime": noise_name,
                "stiffness_n_per_m": trial.target.stiffness_n_per_m,
                "damping_n_s_per_m": trial.target.damping_n_s_per_m,
                "effective_mass_kg": trial.target.effective_mass_kg,
                "predicted_risk_class": predicted, "actual_risk_class": actual_class,
                "predicted_safe": predicted == "SAFE", "actual_safe": actual_class == "SAFE",
                "false_safe": predicted == "SAFE" and actual_class != "SAFE",
                "binary_correct": (predicted == "SAFE") == (actual_class == "SAFE"),
                "predicted_risk_score": prediction["predicted_risk_score"],
                "decision_margin": prediction["decision_margin"],
                "passive_persistence_veto_active": prediction["passive_persistence_veto_active"],
                "rd_decay_rate_log": prediction["rd_decay_rate_log"],
                "rd_threshold_dwell_fraction": prediction["rd_threshold_dwell_fraction"],
                "rd_time_to_threshold_fraction": prediction["rd_time_to_threshold_fraction"],
                **{f"upper_{key}": value for key, value in prediction["upper_outcomes"].items()},
            })
        if trial.stratum == "representative":
            representative[trial.trial_id] = probe; representative_future[trial.trial_id] = future
            _save_trajectory(run_dir / f"raw/{trial.trial_id}_probe.npz", probe)
            _save_trajectory(run_dir / f"raw/{trial.trial_id}_future.npz", future)
        print(f"[{trial_index + 1:03d}/{len(trials):03d}] {trial.trial_id}", flush=True)

    ratios = []
    for row in physics_rows:
        safe = locked_config["risk_envelope"]["safe"]
        severity = max(
            float(row["future_peak_displacement_m"]) / float(safe["peak_displacement_m"]),
            float(row["future_peak_velocity_m_per_s"]) / float(safe["peak_velocity_m_per_s"]),
            float(row["future_late_hold_oscillation_rms_m"]) / float(safe["late_hold_oscillation_rms_m"]),
            float(row["future_hold_settling_time_s"]) / float(safe["hold_settling_time_s"]),
        )
        row["actual_severity_ratio"] = severity; ratios.append((abs(np.log(max(severity, 1e-12))), row["trial_id"]))
    boundary_ids = {trial_id for _, trial_id in sorted(ratios)[:min(20, len(ratios))]}
    for row in physics_rows: row["boundary_case"] = row["trial_id"] in boundary_ids
    for row in prediction_rows: row["boundary_case"] = row["trial_id"] in boundary_ids

    summaries: list[Mapping[str, Any]] = []
    for noise in noise_multipliers:
        selected = [row for row in prediction_rows if row["noise_regime"] == noise]
        summaries.append({"noise_regime": noise, **_binary_metrics(selected, stratum="overall")})
        boundary = [row for row in selected if row["boundary_case"]]
        summaries.append({"noise_regime": noise, **_binary_metrics(boundary, stratum="boundary")})
    _write_csv(run_dir / "physics_metrics.csv", physics_rows); _write_csv(run_dir / "predictions.csv", prediction_rows)
    _write_csv(run_dir / "classification_summary.csv", summaries); _write_csv(run_dir / "probe_safety_events.csv", safety_events)
    np.savez_compressed(run_dir / "raw/representative_index.npz", names=np.asarray(list(representative), dtype=str))
    convergence_target = TargetParameters(450.0, 0.5, 1.0); convergence_rows = []
    for step in config["simulation"]["convergence_steps_s"]:
        trajectory = simulate_coupled_contact(config, convergence_target, integration_step_s=float(step))
        convergence_rows.append({"step_s": step, **probe_tracking_metrics(trajectory, config)})
    _write_csv(run_dir / "timestep_convergence.csv", convergence_rows)

    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "git_sha": _git(root, "rev-parse", "HEAD"), "stage2_tag_sha": _git(root, "rev-list", "-n", "1", str(config["frozen_baseline"]["stage2_tag"])),
    }
    summary = {
        "experiment_id": "EXP-0008", "run_id": run_id, "trial_count": len(trials),
        "prediction_count": len(prediction_rows), "no_contact_all_passed": all(bool(row["passed"]) for row in no_contact_rows),
        "contact_all_passed": all(bool(row["passed"]) for row in contact_rows),
        "probe_safety_violation_count": len(safety_events), "classification": summaries,
        "median_probe_rms_tracking_error_n": float(np.median([row["probe_rms_tracking_error_n"] for row in physics_rows])),
        "median_relative_probe_rms_tracking_error": float(np.median([row["probe_relative_rms_tracking_error"] for row in physics_rows])),
        "median_probe_phase_lag_rad": float(np.median([row["probe_weighted_phase_lag_rad"] for row in physics_rows])),
        "median_delivery_bandwidth_hz": float(np.median([row["probe_delivery_bandwidth_hz"] for row in physics_rows])),
        "maximum_attitude_deg": float(np.max([row["peak_attitude_deg"] for row in physics_rows])),
        "motor_saturation_trial_fraction": float(np.mean([row["motor_saturation_fraction"] > 0 for row in physics_rows])),
        "stage3a_result": "characterized",
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_id, "created_utc": timestamp, "config_sha256": _sha(config_path),
        "locked_policy_sha256": _sha(root / str(config["frozen_baseline"]["locked_policy_bundle"])),
        "git_sha": environment["git_sha"], "seeds": config["sweep"]["monte_carlo_seeds"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/experiments/exp_0008_coupled_uav_contact.yaml")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    run_dir = run(root, root / args.config); print(run_dir)


if __name__ == "__main__":
    main()
