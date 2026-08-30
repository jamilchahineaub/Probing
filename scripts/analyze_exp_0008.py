#!/usr/bin/env python3
"""Post-run scientific audit and figure generation for immutable EXP-0008 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from probeing.contact import TargetParameters
from probeing.experiments.coupled_uav_contact import simulate_coupled_contact
from probeing.experiments.decision_sufficiency import (
    TargetCase, _maneuver_signal, evaluate_future_response, simulate_population,
)
from probeing.experiments.locked_policy_replication import _binary_metrics, _load_locked_policy
from probeing.plotting.stage3_contact import generate_stage3_figures


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def save_trajectory(path: Path, trajectory: Any) -> None:
    np.savez_compressed(
        path, time_s=trajectory.time_s, desired_probe_force_n=trajectory.desired_probe_force_n,
        realized_contact_force_n=trajectory.realized_contact_force_n,
        contact_active=trajectory.contact_active, contact_penetration_m=trajectory.contact_penetration_m,
        contact_force_world_n=trajectory.contact_force_world_n, contact_torque_body_nm=trajectory.contact_torque_body_nm,
        target_displacement_m=trajectory.target_displacement_m, target_velocity_m_per_s=trajectory.target_velocity_m_per_s,
        target_acceleration_m_per_s2=trajectory.target_acceleration_m_per_s2,
        vehicle_position_world_m=trajectory.vehicle.position_world_m, vehicle_velocity_world_m_per_s=trajectory.vehicle.velocity_world_m_per_s,
        vehicle_quaternion_wxyz=trajectory.vehicle.quaternion_wxyz, vehicle_euler_xyz_rad=trajectory.vehicle.euler_xyz_rad,
        vehicle_angular_velocity_body_rad_s=trajectory.vehicle.angular_velocity_body_rad_s,
        rotor_speed_rad_s=trajectory.vehicle.rotor_speed_rad_s, rotor_thrust_n=trajectory.vehicle.rotor_thrust_n,
        actuator_reserve=trajectory.vehicle.actuator_reserve, motor_saturated=trajectory.vehicle.motor_saturated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; run_dir = root / args.run_dir
    cfg = yaml.safe_load((root / "configs/experiments/exp_0008_coupled_uav_contact.yaml").read_text())
    rep = yaml.safe_load((root / "configs/experiments/exp_0007_locked_policy_replication.yaml").read_text())
    locked, _, _ = _load_locked_policy(rep, root)
    physics = read_rows(run_dir / "physics_metrics.csv"); predictions = read_rows(run_dir / "predictions.csv")

    cases = tuple(TargetCase(row["trial_id"], "postrun", int(row["seed"]), int(row["case_index"]), float(row["stiffness_n_per_m"]), float(row["damping_n_s_per_m"]), float(row["effective_mass_kg"])) for row in physics)
    maneuver_time, maneuver_force = _maneuver_signal(locked)
    direct_population = simulate_population(cases, maneuver_time, maneuver_force, contact_mode="unilateral")
    direct_truth = evaluate_future_response(direct_population, cases, locked)
    direct_by_id = {row["target_id"]: row for row in direct_truth}
    comparison: list[dict[str, Any]] = []
    for row in physics:
        direct = direct_by_id[row["trial_id"]]
        coupled = row["actual_risk_class"]
        comparison.append({
            "trial_id": row["trial_id"], "stratum": row["stratum"],
            "direct_risk_class": direct["risk_class"], "coupled_risk_class": coupled,
            "binary_label_changed": (direct["risk_class"] == "SAFE") != (coupled == "SAFE"),
            "direct_peak_displacement_m": direct["peak_displacement_m"], "coupled_peak_displacement_m": float(row["future_peak_displacement_m"]),
            "direct_settling_time_s": direct["hold_settling_time_s"], "coupled_settling_time_s": float(row["future_hold_settling_time_s"]),
            "coupled_sustained_force_rms_error_n": float(row["future_realized_sustained_force_rms_error_n"]),
        })
    write_rows(run_dir / "reduced_vs_coupled_truth.csv", comparison)

    extended = []
    for noise in ("low", "nominal", "high"):
        for name, strata in (
            ("monte_carlo", {"monte_carlo"}),
            ("target_population", {"representative", "structured_grid", "monte_carlo"}),
            ("one_factor", {"one_factor"}),
        ):
            rows = [row for row in predictions if row["noise_regime"] == noise and row["stratum"] in strata]
            extended.append({"noise_regime": noise, **_binary_metrics(rows, stratum=name)})
    write_rows(run_dir / "classification_summary_extended.csv", extended)

    nominal_false = [row for row in predictions if row["noise_regime"] == "nominal" and row["false_safe"] == "True"]
    false_ids = {row["trial_id"] for row in nominal_false}; physics_by_id = {row["trial_id"]: row for row in physics}
    cause_counts = {"settling": 0, "peak_displacement": 0, "peak_velocity": 0, "oscillation": 0, "contact_loss": 0}
    safe = locked["risk_envelope"]["safe"]
    for trial_id in false_ids:
        row = physics_by_id[trial_id]
        cause_counts["settling"] += float(row["future_hold_settling_time_s"]) > float(safe["hold_settling_time_s"])
        cause_counts["peak_displacement"] += float(row["future_peak_displacement_m"]) > float(safe["peak_displacement_m"])
        cause_counts["peak_velocity"] += float(row["future_peak_velocity_m_per_s"]) > float(safe["peak_velocity_m_per_s"])
        cause_counts["oscillation"] += float(row["future_late_hold_oscillation_rms_m"]) > float(safe["late_hold_oscillation_rms_m"])
        cause_counts["contact_loss"] += float(row["future_peak_displacement_m"]) > float(locked["risk_envelope"]["contact_tracking_displacement_m"])
    changed = {row["trial_id"] for row in comparison if row["binary_label_changed"]}
    target_population_ids = {row["trial_id"] for row in physics if row["stratum"] != "one_factor"}
    summary = {
        "target_trial_count": len(physics), "independent_monte_carlo_target_count": 60,
        "nominal_false_safe_count_all": len(nominal_false),
        "nominal_false_safe_causes": cause_counts,
        "binary_label_changed_by_coupling_count": len(changed),
        "binary_label_changed_target_population_count": len(changed & target_population_ids),
        "nominal_false_safe_with_changed_ground_truth_count": len(false_ids & changed),
        "median_direct_settling_time_s": float(np.median([float(row["direct_settling_time_s"]) for row in comparison])),
        "median_coupled_settling_time_s": float(np.median([float(row["coupled_settling_time_s"]) for row in comparison])),
        "recontact_trial_fraction": float(np.mean([float(row["contact_loss_count"]) > 0 for row in physics])),
        "probe_force_limit_violation_trial_fraction": float(np.mean([float(row["peak_contact_force_n"]) > float(cfg["safety_limits"]["maximum_contact_force_n"]) for row in physics])),
        "persistence_veto_fraction_nominal": float(np.mean([row["passive_persistence_veto_active"] == "True" for row in predictions if row["noise_regime"] == "nominal"])),
        "stage3a_gate": "CONTINUE_STAGE_3A",
        "scientific_interpretation": "Frozen target-only policy does not transfer safely: coupled sustained-force tracking creates slow-settling outcomes not predicted by passive target ring-down.",
    }
    (run_dir / "postrun_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if nominal_false:
        example = physics_by_id[nominal_false[0]["trial_id"]]
        target = TargetParameters(float(example["stiffness_n_per_m"]), float(example["damping_n_s_per_m"]), float(example["effective_mass_kg"]))
        trajectory = simulate_coupled_contact(cfg, target)
        save_trajectory(run_dir / "raw/nominal_false_safe_example.npz", trajectory)
        (run_dir / "raw/nominal_false_safe_example.json").write_text(json.dumps({"trial_id": example["trial_id"], "target": target.__dict__}, indent=2), encoding="utf-8")

    results_dir = root / "results/figures"; results_dir.mkdir(parents=True, exist_ok=True)
    generate_stage3_figures(run_dir, results_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
