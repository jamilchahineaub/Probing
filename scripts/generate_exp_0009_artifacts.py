#!/usr/bin/env python3
"""Generate EXP-0009 development evidence, figures, and trajectory GIFs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from probeing.contact import TargetParameters
from probeing.experiments.coupled_uav_contact import simulate_coupled_contact
from probeing.experiments.hybrid_contact_delivery import hybrid_delivery_metrics, simulate_hybrid_contact
from probeing.plotting.stage3_hybrid import (
    animate_exp8_comparison,
    animate_hybrid,
    generate_exp0009_figures,
)
from run_exp_0009 import _save_trajectory


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _write_csv(path: Path, rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _save_exp8(path: Path, trajectory) -> None:
    phase=np.where(trajectory.time_s<=3.0,"PROBE","PASSIVE_OBSERVE")
    np.savez_compressed(path,time_s=trajectory.time_s,phase=phase,
        total_contact_reference_n=trajectory.desired_probe_force_n,
        probe_variation_reference_n=trajectory.desired_probe_force_n,
        realized_contact_force_n=trajectory.realized_contact_force_n,
        contact_active=trajectory.contact_active,target_displacement_m=trajectory.target_displacement_m,
        vehicle_position_world_m=trajectory.vehicle.position_world_m,
        vehicle_euler_xyz_rad=trajectory.vehicle.euler_xyz_rad)


def run(root: Path, run_dir: Path) -> None:
    config=yaml.safe_load((root/"configs/experiments/exp_0009_hybrid_contact_delivery.yaml").read_text())
    exp8_config=yaml.safe_load((root/"configs/experiments/exp_0008_coupled_uav_contact.yaml").read_text())
    physical=_rows(run_dir/"physical_delivery_trials.csv"); predictions=_rows(run_dir/"policy_predictions.csv")
    by_id={row["trial_id"]:row for row in physical}; nominal={row["trial_id"]:row for row in predictions if row["noise_regime"]=="nominal"}
    complete=[row for row in physical if row["completed"]=="True"]
    boundary=[row for row in complete if row["stratum"]=="boundary"]
    selections={
        "successful":"exp0009_broad_s14101_c11",
        "safe":"exp0009_broad_s14101_c11",
        "non_safe":"exp0009_broad_s14106_c00",
        "lightly_damped":min(complete,key=lambda row:float(row["damping_n_s_per_m"]))["trial_id"],
        "stiff":max(complete,key=lambda row:float(row["stiffness_n_per_m"]))["trial_id"],
        "boundary":min(boundary,key=lambda row:abs(np.log(float(row["reduced_order_boundary_severity"]))))["trial_id"],
        "failure":"exp0009_boundary_s14201_c01",
    }
    (run_dir/"representative_selections.json").write_text(json.dumps(selections,indent=2),encoding="utf-8")

    development_dir=run_dir/"raw/development"; development_dir.mkdir(parents=True,exist_ok=True); development_rows=[]
    for row in config["development_population"]["targets"]:
        target=TargetParameters(float(row["stiffness_n_per_m"]),float(row["damping_n_s_per_m"]),float(row["effective_mass_kg"]))
        trajectory=simulate_hybrid_contact(config,target); metrics=dict(hybrid_delivery_metrics(trajectory,config)); name=str(row["name"])
        _save_trajectory(development_dir/f"{name}.npz",trajectory)
        development_rows.append({**row,"terminal_phase":str(trajectory.phase[-1]),"terminal_reason":trajectory.transitions[-1].reason if trajectory.transitions else str(trajectory.phase[-1]),**metrics})
    _write_csv(run_dir/"controller_development_metrics.csv",development_rows)

    stiff_id=selections["stiff"]; stiff=by_id[stiff_id]; target=TargetParameters(float(stiff["stiffness_n_per_m"]),float(stiff["damping_n_s_per_m"]),float(stiff["effective_mass_kg"]))
    baseline=simulate_coupled_contact(exp8_config,target); baseline_path=run_dir/"raw/paired_exp0008_stiff_probe.npz"; _save_exp8(baseline_path,baseline)
    exp8_run=root/str(config["frozen_references"]["exp_0008_run"]); exp8_summary=json.loads((exp8_run/"summary.json").read_text())
    generate_exp0009_figures(run_dir,root/"results/figures",selections,exp8_summary)

    animation_dir=run_dir/"animations"; indexed=root/"results/animations"; indexed.mkdir(parents=True,exist_ok=True)
    animation_specs=[
        ("EXP-0009__contact_acquisition.gif",selections["successful"],"Contact acquisition and preload",None),
        ("EXP-0009__successful_probe.gif",selections["successful"],"Complete hybrid sequence",None),
        ("EXP-0009__lightly_damped.gif",selections["lightly_damped"],"Difficult lightly damped target",None),
        ("EXP-0009__stiff_target.gif",selections["stiff"],"Stiff target",None),
        ("EXP-0009__boundary_target.gif",selections["boundary"],"Decision-boundary target",None),
        ("EXP-0009__failure_case.gif",selections["failure"],"Failure: stable contact acquisition not achieved",None),
    ]
    for filename,trial_id,title,_ in animation_specs:
        values=np.load(run_dir/"raw/held_out"/f"{trial_id}__probe.npz",allow_pickle=False)
        decision=(
            nominal[trial_id]["predicted_risk_class"]
            if trial_id in nominal and nominal[trial_id]["decision_source"] == "frozen_policy"
            else "NON-SAFE (ABORT)"
        )
        stop=None
        if "contact_acquisition" in filename:
            preload=np.flatnonzero(values["phase"]=="PRELOAD"); stop=float(values["time_s"][preload[0]]+.75) if preload.size else float(values["time_s"][-1])
        animate_hybrid(values,animation_dir/filename,title=title,decision=decision,stop_s=stop)
        shutil.copy2(animation_dir/filename,indexed/filename)
    exp8=np.load(baseline_path,allow_pickle=False); exp9=np.load(run_dir/"raw/held_out"/f"{stiff_id}__probe.npz",allow_pickle=False)
    comparison=animation_dir/"EXP-0009__exp0008_vs_exp0009.gif"; animate_exp8_comparison(exp8,exp9,comparison); shutil.copy2(comparison,indexed/comparison.name)
    (run_dir/"animation_manifest.json").write_text(json.dumps({"ffmpeg_available":shutil.which("ffmpeg") is not None,"mp4_generated":False,"files":[path.name for path in sorted(animation_dir.glob("*.gif"))],"selections":selections},indent=2),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",type=Path); arguments=parser.parse_args(); root=Path(__file__).resolve().parents[1]; run(root,arguments.run_dir.resolve())


if __name__=="__main__": main()
