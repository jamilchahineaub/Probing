#!/usr/bin/env python3
"""Run the immutable EXP-0007 locked-policy replication."""
from __future__ import annotations
import argparse, csv, gzip, json, os, shutil, traceback
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import yaml
from probeing.experiments import run_locked_policy_replication
from probeing.plotting import plot_locked_policy_replication
from run_experiment import (REPOSITORY_ROOT, _artifact_reference, _git_provenance,
    _new_run_id, _software_versions, _utc_timestamp, _write_json, _write_metric_table, _write_row_table)
from run_exp_0006 import _experiment_fingerprint

def _rows_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: return
    fields=list(rows[0])
    with gzip.open(path, "xt", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def _raw_gz(path: Path, raw: Mapping[str, np.ndarray]) -> None:
    with gzip.open(path, "xt", encoding="utf-8", newline="") as f:
        w=csv.writer(f); names=list(raw); w.writerow(names); w.writerows(zip(*(raw[n] for n in names)))
def _npz(path: Path, data: Any) -> None:
    with path.open("xb") as f:
        if isinstance(data, Mapping) and data and isinstance(next(iter(data.values())), np.ndarray): np.savez_compressed(f, **data)
        else: np.savez_compressed(f, **{k: np.asarray([r[k] for r in data]) for k in data[0]})

def run(config_path: Path, runs_root: Path, results_root: Path) -> tuple[str,bool,str]:
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment",{}).get("id") != "EXP-0007": raise ValueError("wrong experiment config")
    predecessor=tuple(config["frozen_references"])
    before={eid.upper().replace("_","-"):_experiment_fingerprint(REPOSITORY_ROOT,eid.upper().replace("_","-")) for eid in predecessor}
    expected={eid.upper().replace("_","-"):str(v["expected_artifact_fingerprint"]) for eid,v in config["frozen_references"].items()}
    if before != expected: raise RuntimeError(f"frozen predecessor fingerprint mismatch: {before}")
    seed=int(config["replication_population"]["broad_seed_range"]["first"])
    timestamp=_utc_timestamp(); run_id=_new_run_id("EXP-0007",timestamp,seed); rd=runs_root/run_id; rd.mkdir(parents=True)
    manifest={"schema_version":1,"run_id":run_id,"experiment_id":"EXP-0007","timestamp_utc":timestamp.isoformat(),"status":"running","success":None,"stage1_decision":None,"git":_git_provenance(REPOSITORY_ROOT),"software_versions":_software_versions(),"configuration":config,"predecessor_fingerprints_before":before,"files":{}}
    mp=rd/"manifest.json"; _write_json(mp,manifest)
    try:
        resolved=rd/"config.resolved.yaml"; resolved.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
        result=run_locked_policy_replication(config, repository_root=REPOSITORY_ROOT)
        paths={"validation_predictions_csv":rd/"validation_predictions.csv.gz","validation_predictions_npz":rd/"validation_predictions.npz","binary_summary":rd/"binary_summary.csv","secondary_summary":rd/"secondary_summary.csv","confidence_summary":rd/"confidence_summary.csv","boundary_summary":rd/"boundary_summary.csv","quantitative_summary":rd/"quantitative_summary.csv","comparison_summary":rd/"comparison_summary.csv","margin_summary":rd/"margin_summary.csv","parameter_rows":rd/"parameter_rows.csv","boundary_selection":rd/"boundary_selection.csv","representative_raw_csv":rd/"representative_raw.csv.gz","representative_raw_npz":rd/"representative_raw.npz","false_safe_audit_csv":rd/"false_safe_audit.csv.gz","false_safe_audit_npz":rd/"false_safe_audit.npz","summary":rd/"summary.json","metrics":rd/"metrics.json","acceptance_checks":rd/"acceptance_checks.json","probe_safety_events":rd/"probe_safety_events.json","stage1_decision":rd/"stage1_decision.json"}
        _rows_gz(paths["validation_predictions_csv"],result.rows); _npz(paths["validation_predictions_npz"],result.rows)
        for key,data in (("binary_summary",result.binary_summary),("secondary_summary",result.secondary_summary),("confidence_summary",result.confidence_summary),("boundary_summary",result.boundary_summary),("quantitative_summary",result.quantitative_summary),("comparison_summary",result.comparison_summary),("margin_summary",result.margin_summary),("parameter_rows",result.parameter_rows),("boundary_selection",result.boundary_cases)):
            _write_row_table(paths[key],data)
        for key,data in (("representative_raw",result.representative_raw),("false_safe_audit",result.false_safe_raw)):
            _raw_gz(paths[key+"_csv"],data); _npz(paths[key+"_npz"],data)
        _write_json(paths["summary"],result.summary); _write_json(paths["metrics"],result.metrics); _write_json(paths["acceptance_checks"],result.acceptance_checks); _write_json(paths["probe_safety_events"],list(result.safety_events)); _write_json(paths["stage1_decision"],{"stage1_decision":result.stage1_decision,"stage1_replication_criterion":config["stage1_replication_criterion"],"pass":result.summary["stage1_replication_criterion_pass"]})
        figs=plot_locked_policy_replication(rows=result.rows,binary_summary=result.binary_summary,secondary_summary=result.secondary_summary,confidence_summary=result.confidence_summary,comparison_summary=result.comparison_summary,margin_summary=result.margin_summary,parameter_rows=result.parameter_rows,representative_raw=result.representative_raw,false_safe_raw=result.false_safe_raw,summary=result.summary,output_directory=rd/"figures")
        idx={}; (results_root/"figures").mkdir(parents=True,exist_ok=True); (results_root/"tables").mkdir(parents=True,exist_ok=True)
        for name,(png,pdf) in figs.items():
            for src in (png,pdf): dest=results_root/"figures"/f"{run_id}__{name}{src.suffix}"; shutil.copy2(src,dest); idx[f"indexed_figure_{name}_{src.suffix[1:]}"]=_artifact_reference(dest)
        for key in ("binary_summary","secondary_summary","confidence_summary","boundary_summary","quantitative_summary","comparison_summary","margin_summary","parameter_rows","boundary_selection"):
            dest=results_root/"tables"/f"{run_id}__{key}.csv"; shutil.copy2(paths[key],dest); idx[f"indexed_{key}"]=_artifact_reference(dest)
        after={eid.upper().replace("_","-"):_experiment_fingerprint(REPOSITORY_ROOT,eid.upper().replace("_","-")) for eid in predecessor}
        if after!=before: raise RuntimeError("predecessor artifacts changed during EXP-0007")
        manifest.update(status="success" if result.success else "stage_gate_failed",success=result.success,stage1_decision=result.stage1_decision,completed_timestamp_utc=_utc_timestamp().isoformat(),calculated_metrics=result.metrics,acceptance_checks=result.acceptance_checks,scientific_summary=result.summary,predecessor_fingerprints_after=after,files={"resolved_config":_artifact_reference(resolved),**{k:_artifact_reference(v) for k,v in paths.items()},**idx})
    except Exception as error:
        manifest.update(status="execution_error",success=False,completed_timestamp_utc=_utc_timestamp().isoformat(),execution_error={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}); _write_json(rd/"manifest.failed.json",manifest); raise
    complete=rd/"manifest.completed.json"; _write_json(complete,manifest); os.replace(complete,mp)
    return run_id,bool(manifest["success"]),str(manifest["stage1_decision"])

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=REPOSITORY_ROOT/"configs/experiments/exp_0007_locked_policy_replication.yaml"); p.add_argument("--runs-root",type=Path,default=REPOSITORY_ROOT/"runs"); p.add_argument("--results-root",type=Path,default=REPOSITORY_ROOT/"results"); a=p.parse_args(); rid,ok,decision=run(a.config.resolve(),a.runs_root.resolve(),a.results_root.resolve()); print(json.dumps({"run_id":rid,"success":ok,"stage1_decision":decision})); raise SystemExit(0 if ok else 1)
