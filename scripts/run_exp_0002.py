#!/usr/bin/env python3
"""Run immutable EXP-0002 practical-identifiability validation artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from probeing.experiments import run_practical_identifiability
from probeing.plotting import plot_practical_identifiability

from run_experiment import (
    REPOSITORY_ROOT,
    _artifact_reference,
    _git_provenance,
    _new_run_id,
    _software_versions,
    _utc_timestamp,
    _write_json,
    _write_metric_table,
    _write_row_table,
)


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("EXP-0002 requires a schema_version 1 YAML mapping")
    if config.get("experiment", {}).get("id") != "EXP-0002":
        raise ValueError("this entry point only runs EXP-0002")
    return config


def _write_rows_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("row table cannot be empty")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("all rows must have identical ordered fields")
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_npz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = {
        name: np.asarray([row[name] for row in rows]) for name in rows[0]
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **columns)


def _write_raw_csv_gzip(path: Path, raw: Mapping[str, np.ndarray]) -> None:
    names = list(raw)
    if len({len(raw[name]) for name in names}) != 1:
        raise ValueError("representative raw columns must have equal length")
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(raw[name] for name in names)))


def _exp0001_fingerprint(repository_root: Path) -> str:
    """Match a sha256sum-of-sha256sum fingerprint for all EXP-0001 artifacts."""

    selected: list[Path] = []
    for root_name in ("configs/experiments", "runs", "results"):
        root = repository_root / root_name
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repository_root).as_posix()
            if "exp_0001" in relative.lower() or "/EXP-0001_" in f"/{relative}":
                selected.append(path)
    aggregate = hashlib.sha256()
    for path in sorted(selected, key=lambda value: value.relative_to(repository_root).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(repository_root).as_posix()
        aggregate.update(f"{digest}  {relative}\n".encode("utf-8"))
    return aggregate.hexdigest()


def run(config_path: Path, runs_root: Path, results_root: Path) -> tuple[str, bool, str]:
    config = _load_config(config_path)
    expected_fingerprint = str(
        config["exp_0001_reference"]["expected_artifact_fingerprint"]
    )
    before_fingerprint = _exp0001_fingerprint(REPOSITORY_ROOT)
    if before_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "EXP-0001 artifact fingerprint differs from the frozen EXP-0002 reference"
        )

    seeds = [int(seed) for seed in config["validation_seeds"]]
    timestamp = _utc_timestamp()
    run_id = _new_run_id("EXP-0002", timestamp, seeds[0])
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = run_directory / "manifest.json"
    git = _git_provenance(REPOSITORY_ROOT)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": "EXP-0002",
        "timestamp_utc": timestamp.isoformat(),
        "status": "running",
        "success": None,
        "stage1_decision": None,
        "validation_seeds": seeds,
        "seed_set": "validation",
        "git_commit_sha": git["commit_sha"],
        "git": git,
        "software_versions": _software_versions(),
        "configuration": config,
        "exp_0001_artifact_fingerprint_before": before_fingerprint,
        "files": {},
    }
    _write_json(manifest_path, manifest)

    try:
        resolved_config = run_directory / "config.resolved.yaml"
        with resolved_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

        def progress(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {"event": "progress", "completed": completed, "total": total},
                    sort_keys=True,
                ),
                flush=True,
            )

        result = run_practical_identifiability(
            config,
            repository_root=REPOSITORY_ROOT,
            progress_callback=progress,
        )

        trials_csv = run_directory / "monte_carlo_trials.csv.gz"
        trials_npz = run_directory / "monte_carlo_trials.npz"
        aggregate_csv = run_directory / "aggregate_metrics.csv"
        candidates_csv = run_directory / "go_no_go_candidates.csv"
        representative_csv = run_directory / "representative_timeseries.csv.gz"
        representative_npz = run_directory / "representative_timeseries.npz"
        metrics_json = run_directory / "metrics.json"
        safety_json = run_directory / "safety_events.json"
        acceptance_json = run_directory / "acceptance_checks.json"
        ramp_json = run_directory / "ramp_failure_analysis.json"
        decision_json = run_directory / "stage1_decision.json"

        _write_rows_gzip(trials_csv, result.trial_metrics)
        _write_rows_npz(trials_npz, result.trial_metrics)
        _write_row_table(aggregate_csv, result.aggregate_metrics)
        _write_row_table(candidates_csv, result.go_candidates)
        _write_raw_csv_gzip(representative_csv, result.representative_raw)
        with representative_npz.open("xb") as stream:
            np.savez_compressed(stream, **result.representative_raw)
        _write_json(metrics_json, result.metrics)
        _write_json(safety_json, list(result.safety_events))
        _write_json(acceptance_json, result.acceptance_checks)
        _write_json(ramp_json, result.ramp_failure_analysis)
        _write_json(
            decision_json,
            {
                "stage1_decision": result.stage1_decision,
                "best_candidate": result.best_candidate,
                "criteria": config["go_no_go"],
            },
        )

        figures = plot_practical_identifiability(
            trials=result.trial_metrics,
            aggregate=result.aggregate_metrics,
            representative_raw=result.representative_raw,
            ramp_analysis=result.ramp_failure_analysis,
            best_candidate=result.best_candidate,
            output_directory=run_directory / "figures",
        )

        results_figure_dir = results_root / "figures"
        results_table_dir = results_root / "tables"
        results_figure_dir.mkdir(parents=True, exist_ok=True)
        results_table_dir.mkdir(parents=True, exist_ok=True)
        indexed_files: dict[str, str] = {}
        for name, (png, pdf) in figures.items():
            indexed_png = results_figure_dir / f"{run_id}__{name}.png"
            indexed_pdf = results_figure_dir / f"{run_id}__{name}.pdf"
            for destination in (indexed_png, indexed_pdf):
                if destination.exists():
                    raise FileExistsError(f"refusing to overwrite {destination}")
            shutil.copy2(png, indexed_png)
            shutil.copy2(pdf, indexed_pdf)
            indexed_files[f"run_figure_{name}_png"] = _artifact_reference(png)
            indexed_files[f"run_figure_{name}_pdf"] = _artifact_reference(pdf)
            indexed_files[f"indexed_figure_{name}_png"] = _artifact_reference(
                indexed_png
            )
            indexed_files[f"indexed_figure_{name}_pdf"] = _artifact_reference(
                indexed_pdf
            )

        table_sources = {
            "aggregate_metrics": aggregate_csv,
            "go_no_go_candidates": candidates_csv,
        }
        for name, source in table_sources.items():
            destination = results_table_dir / f"{run_id}__{name}.csv"
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite {destination}")
            shutil.copy2(source, destination)
            indexed_files[f"indexed_{name}"] = _artifact_reference(destination)
        indexed_summary = results_table_dir / f"{run_id}__metrics.csv"
        _write_metric_table(indexed_summary, result.metrics)
        indexed_files["indexed_summary_metrics"] = _artifact_reference(indexed_summary)

        after_fingerprint = _exp0001_fingerprint(REPOSITORY_ROOT)
        if after_fingerprint != before_fingerprint:
            raise RuntimeError("EXP-0001 artifacts changed while EXP-0002 was running")

        manifest["status"] = "success" if result.success else "failed_integrity"
        manifest["success"] = result.success
        manifest["stage1_decision"] = result.stage1_decision
        manifest["completed_timestamp_utc"] = _utc_timestamp().isoformat()
        manifest["calculated_metrics"] = result.metrics
        manifest["acceptance_checks"] = result.acceptance_checks
        manifest["safety_violations"] = list(result.safety_events)
        manifest["exp_0001_artifact_fingerprint_after"] = after_fingerprint
        manifest["files"] = {
            "resolved_config": _artifact_reference(resolved_config),
            "monte_carlo_trials_csv": _artifact_reference(trials_csv),
            "monte_carlo_trials_npz": _artifact_reference(trials_npz),
            "aggregate_metrics": _artifact_reference(aggregate_csv),
            "go_no_go_candidates": _artifact_reference(candidates_csv),
            "representative_timeseries_csv": _artifact_reference(representative_csv),
            "representative_timeseries_npz": _artifact_reference(representative_npz),
            "metrics": _artifact_reference(metrics_json),
            "safety_events": _artifact_reference(safety_json),
            "acceptance_checks": _artifact_reference(acceptance_json),
            "ramp_failure_analysis": _artifact_reference(ramp_json),
            "stage1_decision": _artifact_reference(decision_json),
            **indexed_files,
        }
    except Exception as error:
        manifest["status"] = "execution_error"
        manifest["success"] = False
        manifest["completed_timestamp_utc"] = _utc_timestamp().isoformat()
        manifest["execution_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        failed = run_directory / "manifest.failed.json"
        _write_json(failed, manifest)
        os.replace(failed, manifest_path)
        raise

    complete = run_directory / "manifest.completed.json"
    _write_json(complete, manifest)
    os.replace(complete, manifest_path)
    return run_id, bool(manifest["success"]), str(manifest["stage1_decision"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "exp_0002_practical_identifiability.yaml",
    )
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "results"
    )
    arguments = parser.parse_args()
    run_id, success, decision = run(
        arguments.config.resolve(),
        arguments.runs_root.resolve(),
        arguments.results_root.resolve(),
    )
    print(
        json.dumps(
            {"run_id": run_id, "success": success, "stage1_decision": decision},
            sort_keys=True,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
