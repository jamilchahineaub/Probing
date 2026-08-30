#!/usr/bin/env python3
"""Run immutable EXP-0006 Stage 1 passive ring-down artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from probeing.experiments import run_passive_ringdown
from probeing.plotting import plot_passive_ringdown

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
        raise ValueError("EXP-0006 requires a schema_version 1 YAML mapping")
    if config.get("experiment", {}).get("id") != "EXP-0006":
        raise ValueError("this entry point only runs EXP-0006")
    validation = [int(seed) for seed in config["seed_partitions"]["validation"]]
    forbidden = [
        int(seed)
        for seed in config["integrity_acceptance"][
            "forbidden_validation_seeds_before_final_run"
        ]
    ]
    if validation != forbidden:
        raise ValueError(
            "final validation seed list must equal the predeclared untouched seed list"
        )
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
    columns = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **columns)


def _write_raw_gzip(path: Path, raw: Mapping[str, np.ndarray]) -> None:
    names = list(raw)
    if not names or len({len(raw[name]) for name in names}) != 1:
        raise ValueError("raw arrays must be non-empty and equal length")
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(raw[name] for name in names)))


def _experiment_fingerprint(repository_root: Path, experiment_id: str) -> str:
    selected: list[Path] = []
    underscored = experiment_id.lower().replace("-", "_")
    for root_name in ("configs/experiments", "runs", "results"):
        root = repository_root / root_name
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repository_root).as_posix()
            if underscored in relative.lower() or f"/{experiment_id}_" in f"/{relative}":
                selected.append(path)
    aggregate = hashlib.sha256()
    for path in sorted(
        selected, key=lambda value: value.relative_to(repository_root).as_posix()
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(repository_root).as_posix()
        aggregate.update(f"{digest}  {relative}\n".encode("utf-8"))
    return aggregate.hexdigest()


def run(config_path: Path, runs_root: Path, results_root: Path) -> tuple[str, bool, str]:
    config = _load_config(config_path)
    predecessor_ids = ("EXP-0001", "EXP-0002", "EXP-0003", "EXP-0004", "EXP-0005")
    before = {
        experiment_id: _experiment_fingerprint(REPOSITORY_ROOT, experiment_id)
        for experiment_id in predecessor_ids
    }
    expected = {
        experiment_id.upper().replace("_", "-"): str(
            reference["expected_artifact_fingerprint"]
        )
        for experiment_id, reference in config["frozen_references"].items()
    }
    if before != expected:
        raise RuntimeError(f"frozen predecessor artifact fingerprint mismatch: {before}")

    partitions = {
        name: [int(seed) for seed in config["seed_partitions"][name]]
        for name in ("training", "calibration", "validation")
    }
    timestamp = _utc_timestamp()
    run_id = _new_run_id("EXP-0006", timestamp, partitions["validation"][0])
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = run_directory / "manifest.json"
    git = _git_provenance(REPOSITORY_ROOT)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": "EXP-0006",
        "timestamp_utc": timestamp.isoformat(),
        "status": "running",
        "success": None,
        "stage1_decision": None,
        "seed_partitions": partitions,
        "seed_set": "untouched_validation",
        "git_commit_sha": git["commit_sha"],
        "git": git,
        "software_versions": _software_versions(),
        "configuration": config,
        "predecessor_fingerprints_before": before,
        "files": {},
    }
    _write_json(manifest_path, manifest)

    try:
        resolved = run_directory / "config.resolved.yaml"
        with resolved.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

        def progress(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {"event": "progress", "completed": completed, "total": total},
                    sort_keys=True,
                ),
                flush=True,
            )

        result = run_passive_ringdown(config, progress_callback=progress)
        paths = {
            "validation_rows_csv": run_directory / "validation_predictions.csv.gz",
            "validation_rows_npz": run_directory / "validation_predictions.npz",
            "duration_summary": run_directory / "duration_summary.csv",
            "feature_set_summary": run_directory / "feature_set_summary.csv",
            "quantitative_summary": run_directory / "quantitative_summary.csv",
            "early_stop_rows_csv": run_directory / "early_stop_rows.csv.gz",
            "early_stop_rows_npz": run_directory / "early_stop_rows.npz",
            "early_stop_summary": run_directory / "early_stop_summary.csv",
            "legacy_audit": run_directory / "exp0005_false_safe_audit.csv",
            "feature_importance": run_directory / "feature_importance.csv",
            "class_distribution": run_directory / "class_distribution.csv",
            "representative_csv": run_directory / "representative_chirp_ringdown.csv.gz",
            "representative_npz": run_directory / "representative_chirp_ringdown.npz",
            "legacy_raw_csv": run_directory / "exp0005_false_safe_timeseries.csv.gz",
            "legacy_raw_npz": run_directory / "exp0005_false_safe_timeseries.npz",
            "summary": run_directory / "summary.json",
            "metrics": run_directory / "metrics.json",
            "safety_events": run_directory / "probe_safety_events.json",
            "acceptance_checks": run_directory / "acceptance_checks.json",
            "stage1_decision": run_directory / "stage1_decision.json",
        }
        _write_rows_gzip(paths["validation_rows_csv"], result.validation_rows)
        _write_rows_npz(paths["validation_rows_npz"], result.validation_rows)
        _write_row_table(paths["duration_summary"], result.duration_summary)
        _write_row_table(paths["feature_set_summary"], result.feature_set_summary)
        _write_row_table(paths["quantitative_summary"], result.quantitative_summary)
        _write_rows_gzip(paths["early_stop_rows_csv"], result.early_stop_rows)
        _write_rows_npz(paths["early_stop_rows_npz"], result.early_stop_rows)
        _write_row_table(paths["early_stop_summary"], result.early_stop_summary)
        _write_row_table(paths["legacy_audit"], result.legacy_audit_rows)
        _write_row_table(paths["feature_importance"], result.feature_importance)
        _write_row_table(paths["class_distribution"], result.class_distribution)
        _write_raw_gzip(paths["representative_csv"], result.representative_raw)
        _write_raw_gzip(paths["legacy_raw_csv"], result.legacy_raw)
        with paths["representative_npz"].open("xb") as stream:
            np.savez_compressed(stream, **result.representative_raw)
        with paths["legacy_raw_npz"].open("xb") as stream:
            np.savez_compressed(stream, **result.legacy_raw)
        _write_json(paths["summary"], result.summary)
        _write_json(paths["metrics"], result.metrics)
        _write_json(paths["safety_events"], list(result.safety_events))
        _write_json(paths["acceptance_checks"], result.acceptance_checks)
        _write_json(
            paths["stage1_decision"],
            {
                "stage1_decision": result.stage1_decision,
                "stage1_safety_criterion": config["stage1_safety_criterion"],
                "passive_ringdown_safety_criterion_pass": result.summary[
                    "passive_ringdown_safety_criterion_pass"
                ],
                "qualifying_fixed_durations_s": result.summary[
                    "qualifying_fixed_durations_s"
                ],
                "selected_reporting_duration_s": result.summary[
                    "selected_reporting_duration_s"
                ],
            },
        )

        figures = plot_passive_ringdown(
            validation_rows=result.validation_rows,
            duration_summary=result.duration_summary,
            feature_set_summary=result.feature_set_summary,
            early_stop_rows=result.early_stop_rows,
            legacy_audit_rows=result.legacy_audit_rows,
            representative_raw=result.representative_raw,
            legacy_raw=result.legacy_raw,
            summary=result.summary,
            config=config,
            output_directory=run_directory / "figures",
        )
        results_figure_dir = results_root / "figures"
        results_table_dir = results_root / "tables"
        results_figure_dir.mkdir(parents=True, exist_ok=True)
        results_table_dir.mkdir(parents=True, exist_ok=True)
        indexed: dict[str, str] = {}
        for name, (png, pdf) in figures.items():
            for source in (png, pdf):
                destination = results_figure_dir / f"{run_id}__{name}{source.suffix}"
                if destination.exists():
                    raise FileExistsError(f"refusing to overwrite {destination}")
                shutil.copy2(source, destination)
                indexed[f"indexed_figure_{name}_{source.suffix[1:]}"] = (
                    _artifact_reference(destination)
                )
            indexed[f"run_figure_{name}_png"] = _artifact_reference(png)
            indexed[f"run_figure_{name}_pdf"] = _artifact_reference(pdf)

        for name in (
            "duration_summary",
            "feature_set_summary",
            "quantitative_summary",
            "early_stop_summary",
            "legacy_audit",
            "feature_importance",
            "class_distribution",
        ):
            destination = results_table_dir / f"{run_id}__{name}.csv"
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite {destination}")
            shutil.copy2(paths[name], destination)
            indexed[f"indexed_{name}"] = _artifact_reference(destination)
        indexed_metrics = results_table_dir / f"{run_id}__metrics.csv"
        _write_metric_table(indexed_metrics, result.metrics)
        indexed["indexed_metrics"] = _artifact_reference(indexed_metrics)

        after = {
            experiment_id: _experiment_fingerprint(REPOSITORY_ROOT, experiment_id)
            for experiment_id in predecessor_ids
        }
        if after != before:
            raise RuntimeError("EXP-0001 through EXP-0005 artifacts changed during EXP-0006")

        manifest["status"] = "success" if result.success else "failed_integrity"
        manifest["success"] = result.success
        manifest["stage1_decision"] = result.stage1_decision
        manifest["completed_timestamp_utc"] = _utc_timestamp().isoformat()
        manifest["calculated_metrics"] = result.metrics
        manifest["acceptance_checks"] = result.acceptance_checks
        manifest["scientific_summary"] = result.summary
        manifest["safety_violations"] = list(result.safety_events)
        manifest["predecessor_fingerprints_after"] = after
        manifest["files"] = {
            "resolved_config": _artifact_reference(resolved),
            **{name: _artifact_reference(path) for name, path in paths.items()},
            **indexed,
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
        / "exp_0006_passive_ringdown.yaml",
    )
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--results-root", type=Path, default=REPOSITORY_ROOT / "results")
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
