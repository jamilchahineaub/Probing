#!/usr/bin/env python3
"""Run a configured experiment without overwriting prior artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
from typing import Any, Mapping

import numpy as np
import yaml

from probeing.experiments import run_interaction_identification
from probeing.experiments.reduced_kelvin_voigt import run_kelvin_voigt_validation
from probeing.plotting import (
    plot_interaction_identification,
    plot_kelvin_voigt_validation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _utc_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id(experiment_id: str, timestamp: datetime, seed: int) -> str:
    time_part = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{experiment_id}_{time_part}_s{seed}_{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _artifact_reference(path: Path) -> str:
    """Prefer repository-relative paths while supporting external smoke roots."""

    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _git_provenance(repository_root: Path) -> Mapping[str, Any]:
    command = ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"]
    try:
        top_level = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {
            "commit_sha": None,
            "status": "git_unavailable_or_not_a_work_tree",
            "top_level": None,
        }

    if Path(top_level).resolve() != repository_root.resolve():
        return {
            "commit_sha": None,
            "status": "workspace_not_git_root",
            "top_level": top_level,
        }

    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {
            "commit_sha": None,
            "status": "git_repository_has_no_commit",
            "top_level": top_level,
        }
    try:
        worktree_status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        worktree_status = ["status unavailable"]
    return {
        "commit_sha": commit,
        "status": "available",
        "top_level": top_level,
        "worktree_dirty": bool(worktree_status),
        "worktree_status": worktree_status,
    }


def _software_versions() -> Mapping[str, str]:
    package_names = ("numpy", "matplotlib", "PyYAML", "pytest", "scipy")
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _write_raw_csv(path: Path, raw: Mapping[str, np.ndarray]) -> None:
    names = list(raw)
    lengths = {len(raw[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("raw output columns must have equal length")
    open_stream = gzip.open if path.suffix == ".gz" else Path.open
    mode = "xt" if path.suffix == ".gz" else "x"
    with open_stream(path, mode, encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(raw[name] for name in names)))


def _write_row_table(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    if not rows:
        raise ValueError("row table cannot be empty")
    names = list(rows[0])
    if any(list(row) != names for row in rows):
        raise ValueError("all table rows must have the same ordered fields")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_table(path: Path, metrics: Mapping[str, float | int]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for name, value in sorted(metrics.items()):
            writer.writerow((name, value))


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported or missing configuration schema_version")
    return config


def run(config_path: Path, runs_root: Path, results_root: Path) -> tuple[str, bool]:
    config = _load_config(config_path)
    experiment_id = str(config["experiment"]["id"])
    random_seed = int(config["experiment"]["random_seed"])
    np.random.seed(random_seed)
    timestamp = _utc_timestamp()
    run_id = _new_run_id(experiment_id, timestamp, random_seed)
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    git = _git_provenance(REPOSITORY_ROOT)
    manifest_path = run_directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp.isoformat(),
        "status": "running",
        "success": None,
        "random_seed": random_seed,
        "seed_set": config["experiment"]["seed_set"],
        "git_commit_sha": git["commit_sha"],
        "git": git,
        "software_versions": _software_versions(),
        "configuration": config,
        "target_model_parameters": config.get("targets", config.get("target")),
        "uav_parameters": config.get(
            "uav", {"status": "excluded from Milestone A"}
        ),
        "estimator_settings": config.get("estimators", config.get("estimator")),
        "controller_settings": config.get(
            "controller", {"status": "excluded from Milestone A"}
        ),
        "files": {},
    }
    _write_json(manifest_path, manifest)

    try:
        config_output = run_directory / "config.resolved.yaml"
        with config_output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

        milestone_a_identification = "targets" in config
        result = (
            run_interaction_identification(config)
            if milestone_a_identification
            else run_kelvin_voigt_validation(config)
        )
        raw_csv = run_directory / "raw_timeseries.csv.gz"
        _write_raw_csv(raw_csv, result.raw)
        raw_npz = run_directory / "raw_timeseries.npz"
        with raw_npz.open("xb") as stream:
            np.savez_compressed(stream, **result.raw)

        metrics_path = run_directory / "metrics.json"
        safety_path = run_directory / "safety_events.json"
        acceptance_path = run_directory / "acceptance_checks.json"
        _write_json(metrics_path, result.metrics)
        _write_json(safety_path, list(result.safety_events))
        _write_json(acceptance_path, result.acceptance_checks)

        results_figure_dir = results_root / "figures"
        results_table_dir = results_root / "tables"
        results_figure_dir.mkdir(parents=True, exist_ok=True)
        results_table_dir.mkdir(parents=True, exist_ok=True)
        indexed_table = results_table_dir / f"{run_id}__metrics.csv"
        figure_files: dict[str, str] = {}
        indexed_figure_files: dict[str, str] = {}

        if milestone_a_identification:
            case_table = run_directory / "case_metrics.csv"
            _write_row_table(case_table, result.case_metrics)
            indexed_case_table = results_table_dir / f"{run_id}__case_metrics.csv"
            if indexed_case_table.exists():
                raise FileExistsError(
                    f"refusing to overwrite indexed result {indexed_case_table}"
                )
            shutil.copy2(case_table, indexed_case_table)
            figures = plot_interaction_identification(
                raw=result.raw,
                case_metrics=result.case_metrics,
                representative_case_id=result.representative_case_id,
                output_directory=run_directory / "figures",
            )
            for figure_name, (figure_png, figure_pdf) in figures.items():
                figure_files[f"run_figure_{figure_name}_png"] = _artifact_reference(
                    figure_png
                )
                figure_files[f"run_figure_{figure_name}_pdf"] = _artifact_reference(
                    figure_pdf
                )
                indexed_png = results_figure_dir / f"{run_id}__{figure_name}.png"
                indexed_pdf = results_figure_dir / f"{run_id}__{figure_name}.pdf"
                for path in (indexed_png, indexed_pdf):
                    if path.exists():
                        raise FileExistsError(f"refusing to overwrite indexed result {path}")
                shutil.copy2(figure_png, indexed_png)
                shutil.copy2(figure_pdf, indexed_pdf)
                indexed_figure_files[
                    f"indexed_figure_{figure_name}_png"
                ] = _artifact_reference(indexed_png)
                indexed_figure_files[
                    f"indexed_figure_{figure_name}_pdf"
                ] = _artifact_reference(indexed_pdf)
        else:
            figure_base = run_directory / "figures" / "kelvin_voigt_validation"
            figure_png, figure_pdf = plot_kelvin_voigt_validation(
                raw=result.raw,
                stiffness_n_per_m=float(config["target"]["stiffness_n_per_m"]),
                damping_n_s_per_m=float(config["target"]["damping_n_s_per_m"]),
                output_base=figure_base,
            )
            indexed_png = results_figure_dir / f"{run_id}__kelvin_voigt_validation.png"
            indexed_pdf = results_figure_dir / f"{run_id}__kelvin_voigt_validation.pdf"
            for path in (indexed_png, indexed_pdf):
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite indexed result {path}")
            shutil.copy2(figure_png, indexed_png)
            shutil.copy2(figure_pdf, indexed_pdf)
            figure_files = {
                "run_figure_png": _artifact_reference(figure_png),
                "run_figure_pdf": _artifact_reference(figure_pdf),
            }
            indexed_figure_files = {
                "indexed_figure_png": _artifact_reference(indexed_png),
                "indexed_figure_pdf": _artifact_reference(indexed_pdf),
            }

        if indexed_table.exists():
            raise FileExistsError(f"refusing to overwrite indexed result {indexed_table}")
        _write_metric_table(indexed_table, result.metrics)

        manifest["status"] = "success" if result.success else "failed_acceptance"
        manifest["success"] = result.success
        manifest["completed_timestamp_utc"] = _utc_timestamp().isoformat()
        manifest["calculated_metrics"] = result.metrics
        manifest["safety_violations"] = list(result.safety_events)
        manifest["acceptance_checks"] = result.acceptance_checks
        manifest["files"] = {
            "resolved_config": _artifact_reference(config_output),
            "raw_csv": _artifact_reference(raw_csv),
            "raw_npz": _artifact_reference(raw_npz),
            "metrics": _artifact_reference(metrics_path),
            "safety_events": _artifact_reference(safety_path),
            "acceptance_checks": _artifact_reference(acceptance_path),
            "indexed_metrics_table": _artifact_reference(indexed_table),
            **(
                {
                    "case_metrics": _artifact_reference(case_table),
                    "indexed_case_metrics": _artifact_reference(indexed_case_table),
                }
                if milestone_a_identification
                else {}
            ),
            **figure_files,
            **indexed_figure_files,
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
        temporary_manifest = run_directory / "manifest.failed.json"
        _write_json(temporary_manifest, manifest)
        os.replace(temporary_manifest, manifest_path)
        raise

    temporary_manifest = run_directory / "manifest.completed.json"
    _write_json(temporary_manifest, manifest)
    os.replace(temporary_manifest, manifest_path)
    return run_id, bool(manifest["success"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "results"
    )
    arguments = parser.parse_args()
    run_id, success = run(
        arguments.config.resolve(),
        arguments.runs_root.resolve(),
        arguments.results_root.resolve(),
    )
    print(json.dumps({"run_id": run_id, "success": success}, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
