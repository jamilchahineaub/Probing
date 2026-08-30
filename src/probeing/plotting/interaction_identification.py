"""Plots for the Milestone A interaction-identification experiment."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

_matplotlib_cache = Path(tempfile.gettempdir()) / "probeing-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _save_figure(figure: Any, output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    if png_path.exists() or pdf_path.exists():
        raise FileExistsError(f"refusing to overwrite figure at {output_base}")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _rows_by(
    rows: Sequence[Mapping[str, Any]], *, scenario: str | None = None, probe: str | None = None
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if (scenario is None or row["scenario"] == scenario)
        and (probe is None or row["probe"] == probe)
    ]


def plot_interaction_identification(
    *,
    raw: Mapping[str, NDArray[Any]],
    case_metrics: Sequence[Mapping[str, Any]],
    representative_case_id: str,
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    """Create all figures requested for EXP-0001."""

    outputs: dict[str, tuple[Path, Path]] = {}
    representative_mask = raw["case_id"] == representative_case_id
    if not np.any(representative_mask):
        raise ValueError("representative_case_id is absent from raw data")
    representative_target = str(raw["target"][representative_mask][0])
    representative_scenario = str(raw["scenario"][representative_mask][0])
    time = raw["time_s"][representative_mask].astype(float)

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True)
    axes[0].plot(
        time,
        raw["true_contact_force_n"][representative_mask].astype(float),
        color="black",
        linewidth=1.8,
        label="true contact force",
    )
    axes[0].plot(
        time,
        raw["measured_contact_force_n"][representative_mask].astype(float),
        color="tab:red",
        alpha=0.55,
        linewidth=0.8,
        label="measured contact force",
    )
    axes[0].set_ylabel("force [N]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        1.0e3 * raw["true_displacement_m"][representative_mask].astype(float),
        color="tab:blue",
        linewidth=1.8,
        label="true displacement",
    )
    axes[1].plot(
        time,
        1.0e3 * raw["measured_displacement_m"][representative_mask].astype(float),
        color="tab:orange",
        alpha=0.55,
        linewidth=0.8,
        label="measured displacement",
    )
    axes[1].set_ylabel("displacement [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    target_scenario_mask = (raw["target"] == representative_target) & (
        raw["scenario"] == representative_scenario
    )
    for probe_name in dict.fromkeys(raw["probe"][target_scenario_mask].tolist()):
        mask = target_scenario_mask & (raw["probe"] == probe_name)
        axes[2].plot(
            raw["time_s"][mask].astype(float),
            raw["probe_force_n"][mask].astype(float),
            label=str(probe_name),
        )
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("probe signal [N]")
    axes[2].legend(loc="best", ncol=3)
    axes[2].grid(True, alpha=0.3)
    figure.suptitle(f"EXP-0001 time histories — {representative_case_id}")
    figure.tight_layout()
    outputs["timeseries"] = _save_figure(
        figure, output_directory / "exp_0001_timeseries"
    )

    parameter_specs = (
        ("stiffness_n_per_m", "stiffness k [N/m]"),
        ("damping_n_s_per_m", "damping c [N s/m]"),
        ("effective_mass_kg", "effective mass m_eff [kg]"),
    )
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in case_metrics))
    scenario_colors = dict(zip(scenarios, ("tab:blue", "tab:orange", "tab:green")))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for axis, (suffix, label) in zip(axes, parameter_specs):
        all_values: list[float] = []
        for scenario in scenarios:
            rows = _rows_by(case_metrics, scenario=scenario)
            truth = np.asarray([row[f"true_{suffix}"] for row in rows], dtype=float)
            batch = np.asarray([row[f"batch_{suffix}"] for row in rows], dtype=float)
            recursive = np.asarray([row[f"rls_{suffix}"] for row in rows], dtype=float)
            color = scenario_colors[scenario]
            axis.scatter(
                truth,
                batch,
                marker="o",
                facecolors="none",
                edgecolors=color,
                label=f"batch, {scenario}",
            )
            axis.scatter(
                truth,
                recursive,
                marker="x",
                color=color,
                label=f"RLS, {scenario}",
            )
            all_values.extend(truth.tolist() + batch.tolist() + recursive.tolist())
        lower = min(all_values)
        upper = max(all_values)
        margin = 0.05 * max(upper - lower, abs(upper), 1.0)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "k--")
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlabel(f"true {label}")
        axis.set_ylabel(f"estimated {label}")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("True versus estimated interaction parameters")
    figure.tight_layout()
    outputs["parameter_comparison"] = _save_figure(
        figure, output_directory / "exp_0001_parameter_comparison"
    )

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    error_specs = (
        ("rls_stiffness_relative_error", "k"),
        ("rls_damping_relative_error", "c"),
        ("rls_effective_mass_relative_error", "m_eff"),
    )
    for column, label in error_specs:
        error_percent = 100.0 * np.abs(raw[column][representative_mask].astype(float))
        axis.semilogy(time, np.maximum(error_percent, 1.0e-10), label=label)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("absolute parameter error [%]")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best")
    axis.set_title(f"RLS parameter estimation error — {representative_case_id}")
    figure.tight_layout()
    outputs["rls_error"] = _save_figure(
        figure, output_directory / "exp_0001_rls_error"
    )

    probes = list(dict.fromkeys(str(row["probe"]) for row in case_metrics))
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    group_width = 0.8
    series = [
        (scenario, estimator)
        for scenario in scenarios
        for estimator in ("batch", "rls")
    ]
    bar_width = group_width / len(series)
    x_position = np.arange(len(probes), dtype=float)
    for series_index, (scenario, estimator) in enumerate(series):
        values = []
        for probe_name in probes:
            rows = [
                row
                for row in case_metrics
                if row["scenario"] == scenario and row["probe"] == probe_name
            ]
            values.append(
                100.0
                * float(np.mean([row[f"{estimator}_relative_error_rms"] for row in rows]))
            )
        offset = (series_index - 0.5 * (len(series) - 1)) * bar_width
        axis.bar(
            x_position + offset,
            np.maximum(values, 1.0e-12),
            width=bar_width,
            label=f"{estimator}, {scenario}",
        )
    axis.set_yscale("log")
    axis.set_xticks(x_position, probes)
    axis.set_ylabel("mean RMS relative parameter error [%]")
    axis.set_xlabel("probing signal")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend(loc="best", fontsize=8, ncol=2)
    axis.set_title("Identification error for each bounded probing signal")
    figure.tight_layout()
    outputs["probe_error"] = _save_figure(
        figure, output_directory / "exp_0001_probe_error"
    )

    perfect_scenario = scenarios[0]
    condition = []
    correlation = []
    for probe_name in probes:
        rows = _rows_by(case_metrics, scenario=perfect_scenario, probe=probe_name)
        condition.append(max(float(row["normalized_condition_number"]) for row in rows))
        correlation.append(max(float(row["maximum_abs_parameter_correlation"]) for row in rows))
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    axes[0].bar(probes, condition, color="tab:purple")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("worst normalized condition number")
    axes[0].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].bar(probes, correlation, color="tab:brown")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("worst |parameter correlation|")
    axes[1].set_xlabel("probing signal")
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("Regression observability diagnostics across perfect-measurement cases")
    figure.tight_layout()
    outputs["observability"] = _save_figure(
        figure, output_directory / "exp_0001_observability"
    )
    return outputs
