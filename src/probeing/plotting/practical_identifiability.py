"""Strong diagnostic plots for EXP-0002 practical identifiability."""

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


def _save(figure: Any, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise FileExistsError(f"refusing to overwrite figure at {base}")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _select(rows: Sequence[Mapping[str, Any]], **conditions: Any) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]


def _boxplot(axis: Any, data: list[list[float]], labels: list[str], ylabel: str) -> None:
    axis.boxplot(data, labels=labels, showfliers=False)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.3)


def plot_practical_identifiability(
    *,
    trials: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    representative_raw: Mapping[str, NDArray[Any]],
    ramp_analysis: Mapping[str, Any],
    best_candidate: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    """Create the EXP-0002 sensing, observability, disturbance, and failure plots."""

    outputs: dict[str, tuple[Path, Path]] = {}
    practical_regimes = {"no_direct_acceleration", "imu_like"}
    practical = [row for row in trials if row["sensing_regime"] in practical_regimes]
    severities = sorted(
        {str(row["severity"]) for row in trials},
        key=lambda name: next(
            int(row["severity_index"]) for row in trials if row["severity"] == name
        ),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    noise_data = [
        [100.0 * float(row["parameter_relative_error_rms"]) for row in practical if row["severity"] == severity]
        for severity in severities
    ]
    noise_labels = [
        f"{severity}\nσx={next(row['displacement_std_m'] for row in trials if row['severity'] == severity):.0e} m"
        for severity in severities
    ]
    _boxplot(axes[0], noise_data, noise_labels, "RMS parameter error [%]")
    axes[0].set_yscale("log")
    axes[0].set_title("Estimation error vs imperfection severity")
    rate_order = sorted({float(row["sample_rate_hz"]) for row in practical})
    rate_data = [
        [100.0 * float(row["parameter_relative_error_rms"]) for row in practical if float(row["sample_rate_hz"]) == rate]
        for rate in rate_order
    ]
    _boxplot(
        axes[1],
        rate_data,
        [f"{rate:g}" for rate in rate_order],
        "RMS parameter error [%]",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sample rate [Hz]")
    axes[1].set_title("Estimation error vs sampling rate")
    figure.suptitle("Coupled sensing-imperfection sweep (10 validation seeds)")
    figure.tight_layout()
    outputs["noise_and_sampling"] = _save(
        figure, output_directory / "exp_0002_noise_and_sampling"
    )

    nominal_unilateral = [
        row
        for row in practical
        if row["severity"] == "nominal" and row["contact_mode"] == "unilateral"
    ]
    sensing_labels = list(
        dict.fromkeys(
            f"{row['sensing_regime']}\n{row['pipeline']}" for row in nominal_unilateral
        )
    )
    parameter_fields = (
        ("stiffness_relative_error", "|k error| [%]"),
        ("damping_relative_error", "|c error| [%]"),
        ("effective_mass_relative_error", "|m_eff error| [%]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    for axis, (field, ylabel) in zip(axes.flat[:3], parameter_fields):
        data = [
            [
                100.0 * abs(float(row[field]))
                for row in nominal_unilateral
                if f"{row['sensing_regime']}\n{row['pipeline']}" == label
            ]
            for label in sensing_labels
        ]
        _boxplot(axis, data, sensing_labels, ylabel)
        axis.set_yscale("log")
        axis.tick_params(axis="x", labelrotation=25)
    probes = list(dict.fromkeys(str(row["probe"]) for row in nominal_unilateral))
    mass_p95 = [
        float(
            np.percentile(
                [
                    100.0 * abs(float(row["effective_mass_relative_error"]))
                    for row in nominal_unilateral
                    if row["probe"] == probe
                ],
                95.0,
            )
        )
        for probe in probes
    ]
    axes[1, 1].bar(probes, mass_p95, color="tab:red")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("95th-percentile |m_eff error| [%]")
    axes[1, 1].set_title("Effective-mass error by probe")
    axes[1, 1].tick_params(axis="x", labelrotation=20)
    axes[1, 1].grid(True, axis="y", which="both", alpha=0.3)
    figure.suptitle("Parameter errors by practical measurement pipeline — nominal unilateral")
    figure.tight_layout()
    outputs["pipeline_and_parameter_errors"] = _save(
        figure, output_directory / "exp_0002_pipeline_and_parameter_errors"
    )

    reference_rows = [
        row
        for row in trials
        if row["sensing_regime"] == "optimistic_reference"
        and row["severity"] == "mild"
        and row["contact_mode"] == "bilateral"
    ]
    probes_reference = list(dict.fromkeys(str(row["probe"]) for row in reference_rows))
    singular = []
    condition = []
    correlation = []
    for probe in probes_reference:
        rows = [row for row in reference_rows if row["probe"] == probe]
        singular.append(np.median([row["normalized_singular_value_3"] for row in rows]))
        condition.append(np.median([row["normalized_condition_number"] for row in rows]))
        correlation.append(np.median([row["maximum_abs_parameter_correlation"] for row in rows]))
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    axes[0].bar(probes_reference, singular, color="tab:green")
    axes[0].set_ylabel("median minimum singular value")
    axes[1].bar(probes_reference, condition, color="tab:purple")
    axes[1].set_ylabel("median normalized condition")
    axes[1].set_yscale("log")
    axes[2].bar(probes_reference, correlation, color="tab:brown")
    axes[2].set_ylabel("median max |correlation|")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("probe")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Singular values, conditioning, and correlation by probe")
    figure.tight_layout()
    outputs["observability_by_probe"] = _save(
        figure, output_directory / "exp_0002_observability_by_probe"
    )

    raw = representative_raw
    signal_cases = [
        ("optimistic_reference", "direct", "mild", "clean/reference"),
        ("no_direct_acceleration", "finite_difference", "nominal", "noisy finite difference"),
        ("no_direct_acceleration", "savitzky_golay", "nominal", "Savitzky–Golay"),
        ("imu_like", "low_pass", "severe", "severe IMU-like"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    for regime, pipeline, severity, label in signal_cases:
        mask = (
            (raw["target"] == "compliant")
            & (raw["probe"] == "chirp")
            & (raw["contact_mode"] == "unilateral")
            & (raw["sensing_regime"] == regime)
            & (raw["pipeline"] == pipeline)
            & (raw["severity"] == severity)
        )
        if not np.any(mask):
            continue
        time = raw["time_s"][mask].astype(float)
        axes[0].plot(time, 1.0e3 * raw["processed_displacement_m"][mask].astype(float), label=label)
        axes[1].plot(time, raw["processed_acceleration_m_per_s2"][mask].astype(float), label=label)
        axes[2].plot(time, raw["processed_force_n"][mask].astype(float), label=label)
    axes[0].set_ylabel("processed x [mm]")
    axes[1].set_ylabel("processed acceleration [m/s²]")
    axes[2].set_ylabel("processed force [N]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Representative clean, noisy, derived, and filtered signals")
    figure.tight_layout()
    outputs["representative_signals"] = _save(
        figure, output_directory / "exp_0002_representative_signals"
    )

    pareto_rows = [
        row
        for row in aggregate
        if row["severity"] == "nominal"
        and row["contact_mode"] == "unilateral"
        and row["sensing_regime"] in practical_regimes
    ]
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    error = np.asarray([100.0 * row["parameter_error_p95"] for row in pareto_rows])
    scatter = axis.scatter(
        1.0e3 * np.asarray([row["peak_target_displacement_m"] for row in pareto_rows]),
        [row["normalized_information_score"] for row in pareto_rows],
        c=np.log10(np.maximum(error, 1.0e-8)),
        cmap="viridis_r",
        alpha=0.75,
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("log10 95th-percentile parameter error [%]")
    axis.set_xlabel("peak target displacement [mm]")
    axis.set_ylabel("normalized information score")
    axis.set_title("Information vs structural disturbance (heuristic Pareto view)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    outputs["information_disturbance"] = _save(
        figure, output_directory / "exp_0002_information_disturbance"
    )

    chosen_regime = str(best_candidate["sensing_regime"])
    chosen_pipeline = str(best_candidate["pipeline"])
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    probes_contact = list(dict.fromkeys(str(row["probe"]) for row in trials))
    width = 0.36
    x = np.arange(len(probes_contact))
    for index, contact_mode in enumerate(("bilateral", "unilateral")):
        values = []
        for probe in probes_contact:
            rows = [
                row
                for row in trials
                if row["severity"] == "nominal"
                and row["contact_mode"] == contact_mode
                and row["sensing_regime"] == chosen_regime
                and row["pipeline"] == chosen_pipeline
                and row["probe"] == probe
            ]
            values.append(
                100.0
                * float(np.percentile([row["parameter_relative_error_rms"] for row in rows], 95.0))
            )
        axis.bar(x + (index - 0.5) * width, values, width=width, label=contact_mode)
    axis.set_xticks(x, probes_contact)
    axis.set_yscale("log")
    axis.set_ylabel("95th-percentile RMS parameter error [%]")
    axis.set_title(f"Unilateral vs bilateral — {chosen_regime}, {chosen_pipeline}")
    axis.legend()
    axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    outputs["contact_mode_comparison"] = _save(
        figure, output_directory / "exp_0002_contact_mode_comparison"
    )

    heat_labels = list(
        dict.fromkeys(
            f"{row['sensing_regime']}\n{row['pipeline']}" for row in nominal_unilateral
        )
    )
    heat = np.empty((len(heat_labels), len(probes)))
    for row_index, label in enumerate(heat_labels):
        for column_index, probe in enumerate(probes):
            values = [
                row["parameter_relative_error_rms"]
                for row in nominal_unilateral
                if f"{row['sensing_regime']}\n{row['pipeline']}" == label
                and row["probe"] == probe
            ]
            heat[row_index, column_index] = 100.0 * np.percentile(values, 95.0)
    figure, axis = plt.subplots(figsize=(10.0, 6.0))
    image = axis.imshow(np.log10(np.maximum(heat, 1.0e-8)), aspect="auto", cmap="magma")
    axis.set_xticks(np.arange(len(probes)), probes)
    axis.set_yticks(np.arange(len(heat_labels)), heat_labels)
    axis.set_xlabel("probe")
    axis.set_title("Failure heatmap: log10 95th-percentile RMS parameter error [%]")
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            axis.text(column_index, row_index, f"{heat[row_index, column_index]:.1f}", ha="center", va="center", color="white", fontsize=7)
    figure.colorbar(image, ax=axis, label="log10 error [%]")
    figure.tight_layout()
    outputs["failure_heatmap"] = _save(
        figure, output_directory / "exp_0002_failure_heatmap"
    )

    comparisons = ramp_analysis["probe_diagnostics"]
    ramp_probes = list(comparisons)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ramp_fields = (
        ("force_dc_energy_fraction", "force DC energy fraction"),
        ("acceleration_feature_rms_m_per_s2", "acceleration feature RMS [m/s²]"),
        ("inertial_contribution_rms_fraction", "inertial contribution / force RMS"),
        ("maximum_abs_parameter_correlation", "max |parameter correlation|"),
    )
    for axis, (field, ylabel) in zip(axes.flat, ramp_fields):
        axis.bar(ramp_probes, [comparisons[probe][field] for probe in ramp_probes])
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("EXP-0001 ramp failure: static content and weak inertial excitation")
    figure.tight_layout()
    outputs["ramp_failure_analysis"] = _save(
        figure, output_directory / "exp_0002_ramp_failure_analysis"
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharey=True)
    for axis, (label, title) in zip(
        axes,
        (("stiffness", "k"), ("damping", "c"), ("effective_mass", "m_eff")),
    ):
        values = []
        for sensing_label in sensing_labels:
            rows = [
                row
                for row in nominal_unilateral
                if f"{row['sensing_regime']}\n{row['pipeline']}" == sensing_label
            ]
            values.append(
                100.0
                * float(np.mean([row[f"{label}_eiv_relative_shift"] for row in rows]))
            )
        axis.bar(sensing_labels, values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("mean EIV-induced parameter shift [%]")
    figure.suptitle("Errors-in-variables bias by practical pipeline")
    figure.tight_layout()
    outputs["errors_in_variables"] = _save(
        figure, output_directory / "exp_0002_errors_in_variables"
    )

    sensor_mask = (
        (raw["target"] == "compliant")
        & (raw["probe"] == "chirp")
        & (raw["contact_mode"] == "unilateral")
        & (raw["sensing_regime"] == "sensorless_force_exploratory")
        & (raw["severity"] == "nominal")
    )
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    time = raw["time_s"][sensor_mask].astype(float)
    axes[0].plot(time, raw["true_contact_force_n"][sensor_mask].astype(float), label="true contact force")
    axes[0].plot(time, raw["processed_force_n"][sensor_mask].astype(float), label="command-based proxy")
    axes[0].set_ylabel("force [N]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    measured_force_rows = [
        row
        for row in trials
        if row["severity"] == "nominal"
        and row["contact_mode"] == "unilateral"
        and row["pipeline"] == "savitzky_golay"
        and row["sensing_regime"] in {"imu_like", "sensorless_force_exploratory"}
    ]
    labels = ["measured force", "exploratory proxy"]
    data = [
        [
            100.0 * row["parameter_relative_error_rms"]
            for row in measured_force_rows
            if row["sensing_regime"] == regime
        ]
        for regime in ("imu_like", "sensorless_force_exploratory")
    ]
    _boxplot(axes[1], data, labels, "RMS parameter error [%]")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("force source")
    axes[1].set_title("Exploratory sensorless force is not used for GO")
    figure.tight_layout()
    outputs["sensorless_force_exploratory"] = _save(
        figure, output_directory / "exp_0002_sensorless_force_exploratory"
    )
    return outputs
