"""Publication-style diagnostic plots for EXP-0004."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PARAMETERS = ("stiffness", "damping", "effective_mass")
LABELS = {"stiffness": "k", "damping": "c", "effective_mass": r"$m_{eff}$"}


def _save(figure: plt.Figure, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    pdf = directory / f"{name}.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _final_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row["is_final"])]


def plot_sequential_identification(
    *,
    trial_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    stage_aggregate: Sequence[Mapping[str, Any]],
    strategy_summary: Sequence[Mapping[str, Any]],
    frequency_information: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    """Create the predeclared EXP-0004 figure set."""

    plt.style.use("default")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})
    figures: dict[str, tuple[Path, Path]] = {}
    adaptive_stage = [row for row in stage_aggregate if row["strategy"] == "uncertainty_driven"]
    adaptive_stage.sort(key=lambda row: int(row["probe_index"]))

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for parameter in PARAMETERS:
        axis.plot(
            [row["probe_index"] for row in adaptive_stage],
            [100.0 * float(row[f"{parameter}_median_relative_standard_error"]) for row in adaptive_stage],
            marker="o",
            label=LABELS[parameter],
        )
    axis.set(xlabel="Completed probes", ylabel="Median estimated relative standard error (%)")
    axis.set_xticks([1, 2, 3])
    axis.legend(title="Parameter")
    figures["uncertainty_after_each_probe"] = _save(
        figure, output_directory, "uncertainty_after_each_probe"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for axis, parameter in zip(axes, PARAMETERS):
        axis.plot(
            [row["probe_index"] for row in adaptive_stage],
            [100.0 * float(row[f"{parameter}_median_abs_relative_error"]) for row in adaptive_stage],
            marker="o",
            color="tab:blue",
        )
        axis.fill_between(
            [row["probe_index"] for row in adaptive_stage],
            0.0,
            [100.0 * float(row[f"{parameter}_p95_abs_relative_error"]) for row in adaptive_stage],
            alpha=0.18,
            color="tab:blue",
            label="p95 envelope",
        )
        axis.set_title(LABELS[parameter])
        axis.set_xlabel("Completed probes")
        axis.set_xticks([1, 2, 3])
    axes[0].set_ylabel("Absolute relative error (%)")
    axes[-1].legend(loc="upper right")
    figures["parameter_error_after_each_probe"] = _save(
        figure, output_directory, "parameter_error_after_each_probe"
    )

    adaptive_trials = [row for row in trial_rows if row["strategy"] == "uncertainty_driven"]
    targets = sorted({str(row["target"]) for row in adaptive_trials})
    probes = sorted({str(row["probe_name"]) for row in adaptive_trials})
    counts = np.zeros((len(targets), len(probes)), dtype=float)
    for row in adaptive_trials:
        counts[targets.index(str(row["target"])), probes.index(str(row["probe_name"]))] += 1.0
    totals = np.maximum(np.sum(counts, axis=1, keepdims=True), 1.0)
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    image = axis.imshow(counts / totals, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(np.arange(len(probes)), probes, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(targets)), targets)
    axis.set_title("Adaptive probe usage fraction by target")
    figure.colorbar(image, ax=axis, label="Fraction of executed probes")
    figures["selected_probe_sequence_by_target"] = _save(
        figure, output_directory, "selected_probe_sequence_by_target"
    )

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for strategy in sorted({str(row["strategy"]) for row in stage_aggregate}):
        values = sorted(
            [row for row in stage_aggregate if row["strategy"] == strategy],
            key=lambda row: int(row["probe_index"]),
        )
        axis.plot(
            [row["probe_index"] for row in values],
            [100.0 * float(row["full_vector_success_probability"]) for row in values],
            marker="o",
            label=strategy.replace("_", " "),
        )
    axis.set(xlabel="Completed probes", ylabel="Full-vector success probability (%)")
    axis.set_xticks([1, 2, 3])
    axis.legend(fontsize=8)
    figures["success_rate_vs_probe_count"] = _save(
        figure, output_directory, "success_rate_vs_probe_count"
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in strategy_summary:
        axis.scatter(
            float(row["median_force_squared_dose_n2_s"]),
            float(row["median_information_per_force_dose"]),
            s=70,
            label=str(row["strategy"]).replace("_", " "),
        )
    axis.set(
        xlabel=r"Median cumulative command dose (N$^2$ s)",
        ylabel="Median log-information / command dose",
        title="Information versus bounded disturbance",
    )
    axis.legend(fontsize=8)
    figures["information_vs_disturbance"] = _save(
        figure, output_directory, "information_vs_disturbance"
    )

    names = [str(row["strategy"]) for row in strategy_summary]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(
        np.arange(len(names)),
        [100.0 * float(row["full_vector_success_probability"]) for row in strategy_summary],
        color=["tab:orange" if name == "uncertainty_driven" else "tab:blue" for name in names],
    )
    axes[0].set_ylabel("Full-vector success (%)")
    axes[1].bar(
        np.arange(len(names)),
        [100.0 * float(row["worst_target_parameter_error_p95"]) for row in strategy_summary],
        color=["tab:orange" if name == "uncertainty_driven" else "tab:blue" for name in names],
    )
    axes[1].set_ylabel("Worst-target p95 RMS error (%)")
    for axis in axes:
        axis.set_xticks(np.arange(len(names)), [name.replace("_", " ") for name in names], rotation=35, ha="right")
    figures["fixed_vs_adaptive"] = _save(figure, output_directory, "fixed_vs_adaptive")

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    final = _final_rows(trial_rows)
    for strategy in sorted({str(row["strategy"]) for row in final}):
        values = [row for row in final if row["strategy"] == strategy]
        axis.scatter(
            [row["cumulative_absolute_input_energy_j"] for row in values],
            [100.0 * row["parameter_relative_error_rms"] for row in values],
            alpha=0.35,
            s=18,
            label=strategy.replace("_", " "),
        )
    axis.set(xlabel="Cumulative absolute input energy (J)", ylabel="Full-vector RMS relative error (%)")
    axis.set_yscale("log")
    axis.legend(fontsize=7)
    figures["cumulative_energy_vs_accuracy"] = _save(
        figure, output_directory, "cumulative_energy_vs_accuracy"
    )

    selected = [
        row
        for row in selection_rows
        if bool(row["selected"]) and np.isfinite(float(row["realized_weak_error_reduction"]))
    ]
    selected.sort(key=lambda row: float(row["realized_weak_error_reduction"]), reverse=True)
    examples = selected[: min(12, len(selected))]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    if examples:
        before = [100.0 * float(row["current_weak_abs_relative_error_evaluation_only"]) for row in examples]
        after = [
            100.0
            * (
                float(row["current_weak_abs_relative_error_evaluation_only"])
                - float(row["realized_weak_error_reduction"])
            )
            for row in examples
        ]
        positions = np.arange(len(examples))
        width = 0.38
        axis.bar(positions - width / 2, before, width, label="before selected probe")
        axis.bar(positions + width / 2, after, width, label="after selected probe")
        labels = [
            f"{row['target']}\n{row['weak_parameter']} → {row['candidate_probe']}"
            for row in examples
        ]
        axis.set_xticks(positions, labels, rotation=40, ha="right", fontsize=7)
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No completed adaptive follow-up probes", ha="center", va="center")
    axis.set_ylabel("Weak-parameter absolute relative error (%)")
    figures["second_probe_resolution_examples"] = _save(
        figure, output_directory, "second_probe_resolution_examples"
    )

    probe_names = sorted({str(row["probe_name"]) for row in frequency_information})
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    for axis, parameter in zip(axes, PARAMETERS):
        values = [
            np.median(
                [
                    float(row[f"{parameter}_information_per_force_dose"])
                    for row in frequency_information
                    if row["probe_name"] == probe
                ]
            )
            for probe in probe_names
        ]
        axis.bar(np.arange(len(probe_names)), values)
        axis.set_yscale("log")
        axis.set_title(LABELS[parameter])
        axis.set_xticks(np.arange(len(probe_names)), probe_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Median residualized information / command dose")
    figures["parameter_information_by_probe"] = _save(
        figure, output_directory, "parameter_information_by_probe"
    )
    return figures
