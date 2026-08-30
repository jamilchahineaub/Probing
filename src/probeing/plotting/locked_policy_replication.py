"""Figures for the locked EXP-0006 policy replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(figure: plt.Figure, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png, pdf = directory / f"{name}.png", directory / f"{name}.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _binary_confusion(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for row in rows:
        actual = 0 if row["actual_binary_class"] == "SAFE" else 1
        predicted = 0 if row["predicted_binary_class"] == "SAFE" else 1
        matrix[actual, predicted] += 1
    return matrix


def _draw_matrix(axis: plt.Axes, matrix: np.ndarray, title: str) -> None:
    axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13)
    axis.set_xticks((0, 1), ("SAFE", "NON-SAFE"))
    axis.set_yticks((0, 1), ("SAFE", "NON-SAFE"))
    axis.set(xlabel="Predicted", ylabel="Actual", title=title)


def plot_locked_policy_replication(
    *,
    rows: Sequence[Mapping[str, Any]],
    binary_summary: Sequence[Mapping[str, Any]],
    secondary_summary: Sequence[Mapping[str, Any]],
    confidence_summary: Sequence[Mapping[str, Any]],
    comparison_summary: Sequence[Mapping[str, Any]],
    margin_summary: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    representative_raw: Mapping[str, np.ndarray],
    false_safe_raw: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    plt.style.use("default")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})
    figures: dict[str, tuple[Path, Path]] = {}

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for axis, noise in zip(axes, ("nominal", "high")):
        _draw_matrix(
            axis,
            _binary_confusion([row for row in rows if row["noise_regime"] == noise]),
            f"{noise.title()} noise",
        )
    figure.suptitle("EXP-0007 locked-policy SAFE versus NON-SAFE decisions")
    figures["locked_binary_confusion"] = _save(figure, output_directory, "locked_binary_confusion")

    summary_rows = [row for row in binary_summary if row["stratum"] == "overall"]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(3)
    width = 0.25
    for offset, noise in zip((-width, 0.0, width), ("low", "nominal", "high")):
        row = next(item for item in summary_rows if item["noise_regime"] == noise)
        axis.bar(x[0] + offset, 100.0 * row["false_safe_rate"], width, color="tab:red")
        axis.errorbar(
            x[0] + offset,
            100.0 * row["false_safe_rate"],
            yerr=[[100.0 * row["false_safe_rate"]], [100.0 * (row["false_safe_ci95_upper"] - row["false_safe_rate"])]],
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.scatter(x[1] + offset, 100.0 * row["false_safe_one_sided95_upper"], color="black", s=18)
    axis.axhline(1.0, color="tab:orange", linestyle="--", label="nominal limit")
    axis.axhline(2.0, color="tab:purple", linestyle=":", label="high-noise limit")
    axis.set_xticks(x, ("Observed rate", "One-sided 95% upper", ""))
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylabel("False-safe probability (%)")
    axis.set_title("False-safe rate with binomial confidence bounds")
    axis.legend()
    figures["false_safe_confidence_intervals"] = _save(figure, output_directory, "false_safe_confidence_intervals")

    colors = {"SAFE": "tab:green", "CAUTION": "tab:orange", "UNSAFE": "tab:red"}
    high_rows = [row for row in rows if row["noise_regime"] == "high"]
    figure, axis = plt.subplots(figsize=(7.7, 5.7))
    for label, color in colors.items():
        subset = [row for row in high_rows if row["actual_risk_class"] == label]
        axis.scatter(
            [np.log10(float(row["true_stiffness_n_per_m"])) for row in subset],
            [np.log10(float(row["true_damping_n_s_per_m"])) for row in subset],
            c=color,
            s=18,
            alpha=0.55,
            label=f"actual {label}",
        )
    errors = [row for row in high_rows if not row["binary_correct"]]
    axis.scatter(
        [np.log10(float(row["true_stiffness_n_per_m"])) for row in errors],
        [np.log10(float(row["true_damping_n_s_per_m"])) for row in errors],
        facecolors="none",
        edgecolors="black",
        s=65,
        linewidths=1.0,
        label="binary error",
    )
    axis.set(xlabel="log10 stiffness (N/m)", ylabel="log10 damping (N s/m)", title="High-noise decisions across target parameter space")
    axis.legend(fontsize=8, ncol=2)
    figures["parameter_space_outcomes"] = _save(figure, output_directory, "parameter_space_outcomes")

    boundary = [row for row in high_rows if row["population_stratum"] == "boundary"]
    figure, axis = plt.subplots(figsize=(7.7, 5.7))
    scatter = axis.scatter(
        [np.log10(float(row["true_stiffness_n_per_m"])) for row in boundary],
        [np.log10(float(row["true_effective_mass_kg"])) for row in boundary],
        c=[float(row["actual_risk_score"]) for row in boundary],
        cmap="coolwarm",
        vmin=0.7,
        vmax=1.4,
        s=25,
        alpha=0.75,
    )
    axis.scatter(
        [np.log10(float(row["true_stiffness_n_per_m"])) for row in boundary if row["predicted_binary_class"] == "SAFE"],
        [np.log10(float(row["true_effective_mass_kg"])) for row in boundary if row["predicted_binary_class"] == "SAFE"],
        facecolors="none",
        edgecolors="black",
        s=65,
        label="predicted SAFE",
    )
    figure.colorbar(scatter, ax=axis, label="Actual severity / SAFE envelope")
    axis.set(xlabel="log10 stiffness (N/m)", ylabel="log10 effective mass (kg)", title="Boundary-enriched high-noise decision map")
    axis.legend()
    figures["decision_boundary_heatmap"] = _save(figure, output_directory, "decision_boundary_heatmap")

    def _raw_plot(raw: Mapping[str, np.ndarray], name: str, title: str) -> None:
        figure, axis = plt.subplots(figsize=(9.0, 4.8))
        ids = np.unique(raw["target_id"].astype(str))
        for target_id in ids:
            mask = raw["target_id"].astype(str) == target_id
            axis.plot(raw["time_s"][mask], 1e3 * raw["measured_displacement_m"][mask], linewidth=0.8, label=target_id)
        axis.axvline(3.0, color="black", linestyle="--", label="chirp end")
        axis.axvline(3.5, color="tab:purple", linestyle=":", label="decision")
        axis.set(xlabel="Time (s)", ylabel="Measured displacement (mm)", title=title)
        axis.legend(fontsize=7, ncol=2)
        figures[name] = _save(figure, output_directory, name)

    _raw_plot(representative_raw, "representative_rejected_targets", "Representative high-noise correctly rejected targets")
    audit_title = "No false-safe case observed; nearest non-SAFE rejections shown" if summary["false_safe_cases_high_noise"] == 0 else "Observed false-safe cases (full response audit)"
    _raw_plot(false_safe_raw, "false_safe_case_audit", audit_title)

    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    labels = ("EXP-0006\nreference", "EXP-0007\nnominal", "EXP-0007\nhigh")
    accuracies = [
        100.0 * float(next(row for row in comparison_summary if row["experiment"] == "EXP-0006" and row["noise_regime"] == "nominal")["binary_accuracy"]),
        100.0 * float(next(row for row in comparison_summary if row["experiment"] == "EXP-0007" and row["noise_regime"] == "nominal")["binary_accuracy"]),
        100.0 * float(next(row for row in comparison_summary if row["experiment"] == "EXP-0007" and row["noise_regime"] == "high")["binary_accuracy"]),
    ]
    axis.bar(np.arange(3), accuracies, color=("tab:gray", "tab:blue", "tab:purple"))
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylabel("SAFE/NON-SAFE accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Locked EXP-0006 reference versus independent EXP-0007")
    figures["exp0006_vs_exp0007"] = _save(figure, output_directory, "exp0006_vs_exp0007")

    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    for noise in ("low", "nominal", "high"):
        subset = [row for row in margin_summary if row["noise_regime"] == noise]
        axis.plot(
            [0.5 * (float(row["margin_low"]) + min(float(row["margin_high"]), 1.0)) for row in subset],
            [100.0 * float(row["binary_accuracy"]) for row in subset],
            marker="o",
            label=noise,
        )
    axis.set(xlabel="Decision-margin bin midpoint", ylabel="Binary accuracy (%)", title="Confidence margin behavior")
    axis.legend()
    figures["confidence_margin_behavior"] = _save(figure, output_directory, "confidence_margin_behavior")
    return figures
