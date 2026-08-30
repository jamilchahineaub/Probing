"""Publication-style figures for EXP-0005 decision sufficiency."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


RISK_LABELS = ("SAFE", "CAUTION", "UNSAFE")
COLORS = {"SAFE": "tab:green", "CAUTION": "tab:orange", "UNSAFE": "tab:red"}


def _save(figure: plt.Figure, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    pdf = directory / f"{name}.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _primary(rows: Sequence[Mapping[str, Any]], noise: str = "nominal") -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["predictor"] == "outcome_bound"
        and row["feature_set"] == "combined_task"
        and row["noise_regime"] == noise
    ]


def _classification_row(
    rows: Sequence[Mapping[str, Any]], predictor: str, feature_set: str, noise: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["predictor"] == predictor
        and row["feature_set"] == feature_set
        and row["noise_regime"] == noise
    )


def plot_decision_sufficiency(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    classification_summary: Sequence[Mapping[str, Any]],
    dynamics_performance: Sequence[Mapping[str, Any]],
    feature_space_rows: Sequence[Mapping[str, Any]],
    representative_raw: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    """Create the predeclared EXP-0005 figure set."""

    plt.style.use("default")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})
    figures: dict[str, tuple[Path, Path]] = {}
    primary = _primary(validation_rows)

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.6), sharex=True)
    raw_labels = np.asarray(representative_raw["risk_class"]).astype(str)
    for axis, label in zip(axes, RISK_LABELS):
        mask = raw_labels == label
        time = np.asarray(representative_raw["time_s"], dtype=float)[mask]
        axis.plot(
            time,
            1.0e3 * np.asarray(representative_raw["true_displacement_m"], dtype=float)[mask],
            color=COLORS[label],
            linewidth=1.6,
            label="true displacement",
        )
        axis.plot(
            time,
            1.0e3 * np.asarray(representative_raw["measured_displacement_m"], dtype=float)[mask],
            color="black",
            alpha=0.45,
            linewidth=0.8,
            label="causal sensed displacement",
        )
        force_axis = axis.twinx()
        force_axis.plot(
            time,
            np.asarray(representative_raw["true_force_n"], dtype=float)[mask],
            color="tab:blue",
            alpha=0.45,
            linewidth=0.8,
        )
        force_axis.set_ylabel("Force (N)", color="tab:blue")
        axis.set_ylabel(f"{label}\nDisplacement (mm)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Probe time (s)")
    figure.suptitle("One fixed 0.5–5 Hz chirp: representative causal responses")
    figures["representative_probe_response"] = _save(
        figure, output_directory, "representative_probe_response"
    )

    outcomes = (
        ("peak_displacement_m", "Peak displacement (mm)", 1.0e3),
        ("hold_settling_time_s", "Hold settling time (s)", 1.0),
        ("dominant_response_frequency_hz", "Dominant ringdown frequency (Hz)", 1.0),
        ("disturbance_limited_safe_force_n", "Disturbance-limited force (N)", 1.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
    for axis, (field, label, scale) in zip(axes.flat, outcomes):
        actual = scale * np.asarray([row[f"actual_{field}"] for row in primary], dtype=float)
        predicted = scale * np.asarray([row[f"predicted_{field}"] for row in primary], dtype=float)
        for risk in RISK_LABELS:
            mask = np.asarray([row["actual_risk_class"] == risk for row in primary])
            axis.scatter(actual[mask], predicted[mask], s=18, alpha=0.6, color=COLORS[risk], label=risk)
        limits = [min(float(np.min(actual)), float(np.min(predicted))), max(float(np.max(actual)), float(np.max(predicted)))]
        axis.plot(limits, limits, "k--", linewidth=1.0)
        axis.set(xlabel=f"Actual {label}", ylabel=f"Predicted {label}")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Probe-only predictions versus hidden sustained-contact response")
    figures["quantitative_predictions"] = _save(
        figure, output_directory, "quantitative_predictions"
    )

    confusion = np.zeros((3, 3), dtype=int)
    for row in primary:
        confusion[RISK_LABELS.index(str(row["actual_risk_class"])), RISK_LABELS.index(str(row["predicted_risk_class"]))] += 1
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    image = axes[0].imshow(confusion, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(confusion[i, j]), ha="center", va="center")
    axes[0].set_xticks(np.arange(3), RISK_LABELS)
    axes[0].set_yticks(np.arange(3), RISK_LABELS)
    axes[0].set(xlabel="Predicted", ylabel="Actual", title="Nominal causal confusion matrix")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    false_safe = [row for row in primary if bool(row["false_safe"])]
    if false_safe:
        axes[1].scatter(
            [row["actual_risk_score"] for row in false_safe],
            [row["predicted_risk_score"] for row in false_safe],
            color="tab:red",
            s=35,
        )
        axes[1].set(xlabel="Actual severity", ylabel="Predicted upper-bound severity")
    else:
        axes[1].text(0.5, 0.56, "0 false-safe cases", ha="center", va="center", fontsize=18, color="tab:green")
        axes[1].text(0.5, 0.42, "on nominal untouched validation", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Highest-priority failure audit")
    figures["confusion_and_false_safe"] = _save(
        figure, output_directory, "confusion_and_false_safe"
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for label in RISK_LABELS:
        selected = [row for row in primary if row["actual_risk_class"] == label]
        axis.scatter(
            [row["actual_risk_score"] for row in selected],
            [row["predicted_risk_score"] for row in selected],
            color=COLORS[label],
            alpha=0.6,
            s=24,
            label=label,
        )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(xlabel="Actual interaction severity / SAFE envelope", ylabel="Predicted conservative severity / SAFE envelope")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.legend()
    figures["risk_score_vs_actual_severity"] = _save(
        figure, output_directory, "risk_score_vs_actual_severity"
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for label in RISK_LABELS:
        selected = [row for row in feature_space_rows if row["actual_risk_class"] == label]
        axis.scatter(
            [row["component_1"] for row in selected],
            [row["component_2"] for row in selected],
            color=COLORS[label],
            alpha=0.6,
            s=24,
            label=label,
        )
    axis.set(xlabel="Combined task-feature component 1", ylabel="Component 2", title="Probe feature space (visualization only)")
    axis.legend()
    figures["feature_space"] = _save(figure, output_directory, "feature_space")

    noise_names = [str(row["name"]) for row in config["sensing"]["noise_regimes"]]
    feature_sets = ("full_parameters", "stiffness_only", "frequency_response", "time_domain", "combined_task")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    primary_summary = [
        _classification_row(classification_summary, "outcome_bound", "combined_task", noise)
        for noise in noise_names
    ]
    axes[0].plot(noise_names, [100.0 * row["accuracy"] for row in primary_summary], marker="o", label="accuracy")
    axes[0].plot(noise_names, [100.0 * row["false_safe_rate"] for row in primary_summary], marker="o", label="false-safe rate")
    axes[0].set(ylabel="Rate (%)", xlabel="Sensor-noise regime", title="Causal sensing robustness")
    axes[0].legend()
    positions = np.arange(len(feature_sets))
    nominal_rows = [
        _classification_row(classification_summary, "outcome_bound", feature_set, "nominal")
        for feature_set in feature_sets
    ]
    axes[1].bar(positions, [100.0 * row["accuracy"] for row in nominal_rows], color=["tab:orange" if name == "combined_task" else "tab:blue" for name in feature_sets])
    axes[1].set_xticks(positions, [name.replace("_", " ") for name in feature_sets], rotation=35, ha="right")
    axes[1].set(ylabel="Three-class accuracy (%)", title="Do explicit physical parameters help?")
    figures["sensing_and_feature_set_performance"] = _save(
        figure, output_directory, "sensing_and_feature_set_performance"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for axis, parameter in zip(axes, ("stiffness", "damping", "effective_mass")):
        selected = [row for row in dynamics_performance if row["parameter"] == parameter]
        axis.plot([row["dynamics_bin"] for row in selected], [100.0 * row["accuracy"] for row in selected], marker="o", label="accuracy")
        axis.plot([row["dynamics_bin"] for row in selected], [100.0 * row["false_safe_rate"] for row in selected], marker="o", label="false-safe")
        axis.set_title(parameter.replace("_", " "))
        axis.set_xlabel("True dynamics bin (evaluation only)")
    axes[0].set_ylabel("Rate (%)")
    axes[-1].legend()
    figures["performance_vs_target_dynamics"] = _save(
        figure, output_directory, "performance_vs_target_dynamics"
    )

    mass_error = 100.0 * np.asarray([row["effective_mass_abs_relative_error"] for row in primary])
    vector_error = 100.0 * np.sqrt(
        np.mean(
            np.column_stack(
                [
                    [row["stiffness_abs_relative_error"] for row in primary],
                    [row["damping_abs_relative_error"] for row in primary],
                    [row["effective_mass_abs_relative_error"] for row in primary],
                ]
            )
            ** 2,
            axis=1,
        )
    )
    correct = np.asarray([bool(row["classification_correct"]) for row in primary], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    rng = np.random.default_rng(5)
    axes[0].scatter(mass_error, 100.0 * correct + rng.normal(0.0, 1.2, correct.size), alpha=0.45, s=18)
    axes[0].axvline(30.0, color="tab:red", linestyle="--", label="30% mass error")
    axes[0].set(xlabel=r"$m_{eff}$ diagnostic absolute error (%)", ylabel="Safety decision correct (%, jittered)", title="Decision correctness despite poor mass estimates")
    axes[0].legend(fontsize=8)
    bins = np.quantile(vector_error, [0.0, 0.25, 0.5, 0.75, 1.0])
    bin_accuracy = []
    for index in range(4):
        mask = (vector_error >= bins[index]) & (vector_error <= bins[index + 1] if index == 3 else vector_error < bins[index + 1])
        bin_accuracy.append(100.0 * float(np.mean(correct[mask])))
    axes[1].bar(np.arange(4), bin_accuracy)
    axes[1].set_xticks(np.arange(4), ("best", "Q2", "Q3", "worst"))
    axes[1].set(xlabel="Full [k, c, m] error quartile", ylabel="Safety decision accuracy (%)", title="Full-vector accuracy versus decision accuracy")
    figures["parameter_accuracy_vs_decision_accuracy"] = _save(
        figure, output_directory, "parameter_accuracy_vs_decision_accuracy"
    )
    return figures
