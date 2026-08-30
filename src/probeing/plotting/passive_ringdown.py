"""Publication-style EXP-0006 passive ring-down figures."""

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


def _confusion(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    for row in rows:
        matrix[
            RISK_LABELS.index(str(row["actual_risk_class"])),
            RISK_LABELS.index(str(row["predicted_risk_class"])),
        ] += 1
    return matrix


def _plot_confusion(axis: plt.Axes, matrix: np.ndarray, title: str) -> None:
    axis.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    axis.set_xticks(np.arange(3), RISK_LABELS)
    axis.set_yticks(np.arange(3), RISK_LABELS)
    axis.set(xlabel="Predicted", ylabel="Actual", title=title)


def plot_passive_ringdown(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    duration_summary: Sequence[Mapping[str, Any]],
    feature_set_summary: Sequence[Mapping[str, Any]],
    early_stop_rows: Sequence[Mapping[str, Any]],
    legacy_audit_rows: Sequence[Mapping[str, Any]],
    representative_raw: Mapping[str, np.ndarray],
    legacy_raw: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    """Create the predeclared EXP-0006 figure set."""

    plt.style.use("default")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})
    figures: dict[str, tuple[Path, Path]] = {}
    selected_duration = float(summary["selected_reporting_duration_s"])
    primary = str(config["predictor"]["primary_feature_set"])
    probe_end = float(config["probe"]["duration_s"])

    labels = np.asarray(representative_raw["risk_class"]).astype(str)
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 8.6), sharex=True)
    for axis, label in zip(axes, RISK_LABELS):
        mask = labels == label
        time = np.asarray(representative_raw["time_s"], dtype=float)[mask]
        axis.plot(
            time,
            1.0e3 * np.asarray(representative_raw["true_displacement_m"], dtype=float)[mask],
            color=COLORS[label],
            linewidth=1.5,
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
        axis.axvline(probe_end, color="tab:blue", linestyle="--", linewidth=1.0)
        axis.axvline(probe_end + selected_duration, color="tab:purple", linestyle=":", linewidth=1.2)
        axis.set_ylabel(f"{label}\nDisplacement (mm)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Fixed chirp followed by zero-force causal observation")
    figures["representative_chirp_ringdown"] = _save(
        figure, output_directory, "representative_chirp_ringdown"
    )

    legacy_ids = [str(value) for value in config["legacy_exp_0005_false_safe_audit"]["target_ids"]]
    legacy_labels = np.asarray(legacy_raw["target_id"]).astype(str)
    figure, axes = plt.subplots(len(legacy_ids), 1, figsize=(10.0, 11.5), sharex=True)
    for axis, target_id in zip(axes, legacy_ids):
        mask = legacy_labels == target_id
        time = np.asarray(legacy_raw["time_s"], dtype=float)[mask]
        displacement = 1.0e3 * np.asarray(legacy_raw["measured_displacement_m"], dtype=float)[mask]
        axis.plot(time, displacement, color="tab:blue", linewidth=0.9)
        axis.axvline(probe_end, color="black", linestyle="--", linewidth=0.9)
        predictions = sorted(
            [
                row
                for row in legacy_audit_rows
                if row["target_id"] == target_id
                and row["noise_regime"] == "nominal"
                and row["feature_set"] == primary
            ],
            key=lambda row: float(row["observation_duration_s"]),
        )
        y_marker = float(np.max(displacement)) if displacement.size else 0.0
        for row in predictions:
            duration = float(row["observation_duration_s"])
            if duration <= 0.0:
                continue
            label = str(row["predicted_risk_class"])
            axis.scatter(probe_end + duration, y_marker, color=COLORS[label], s=22, zorder=3)
        actual = str(predictions[0]["actual_risk_class"])
        axis.set_ylabel(f"{target_id.split('_')[-1]}\nmm")
        axis.text(0.99, 0.84, f"actual {actual}", transform=axis.transAxes, ha="right", color=COLORS[actual])
    axes[-1].set_xlabel("Time (s); decision markers: green SAFE, orange CAUTION, red UNSAFE")
    figure.suptitle("Locked EXP-0005 false-safe audit under passive observation")
    figures["exp0005_false_safe_ringdown_audit"] = _save(
        figure, output_directory, "exp0005_false_safe_ringdown_audit"
    )

    selected_rows = [
        row
        for row in validation_rows
        if row["feature_set"] == primary
        and row["noise_regime"] == "nominal"
        and float(row["observation_duration_s"]) == selected_duration
    ]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for label in RISK_LABELS:
        subset = [row for row in selected_rows if row["actual_risk_class"] == label]
        axis.scatter(
            [row["actual_hold_settling_time_s"] for row in subset],
            [row["predicted_hold_settling_time_s"] for row in subset],
            color=COLORS[label],
            alpha=0.6,
            s=24,
            label=label,
        )
    axis.plot([0.0, 3.0], [0.0, 3.0], "k--", linewidth=1.0)
    axis.set(
        xlabel="Actual hidden-maneuver settling time (s)",
        ylabel="Probe + ring-down predicted settling time (s)",
        title=f"Settling prediction after {selected_duration:g} s passive observation",
    )
    axis.legend()
    figures["settling_predicted_vs_actual"] = _save(
        figure, output_directory, "settling_predicted_vs_actual"
    )

    durations = sorted({float(row["observation_duration_s"]) for row in duration_summary})
    noises = ("low", "nominal", "high")
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for noise in noises:
        values = sorted(
            [row for row in duration_summary if row["noise_regime"] == noise],
            key=lambda row: float(row["observation_duration_s"]),
        )
        axis.plot(
            [row["observation_duration_s"] for row in values],
            [100.0 * float(row["false_safe_rate"]) for row in values],
            marker="o",
            label=noise,
        )
    axis.axhline(1.0, color="tab:orange", linestyle="--", linewidth=1.0, label="nominal criterion")
    axis.axhline(2.0, color="tab:red", linestyle=":", linewidth=1.0, label="high-noise criterion")
    axis.set(xlabel="Post-probe observation duration (s)", ylabel="False-safe rate (%)", title="Primary safety metric versus passive wait time")
    axis.set_xticks(durations)
    axis.legend()
    figures["false_safe_vs_observation_duration"] = _save(
        figure, output_directory, "false_safe_vs_observation_duration"
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for noise in noises:
        values = sorted(
            [row for row in duration_summary if row["noise_regime"] == noise],
            key=lambda row: float(row["observation_duration_s"]),
        )
        axis.plot(
            [row["observation_duration_s"] for row in values],
            [100.0 * float(row["accuracy"]) for row in values],
            marker="o",
            label=noise,
        )
    axis.set(xlabel="Post-probe observation duration (s)", ylabel="Three-class accuracy (%)", title="Accuracy is secondary to false-safe risk")
    axis.set_xticks(durations)
    axis.legend()
    figures["accuracy_vs_observation_duration"] = _save(
        figure, output_directory, "accuracy_vs_observation_duration"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for noise in noises:
        values = sorted(
            [row for row in duration_summary if row["noise_regime"] == noise],
            key=lambda row: float(row["observation_duration_s"]),
        )
        axes[0].plot(
            [row["observation_duration_s"] for row in values],
            [row["median_decision_margin"] for row in values],
            marker="o",
            label=noise,
        )
    axes[0].set(xlabel="Observation duration (s)", ylabel="Median response-bound margin", title="Decision confidence versus wait time")
    axes[0].legend()
    for noise in noises:
        selected = [row for row in early_stop_rows if row["noise_regime"] == noise]
        stop = np.asarray([row["early_stop_duration_s"] for row in selected], dtype=float)
        axes[1].plot(
            durations,
            [100.0 * float(np.mean(stop <= duration)) for duration in durations],
            marker="o",
            label=noise,
        )
    axes[1].set(xlabel="Observation duration (s)", ylabel="Early-stop decisions completed (%)", title="Causal stopping completion")
    axes[1].legend()
    figures["confidence_and_early_stopping"] = _save(
        figure, output_directory, "confidence_and_early_stopping"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    feature_fields = (
        ("rd_decay_rate_log", "log(1 + decay rate)"),
        ("rd_decay_fit_r2", "Envelope decay fit $R^2$"),
        ("rd_energy_ratio_log", "Log residual/initial energy proxy"),
    )
    for axis, (field, label) in zip(axes, feature_fields):
        values = [
            [float(row[field]) for row in selected_rows if row["actual_risk_class"] == risk]
            for risk in RISK_LABELS
        ]
        box = axis.boxplot(values, labels=RISK_LABELS, patch_artist=True)
        for patch_box, risk in zip(box["boxes"], RISK_LABELS):
            patch_box.set_facecolor(COLORS[risk])
            patch_box.set_alpha(0.45)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Passive decay features by hidden future risk class")
    figures["decay_features_by_risk"] = _save(
        figure, output_directory, "decay_features_by_risk"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    selected_duration_rows = [
        row for row in duration_summary if float(row["observation_duration_s"]) == selected_duration
    ]
    axes[0].bar(
        np.arange(3) - 0.18,
        [100.0 * next(row for row in selected_duration_rows if row["noise_regime"] == noise)["accuracy"] for noise in noises],
        0.36,
        label="accuracy",
    )
    axes[0].bar(
        np.arange(3) + 0.18,
        [100.0 * next(row for row in selected_duration_rows if row["noise_regime"] == noise)["false_safe_rate"] for noise in noises],
        0.36,
        label="false-safe",
    )
    axes[0].set_xticks(np.arange(3), noises)
    axes[0].set(ylabel="Rate (%)", title=f"Noise robustness at {selected_duration:g} s")
    axes[0].legend()
    feature_sets = ("chirp_only", "ringdown_only", "chirp_ringdown", "physical_chirp_ringdown")
    comparison = [
        next(
            row
            for row in feature_set_summary
            if row["noise_regime"] == "nominal"
            and float(row["observation_duration_s"]) == selected_duration
            and row["feature_set"] == feature_set
        )
        for feature_set in feature_sets
    ]
    axes[1].bar(
        np.arange(4),
        [100.0 * row["accuracy"] for row in comparison],
        color=["tab:orange" if name == primary else "tab:blue" for name in feature_sets],
    )
    axes[1].set_xticks(np.arange(4), [name.replace("_", " ") for name in feature_sets], rotation=30, ha="right")
    axes[1].set(ylabel="Nominal accuracy (%)", title="Feature-set comparison")
    figures["noise_and_feature_set_performance"] = _save(
        figure, output_directory, "noise_and_feature_set_performance"
    )

    chirp_rows = [
        row
        for row in validation_rows
        if row["feature_set"] == "chirp_only"
        and row["noise_regime"] == "nominal"
        and float(row["observation_duration_s"]) == 0.0
    ]
    ringdown_rows = selected_rows
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _plot_confusion(axes[0], _confusion(chirp_rows), "Chirp features only")
    _plot_confusion(
        axes[1],
        _confusion(ringdown_rows),
        f"Chirp + ring-down ({selected_duration:g} s)",
    )
    figure.suptitle("Nominal confusion matrices at matched probe energy")
    figures["chirp_vs_ringdown_confusion"] = _save(
        figure, output_directory, "chirp_vs_ringdown_confusion"
    )
    return figures
