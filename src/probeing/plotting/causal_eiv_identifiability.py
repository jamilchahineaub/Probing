"""Diagnostic figures for EXP-0003."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

_cache = Path(tempfile.gettempdir()) / "probeing-matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


PARAMETERS = (
    ("stiffness", "k"),
    ("damping", "c"),
    ("effective_mass", "m_eff"),
)


def _save(figure: Any, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise FileExistsError(f"refusing to overwrite figure at {base}")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def plot_causal_eiv_identifiability(
    *,
    timing_aggregate: Sequence[Mapping[str, Any]],
    timing_profile_aggregate: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    representative_raw: Mapping[str, NDArray[Any]],
    summary: Mapping[str, Any],
    output_directory: Path,
) -> Mapping[str, tuple[Path, Path]]:
    outputs: dict[str, tuple[Path, Path]] = {}

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharey=True)
    channels = ("displacement", "acceleration", "force", "filter_group_delay")
    for axis, channel in zip(axes.flat, channels):
        rows = [row for row in timing_aggregate if row["sweep_channel"] == channel]
        offsets = sorted({float(row["timing_offset_ms"]) for row in rows})
        for parameter, label in PARAMETERS:
            worst_bias = []
            for offset in offsets:
                subset = [row for row in rows if np.isclose(float(row["timing_offset_ms"]), offset)]
                worst_bias.append(
                    100.0 * max(abs(float(row[f"{parameter}_relative_bias"])) for row in subset)
                )
            axis.plot(offsets, worst_bias, marker="o", label=label)
        axis.axvline(0.0, color="black", lw=0.8)
        axis.set_title(channel.replace("_", " "))
        axis.set_xlabel("relative timing offset [ms]")
        axis.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("worst-target absolute bias [%]")
    axes[1, 0].set_ylabel("worst-target absolute bias [%]")
    axes[0, 0].legend()
    figure.suptitle("Parameter bias sensitivity to fractional and integer-sample timing error")
    figure.tight_layout()
    outputs["timing_bias"] = _save(figure, output_directory / "exp_0003_timing_bias")

    baseline = "baseline_0p5_10_hz"
    ols_baseline = [
        row for row in aggregate if row["frequency_band"] == baseline and row["estimator"] == "ols"
    ]
    pipelines = list(dict.fromkeys(str(row["pipeline"]) for row in ols_baseline))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for pipeline in pipelines:
        rows = [row for row in ols_baseline if row["pipeline"] == pipeline]
        score = _median(
            [
                {
                    "score": np.sqrt(
                        np.mean(
                            [
                                row["stiffness_p95_abs_relative_error"] ** 2,
                                row["damping_p95_abs_relative_error"] ** 2,
                                row["effective_mass_p95_abs_relative_error"] ** 2,
                            ]
                        )
                    )
                }
                for row in rows
            ],
            "score",
        )
        delay_ms = 1000.0 * _median(rows, "effective_acceleration_delay_s")
        attenuation = _median(rows, "acceleration_noise_attenuation_db")
        cost = _median(rows, "computational_cost_units_per_sample")
        axes[0].scatter(delay_ms, 100.0 * score, s=35.0 + cost, label=pipeline)
        axes[1].scatter(cost, attenuation, s=65.0, label=pipeline)
    axes[0].set_xlabel("measured acceleration delay [ms]")
    axes[0].set_ylabel("median target-wise p95 parameter score [%]")
    axes[0].set_yscale("log")
    axes[0].set_title("Accuracy versus effective delay")
    axes[1].set_xlabel("estimated scalar operation units / sample")
    axes[1].set_ylabel("acceleration-noise attenuation [dB]")
    axes[1].set_title("Noise attenuation versus computational cost")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Causal filter tradeoffs; centered SG is the offline reference")
    figure.tight_layout()
    outputs["causal_filter_tradeoff"] = _save(
        figure, output_directory / "exp_0003_causal_filter_tradeoff"
    )

    estimators = ("ols", "tls", "iv")
    causal_baseline = [
        row
        for row in trials
        if row["frequency_band"] == baseline and bool(row["pipeline_is_causal"])
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    for axis, (parameter, label) in zip(axes, PARAMETERS):
        data = [
            [
                100.0 * abs(float(row[f"{parameter}_relative_error"]))
                for row in causal_baseline
                if row["estimator"] == estimator and np.isfinite(row[f"{parameter}_relative_error"])
            ]
            for estimator in estimators
        ]
        axis.boxplot(data, labels=[name.upper() for name in estimators], showfliers=False)
        axis.set_yscale("log")
        axis.set_ylabel(f"|{label} error| [%]")
        axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.suptitle("Held-out errors-in-variables estimator comparison")
    figure.tight_layout()
    outputs["estimator_comparison"] = _save(
        figure, output_directory / "exp_0003_estimator_comparison"
    )

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    colors = {"ols": "tab:blue", "tls": "tab:orange", "iv": "tab:green"}
    for estimator in estimators:
        rows = [row for row in aggregate if row["estimator"] == estimator and bool(row["pipeline_is_causal"])]
        axis.scatter(
            [row["acceleration_regressor_rms"] for row in rows],
            [100.0 * row["effective_mass_p95_abs_relative_error"] for row in rows],
            alpha=0.55,
            s=28,
            color=colors[estimator],
            label=estimator.upper(),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("true acceleration-regressor RMS [m/s²]")
    axis.set_ylabel("95th-percentile |m_eff error| [%]")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    axis.set_title("Effective-mass error versus inertial excitation strength")
    outputs["mass_vs_acceleration_strength"] = _save(
        figure, output_directory / "exp_0003_mass_vs_acceleration_strength"
    )

    bands = list(dict.fromkeys(str(row["frequency_band"]) for row in aggregate))
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    geometry_rows = [
        row
        for row in aggregate
        if row["pipeline"] == summary["best_causal_pipeline"] and row["estimator"] == "ols"
    ]
    minimum_singular, condition, correlation = [], [], []
    for band in bands:
        rows = [row for row in geometry_rows if row["frequency_band"] == band]
        minimum_singular.append(_median(rows, "normalized_singular_value_3") if rows and "normalized_singular_value_3" in rows[0] else np.nan)
        condition.append(_median(rows, "normalized_condition_number"))
        correlation.append(_median(rows, "maximum_abs_parameter_correlation"))
    # The aggregate stores condition/correlation; recompute the singular summary from trials.
    minimum_singular = [
        _median(
            [
                row
                for row in trials
                if row["frequency_band"] == band
                and row["pipeline"] == summary["best_causal_pipeline"]
                and row["estimator"] == "ols"
            ],
            "normalized_singular_value_3",
        )
        for band in bands
    ]
    axes[0].bar(bands, minimum_singular, color="tab:green")
    axes[0].set_ylabel("median min normalized singular")
    axes[1].bar(bands, condition, color="tab:purple")
    axes[1].set_ylabel("median condition number")
    axes[1].set_yscale("log")
    axes[2].bar(bands, correlation, color="tab:brown")
    axes[2].set_ylabel("median max |parameter corr.| ")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Regressor singular values, conditioning, and parameter correlation")
    figure.tight_layout()
    outputs["regressor_geometry"] = _save(
        figure, output_directory / "exp_0003_regressor_geometry"
    )

    best_pipeline = str(summary["best_causal_pipeline"])
    best_estimator = str(summary["best_estimator_by_median_cross_case_score"])
    selected = [
        row
        for row in aggregate
        if row["pipeline"] == best_pipeline and row["estimator"] == best_estimator
    ]
    figure, axis = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(bands), dtype=float)
    width = 0.24
    for index, (parameter, label) in enumerate(PARAMETERS):
        values = [
            100.0
            * max(
                row[f"{parameter}_p95_abs_relative_error"]
                for row in selected
                if row["frequency_band"] == band
            )
            for band in bands
        ]
        axis.bar(x + (index - 1) * width, values, width=width, label=label)
    axis.set_xticks(x, bands, rotation=20)
    axis.set_yscale("log")
    axis.set_ylabel("worst-target 95th-percentile error [%]")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    axis.set_title(f"Chirp band sensitivity — {best_pipeline} + {best_estimator.upper()}")
    outputs["frequency_band_errors"] = _save(
        figure, output_directory / "exp_0003_frequency_band_errors"
    )

    pareto_rows = [
        row for row in aggregate if row["pipeline"] == best_pipeline and row["estimator"] == "ols"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for band in bands:
        rows = [row for row in pareto_rows if row["frequency_band"] == band]
        axes[0].scatter(
            [row["peak_target_displacement_m"] * 1000.0 for row in rows],
            [row["relative_fisher_information_geometric_mean"] for row in rows],
            label=band,
            alpha=0.75,
        )
        axes[1].scatter(
            [row["absolute_input_energy_j"] for row in rows],
            [row["relative_fisher_information_geometric_mean"] for row in rows],
            label=band,
            alpha=0.75,
        )
    axes[0].set_xlabel("peak target displacement [mm]")
    axes[1].set_xlabel("absolute input energy [J]")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("relative regression/Fisher information")
        axis.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle("Information versus structural disturbance (fixed 1 N force bound)")
    figure.tight_layout()
    outputs["information_disturbance_pareto"] = _save(
        figure, output_directory / "exp_0003_information_disturbance_pareto"
    )

    parameterization_rows = [
        row
        for row in trials
        if row["pipeline"] == best_pipeline and row["estimator"] == "ols"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    modal = (
        ("natural_frequency", "omega_n"),
        ("damping_ratio", "zeta"),
        ("inverse_mass", "1/m_eff"),
    )
    for axis, (parameter, label) in zip(axes, modal):
        direct = [
            100.0 * abs(float(row[f"direct_{parameter}_relative_error"]))
            for row in parameterization_rows
            if np.isfinite(row[f"direct_{parameter}_relative_error"])
        ]
        ratio = [
            100.0 * abs(float(row[f"ratio_{parameter}_relative_error"]))
            for row in parameterization_rows
            if np.isfinite(row[f"ratio_{parameter}_relative_error"])
        ]
        axis.boxplot([direct, ratio], labels=["from k,c,m", "ratio form"], showfliers=False)
        axis.set_yscale("log")
        axis.set_ylabel(f"|{label} error| [%]")
        axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.suptitle("Direct versus modal/scale parameterization")
    figure.tight_layout()
    outputs["parameterization_comparison"] = _save(
        figure, output_directory / "exp_0003_parameterization_comparison"
    )

    raw = representative_raw
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    for pipeline in list(dict.fromkeys(raw["pipeline"].astype(str))):
        mask = raw["pipeline"] == pipeline
        time = raw["time_s"][mask].astype(float)
        axes[0].plot(time, 1000.0 * raw["processed_displacement_m"][mask].astype(float), label=pipeline)
        axes[1].plot(time, raw["processed_acceleration_m_per_s2"][mask].astype(float), label=pipeline)
        axes[2].plot(time, raw["processed_force_n"][mask].astype(float), label=pipeline)
    axes[0].set_ylabel("processed x [mm]")
    axes[1].set_ylabel("processed acceleration [m/s²]")
    axes[2].set_ylabel("processed force [N]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Representative causal estimates and centered offline reference")
    figure.tight_layout()
    outputs["representative_causal_signals"] = _save(
        figure, output_directory / "exp_0003_representative_causal_signals"
    )

    return outputs
