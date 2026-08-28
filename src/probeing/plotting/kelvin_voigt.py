"""Physics-focused figures for Kelvin-Voigt validation runs."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping

_matplotlib_cache = Path(tempfile.gettempdir()) / "probeing-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from probeing.metrics import cumulative_trapezoid


def plot_kelvin_voigt_validation(
    *,
    raw: Mapping[str, NDArray[np.float64]],
    stiffness_n_per_m: float,
    damping_n_s_per_m: float,
    output_base: Path,
) -> tuple[Path, Path]:
    """Plot kinematics, force decomposition, and energy consistency."""

    time = raw["time_s"]
    displacement = raw["displacement_m"]
    velocity = raw["velocity_m_per_s"]
    force = raw["force_n"]

    cumulative_work = cumulative_trapezoid(force * velocity, time)
    stored_energy = 0.5 * stiffness_n_per_m * displacement**2
    cumulative_dissipation = cumulative_trapezoid(
        damping_n_s_per_m * velocity**2, time
    )

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    displacement_axis = axes[0]
    velocity_axis = displacement_axis.twinx()
    displacement_line = displacement_axis.plot(
        time, 1.0e3 * displacement, color="tab:blue", label="indentation"
    )[0]
    velocity_line = velocity_axis.plot(
        time, 1.0e3 * velocity, color="tab:orange", label="velocity"
    )[0]
    displacement_axis.set_ylabel("indentation [mm]")
    velocity_axis.set_ylabel("velocity [mm/s]")
    displacement_axis.grid(True, alpha=0.3)
    displacement_axis.legend(
        [displacement_line, velocity_line],
        [displacement_line.get_label(), velocity_line.get_label()],
        loc="center right",
    )

    axes[1].plot(time, force, color="black", linewidth=2.0, label="model force")
    axes[1].plot(
        time,
        raw["analytical_force_n"],
        color="tab:red",
        linestyle="--",
        label="analytical force",
    )
    axes[1].plot(time, raw["elastic_force_n"], label="elastic component")
    axes[1].plot(time, raw["damping_force_n"], label="damping component")
    axes[1].set_ylabel("reaction force [N]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time, 1.0e3 * cumulative_work, label="input work")
    axes[2].plot(time, 1.0e3 * stored_energy, label="stored elastic energy")
    axes[2].plot(
        time, 1.0e3 * cumulative_dissipation, label="damping dissipation"
    )
    axes[2].plot(
        time,
        1.0e3 * (stored_energy + cumulative_dissipation),
        linestyle="--",
        label="stored + dissipated",
    )
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("energy [mJ]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    figure.suptitle("Kelvin–Voigt response to a bounded raised-cosine ramp")
    figure.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    if png_path.exists() or pdf_path.exists():
        raise FileExistsError(f"refusing to overwrite figure at {output_base}")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path
