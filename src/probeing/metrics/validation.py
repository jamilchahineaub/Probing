"""Numerical validation metrics for reduced-order interaction models."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cumulative_trapezoid(y: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Cumulative trapezoidal integral with an initial value of zero."""

    y_array = np.asarray(y, dtype=float)
    x_array = np.asarray(x, dtype=float)
    if y_array.ndim != 1 or x_array.ndim != 1 or y_array.shape != x_array.shape:
        raise ValueError("x and y must be one-dimensional arrays with equal shape")
    if len(x_array) < 2 or not np.all(np.diff(x_array) > 0.0):
        raise ValueError("x must contain at least two strictly increasing samples")
    increments = 0.5 * (y_array[1:] + y_array[:-1]) * np.diff(x_array)
    return np.concatenate(([0.0], np.cumsum(increments)))


def kelvin_voigt_validation_metrics(
    *,
    time_s: ArrayLike,
    displacement_m: ArrayLike,
    velocity_m_per_s: ArrayLike,
    force_n: ArrayLike,
    analytical_force_n: ArrayLike,
    stiffness_n_per_m: float,
    damping_n_s_per_m: float,
) -> Mapping[str, float | int]:
    """Calculate force error and the bilateral Kelvin-Voigt energy residual."""

    time = np.asarray(time_s, dtype=float)
    displacement = np.asarray(displacement_m, dtype=float)
    velocity = np.asarray(velocity_m_per_s, dtype=float)
    force = np.asarray(force_n, dtype=float)
    analytical = np.asarray(analytical_force_n, dtype=float)
    arrays = (time, displacement, velocity, force, analytical)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("validation inputs must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("validation inputs must have equal shape")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("validation inputs must be finite")

    force_error = force - analytical
    cumulative_work = cumulative_trapezoid(force * velocity, time)
    cumulative_dissipation = cumulative_trapezoid(
        damping_n_s_per_m * velocity**2, time
    )
    stored_energy = 0.5 * stiffness_n_per_m * displacement**2
    energy_residual = (
        cumulative_work
        - (stored_energy - stored_energy[0])
        - cumulative_dissipation
    )
    energy_scale = max(
        abs(float(cumulative_work[-1])),
        abs(float(stored_energy[-1] - stored_energy[0])),
        abs(float(cumulative_dissipation[-1])),
        np.finfo(float).eps,
    )

    return {
        "sample_count": int(time.size),
        "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
        "max_force_error_n": float(np.max(np.abs(force_error))),
        "peak_abs_force_n": float(np.max(np.abs(force))),
        "peak_displacement_m": float(np.max(displacement)),
        "peak_abs_velocity_m_per_s": float(np.max(np.abs(velocity))),
        "final_work_j": float(cumulative_work[-1]),
        "final_stored_energy_j": float(stored_energy[-1]),
        "dissipated_energy_j": float(cumulative_dissipation[-1]),
        "final_energy_balance_error_j": float(energy_residual[-1]),
        "max_abs_energy_balance_error_j": float(np.max(np.abs(energy_residual))),
        "relative_energy_balance_error": float(
            np.max(np.abs(energy_residual)) / energy_scale
        ),
    }

