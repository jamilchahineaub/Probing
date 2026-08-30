"""Simple, transparent errors-in-variables estimators for EXP-0003."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.measurements import SyntheticMeasurements

from .linear import RegressionDiagnostics, regression_diagnostics, regression_matrix


@dataclass(frozen=True)
class LinearEIVResult:
    estimator: str
    parameters: NDArray[np.float64]
    predicted_force_n: NDArray[np.float64]
    residual_n: NDArray[np.float64]
    force_rmse_n: float
    diagnostics: RegressionDiagnostics
    valid: bool
    estimator_diagnostics: Mapping[str, float]


def _result(
    name: str,
    design: NDArray[np.float64],
    force: NDArray[np.float64],
    parameters: NDArray[np.float64],
    estimator_diagnostics: Mapping[str, float],
) -> LinearEIVResult:
    valid = bool(np.all(np.isfinite(parameters)))
    predicted = design @ parameters if valid else np.full_like(force, np.nan)
    residual = force - predicted
    return LinearEIVResult(
        estimator=name,
        parameters=np.asarray(parameters, dtype=float),
        predicted_force_n=predicted,
        residual_n=residual,
        force_rmse_n=float(np.sqrt(np.mean(residual**2))) if valid else float("inf"),
        diagnostics=regression_diagnostics(design),
        valid=valid,
        estimator_diagnostics=dict(estimator_diagnostics),
    )


def ordinary_least_squares_eiv(measurements: SyntheticMeasurements) -> LinearEIVResult:
    """OLS wrapper with the common EXP-0003 result interface."""

    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    parameters, _, _, _ = np.linalg.lstsq(design, force, rcond=None)
    return _result("ols", design, force, parameters, {})


def total_least_squares(measurements: SyntheticMeasurements) -> LinearEIVResult:
    """Column-standardized total least squares.

    TLS minimizes perturbations in both regressors and response.  Columns are
    RMS-standardized first because x, velocity, acceleration, and force use
    different physical units.  This is classical isotropic TLS in standardized
    coordinates, not a claim that the true derived-error covariance is known.
    """

    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    augmented = np.column_stack((design, force))
    scales = np.sqrt(np.mean(augmented**2, axis=0))
    safe = np.where(scales > np.finfo(float).tiny, scales, 1.0)
    _, singular_values, right = np.linalg.svd(augmented / safe, full_matrices=False)
    last = right[-1]
    if abs(last[-1]) <= 1.0e-12:
        parameters = np.full(3, np.nan)
    else:
        standardized = -last[:3] / last[-1]
        parameters = safe[-1] * standardized / safe[:3]
    diagnostics = {
        "augmented_smallest_singular_value": float(singular_values[-1]),
        "augmented_condition_number": float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf"),
    }
    return _result("tls", design, force, parameters, diagnostics)


def delayed_input_instruments(
    input_force_n: ArrayLike,
    time_s: ArrayLike,
    delays_s: ArrayLike,
) -> NDArray[np.float64]:
    """Create instruments from delayed known bounded input-force histories."""

    force = np.asarray(input_force_n, dtype=float)
    time = np.asarray(time_s, dtype=float)
    delays = np.asarray(delays_s, dtype=float)
    if force.shape != time.shape or force.ndim != 1 or delays.ndim != 1:
        raise ValueError("input force/time must align and delays must be one-dimensional")
    if delays.size < 3 or np.any(delays < 0.0) or not np.all(np.isfinite(delays)):
        raise ValueError("at least three finite non-negative instrument delays are required")
    return np.column_stack(
        [np.interp(time - delay, time, force, left=force[0], right=force[-1]) for delay in delays]
    )


def instrumental_variables(
    measurements: SyntheticMeasurements,
    instruments: ArrayLike,
) -> LinearEIVResult:
    """Two-stage least squares using input-derived external instruments.

    Known chirp histories are correlated with the structural response while
    remaining independent of synthetic kinematic sensor noise.  Weak
    instruments are diagnosed explicitly; actuator/contact-model mismatch is
    outside this Stage 1 claim.
    """

    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    instrument = np.asarray(instruments, dtype=float)
    if instrument.ndim != 2 or instrument.shape[0] != design.shape[0] or instrument.shape[1] < 3:
        raise ValueError("instruments must have shape (n, q) with q >= 3")
    if not np.all(np.isfinite(instrument)):
        raise ValueError("instruments must be finite")
    centered = instrument - np.mean(instrument, axis=0)
    scales = np.linalg.norm(centered, axis=0)
    keep = scales > np.finfo(float).tiny
    centered = centered[:, keep] / scales[keep]
    # Apply the projection without constructing an n-by-n matrix.
    # Z @ pinv(Z) @ X is equivalent to Z @ lstsq(Z, X).
    first_stage, _, _, _ = np.linalg.lstsq(centered, design, rcond=None)
    projected_design = centered @ first_stage
    parameters, _, rank, singular_values = np.linalg.lstsq(projected_design, force, rcond=None)
    original_energy = np.sum(design**2, axis=0)
    explained_energy = np.sum(projected_design**2, axis=0)
    strength = np.divide(
        explained_energy,
        original_energy,
        out=np.zeros_like(explained_energy),
        where=original_energy > 0.0,
    )
    diagnostics = {
        "instrument_count": float(centered.shape[1]),
        "projected_design_rank": float(rank),
        "projected_design_condition_number": float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf"),
        "minimum_instrument_strength": float(np.min(strength)),
        "mean_instrument_strength": float(np.mean(strength)),
    }
    if rank < 3:
        parameters = np.full(3, np.nan)
    return _result("iv", design, force, parameters, diagnostics)


def estimate_eiv(
    estimator: str,
    measurements: SyntheticMeasurements,
    *,
    instruments: ArrayLike | None = None,
) -> LinearEIVResult:
    if estimator == "ols":
        return ordinary_least_squares_eiv(measurements)
    if estimator == "tls":
        return total_least_squares(measurements)
    if estimator == "iv":
        if instruments is None:
            raise ValueError("IV requires instruments")
        return instrumental_variables(measurements, instruments)
    raise ValueError(f"unknown EIV estimator: {estimator}")
