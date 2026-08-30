"""Transparent linear identification baselines for Milestone A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.measurements import SyntheticMeasurements


PARAMETER_NAMES = ("stiffness_n_per_m", "damping_n_s_per_m", "effective_mass_kg")


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Scale-independent excitation and parameter-correlation diagnostics."""

    rank: int
    normalized_condition_number: float
    normalized_singular_values: NDArray[np.float64]
    regressor_correlation: NDArray[np.float64]
    parameter_correlation: NDArray[np.float64]
    maximum_abs_parameter_correlation: float
    feature_scales: NDArray[np.float64]


@dataclass(frozen=True)
class BatchLeastSquaresResult:
    parameters: NDArray[np.float64]
    predicted_force_n: NDArray[np.float64]
    residual_n: NDArray[np.float64]
    force_rmse_n: float
    diagnostics: RegressionDiagnostics

    @property
    def stiffness_n_per_m(self) -> float:
        return float(self.parameters[0])

    @property
    def damping_n_s_per_m(self) -> float:
        return float(self.parameters[1])

    @property
    def effective_mass_kg(self) -> float:
        return float(self.parameters[2])


@dataclass(frozen=True)
class RecursiveLeastSquaresResult:
    parameter_history: NDArray[np.float64]
    predicted_force_n: NDArray[np.float64]
    residual_n: NDArray[np.float64]
    covariance_history: NDArray[np.float64]
    feature_scales: NDArray[np.float64]

    @property
    def parameters(self) -> NDArray[np.float64]:
        return self.parameter_history[-1]


@dataclass(frozen=True)
class DynamicRatioLeastSquaresResult:
    """Alternative acceleration-form estimate and derived physical parameters."""

    ratios: NDArray[np.float64]
    parameters: NDArray[np.float64]
    natural_frequency_rad_per_s: float
    damping_ratio: float
    acceleration_rmse_m_per_s2: float
    valid_physical_parameters: bool
    diagnostics: RegressionDiagnostics


def regression_matrix(measurements: SyntheticMeasurements) -> NDArray[np.float64]:
    """Return columns ordered as ``[x, x_dot, x_ddot]`` for ``[k, c, m]``."""

    design = np.column_stack(
        (
            measurements.displacement_m,
            measurements.velocity_m_per_s,
            measurements.acceleration_m_per_s2,
        )
    )
    if not np.all(np.isfinite(design)):
        raise ValueError("measurement regressors must be finite")
    return design


def regression_diagnostics(design: ArrayLike) -> RegressionDiagnostics:
    """Measure rank, conditioning, and correlation after column normalization."""

    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] < 3:
        raise ValueError("design must have shape (n, 3) with n >= 3")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must be finite")
    feature_scales = np.linalg.norm(matrix, axis=0)
    safe_scales = np.where(feature_scales > np.finfo(float).tiny, feature_scales, 1.0)
    normalized = matrix / safe_scales
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    tolerance = np.finfo(float).eps * max(normalized.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    regressor_correlation = normalized.T @ normalized
    information_inverse = np.linalg.pinv(regressor_correlation, rcond=1.0e-14)
    covariance_scale = np.sqrt(np.maximum(np.diag(information_inverse), 0.0))
    covariance_denominator = np.outer(covariance_scale, covariance_scale)
    parameter_correlation = np.divide(
        information_inverse,
        covariance_denominator,
        out=np.zeros_like(information_inverse),
        where=covariance_denominator > 0.0,
    )
    off_diagonal = parameter_correlation - np.diag(np.diag(parameter_correlation))
    return RegressionDiagnostics(
        rank=rank,
        normalized_condition_number=condition,
        normalized_singular_values=singular_values,
        regressor_correlation=regressor_correlation,
        parameter_correlation=parameter_correlation,
        maximum_abs_parameter_correlation=float(np.max(np.abs(off_diagonal))),
        feature_scales=feature_scales,
    )


def batch_least_squares(measurements: SyntheticMeasurements) -> BatchLeastSquaresResult:
    """Estimate ``[k, c, m_eff]`` using unregularized batch least squares."""

    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    if not np.all(np.isfinite(force)):
        raise ValueError("force measurements must be finite")
    parameters, _, _, _ = np.linalg.lstsq(design, force, rcond=None)
    predicted = design @ parameters
    residual = force - predicted
    return BatchLeastSquaresResult(
        parameters=np.asarray(parameters, dtype=float),
        predicted_force_n=predicted,
        residual_n=residual,
        force_rmse_n=float(np.sqrt(np.mean(residual**2))),
        diagnostics=regression_diagnostics(design),
    )


def dynamic_ratio_least_squares(
    measurements: SyntheticMeasurements,
) -> DynamicRatioLeastSquaresResult:
    """Estimate ``[k/m, c/m, 1/m]`` from the acceleration-form equation.

    This alternate parameterization moves measured acceleration to the response
    side of the regression. It is included to test whether natural frequency
    and damping ratio remain useful when separate effective mass does not.
    """

    design = np.column_stack(
        (
            -measurements.displacement_m,
            -measurements.velocity_m_per_s,
            measurements.contact_force_n,
        )
    )
    acceleration = np.asarray(measurements.acceleration_m_per_s2, dtype=float)
    ratios, _, _, _ = np.linalg.lstsq(design, acceleration, rcond=None)
    predicted = design @ ratios
    inverse_mass = float(ratios[2])
    frequency_squared = float(ratios[0])
    damping_rate = float(ratios[1])
    valid = bool(
        np.all(np.isfinite(ratios))
        and inverse_mass > np.finfo(float).eps
        and frequency_squared > np.finfo(float).eps
    )
    if valid:
        effective_mass = 1.0 / inverse_mass
        stiffness = frequency_squared * effective_mass
        damping = damping_rate * effective_mass
        natural_frequency = float(np.sqrt(frequency_squared))
        damping_ratio = float(damping_rate / (2.0 * natural_frequency))
        parameters = np.asarray([stiffness, damping, effective_mass], dtype=float)
    else:
        parameters = np.full(3, np.nan, dtype=float)
        natural_frequency = float("nan")
        damping_ratio = float("nan")
    return DynamicRatioLeastSquaresResult(
        ratios=np.asarray(ratios, dtype=float),
        parameters=parameters,
        natural_frequency_rad_per_s=natural_frequency,
        damping_ratio=damping_ratio,
        acceleration_rmse_m_per_s2=float(
            np.sqrt(np.mean((acceleration - predicted) ** 2))
        ),
        valid_physical_parameters=valid,
        diagnostics=regression_diagnostics(design),
    )


def recursive_least_squares(
    measurements: SyntheticMeasurements,
    *,
    forgetting_factor: float = 1.0,
    initial_covariance: float = 1.0e10,
    initial_parameters: ArrayLike = (0.0, 0.0, 0.0),
) -> RecursiveLeastSquaresResult:
    """Sequential RLS with deterministic feature preconditioning.

    Preconditioning only changes numerical units. The final values are always
    returned in physical units ``[N/m, N*s/m, kg]``.
    """

    if not np.isfinite(forgetting_factor) or not 0.0 < forgetting_factor <= 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")
    if not np.isfinite(initial_covariance) or initial_covariance <= 0.0:
        raise ValueError("initial_covariance must be finite and positive")
    design = regression_matrix(measurements)
    force = np.asarray(measurements.contact_force_n, dtype=float)
    feature_scales = np.sqrt(np.mean(design**2, axis=0))
    feature_scales = np.where(feature_scales > np.finfo(float).tiny, feature_scales, 1.0)
    normalized_design = design / feature_scales

    physical_initial = np.asarray(initial_parameters, dtype=float)
    if physical_initial.shape != (3,) or not np.all(np.isfinite(physical_initial)):
        raise ValueError("initial_parameters must contain three finite values")
    scaled_parameters = feature_scales * physical_initial
    covariance = np.eye(3, dtype=float) * initial_covariance
    parameter_history = np.empty((design.shape[0], 3), dtype=float)
    covariance_history = np.empty((design.shape[0], 3, 3), dtype=float)
    predicted = np.empty(design.shape[0], dtype=float)
    residual = np.empty(design.shape[0], dtype=float)

    for index, (regressor, observed_force) in enumerate(zip(normalized_design, force)):
        prediction = float(regressor @ scaled_parameters)
        innovation = float(observed_force - prediction)
        covariance_regressor = covariance @ regressor
        denominator = forgetting_factor + float(regressor @ covariance_regressor)
        gain = covariance_regressor / denominator
        scaled_parameters = scaled_parameters + gain * innovation
        covariance = (
            covariance - np.outer(gain, covariance_regressor)
        ) / forgetting_factor
        covariance = 0.5 * (covariance + covariance.T)
        predicted[index] = prediction
        residual[index] = innovation
        parameter_history[index] = scaled_parameters / feature_scales
        scale_inverse = np.diag(1.0 / feature_scales)
        covariance_history[index] = scale_inverse @ covariance @ scale_inverse

    return RecursiveLeastSquaresResult(
        parameter_history=parameter_history,
        predicted_force_n=predicted,
        residual_n=residual,
        covariance_history=covariance_history,
        feature_scales=feature_scales,
    )
