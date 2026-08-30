from pathlib import Path

import numpy as np
import pytest
import yaml

from probeing.estimators import (
    delayed_input_instruments,
    instrumental_variables,
    ordinary_least_squares_eiv,
    total_least_squares,
)
from probeing.experiments import run_causal_eiv_identifiability
from probeing.measurements import (
    MeasurementNoise,
    SyntheticMeasurements,
    alpha_beta_gamma_filter,
    backward_difference,
    causal_polynomial,
    process_causal_sensing,
)
from probeing.models import InteractionParameters, MassSpringDamperModel, simulate_contact_interaction
from probeing.probing import chirp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "exp_0003_causal_eiv_identifiability.yaml"


def _truth():
    signal = chirp(1.0, 0.5, 10.0, 3.0, 0.002)
    model = MassSpringDamperModel(InteractionParameters(300.0, 5.0, 1.0))
    return simulate_contact_interaction(model, signal.time_s, signal.force_n, contact_mode="unilateral")


def _settings():
    return {
        "low_pass_cutoff_hz": 10.0,
        "polynomial_window_duration_s": 0.11,
        "polynomial_order": 3,
        "observer_alpha": 0.40,
        "observer_beta": 0.08,
        "observer_gamma": 0.008,
        "observer_force_cutoff_hz": 10.0,
    }


def test_backward_differences_are_strictly_causal_and_exact_for_quadratic():
    time = np.arange(101, dtype=float) * 0.01
    values = 2.0 * time**2 - 3.0 * time + 1.0

    velocity = backward_difference(values, 0.01, derivative_order=1)
    acceleration = backward_difference(values, 0.01, derivative_order=2)

    np.testing.assert_allclose(acceleration[2:], 4.0, atol=2.0e-11)
    changed = values.copy()
    changed[-1] += 100.0
    np.testing.assert_array_equal(
        backward_difference(changed, 0.01, derivative_order=1)[:-1], velocity[:-1]
    )


def test_causal_polynomial_recovers_cubic_after_full_window_without_future_samples():
    time = np.arange(201, dtype=float) * 0.01
    values = 0.5 * time**3 - 2.0 * time**2 + time + 4.0
    acceleration = causal_polynomial(
        values,
        0.01,
        window_duration_s=0.11,
        polynomial_order=3,
        derivative_order=2,
    )

    np.testing.assert_allclose(acceleration[12:], 3.0 * time[12:] - 4.0, atol=2.0e-9)
    modified = values.copy()
    modified[150:] += 10.0
    modified_acceleration = causal_polynomial(
        modified,
        0.01,
        window_duration_s=0.11,
        polynomial_order=3,
        derivative_order=2,
    )
    np.testing.assert_array_equal(modified_acceleration[:150], acceleration[:150])


def test_alpha_beta_gamma_filter_is_finite_and_uses_no_future_samples():
    time = np.arange(200, dtype=float) * 0.01
    position = np.sin(2.0 * np.pi * time)
    first = alpha_beta_gamma_filter(position, 0.01, alpha=0.4, beta=0.08, gamma=0.008)
    modified = position.copy()
    modified[100:] += 1.0
    second = alpha_beta_gamma_filter(modified, 0.01, alpha=0.4, beta=0.08, gamma=0.008)

    assert all(np.all(np.isfinite(channel)) for channel in first)
    for original, changed in zip(first, second):
        np.testing.assert_array_equal(original[:100], changed[:100])


@pytest.mark.parametrize(
    "pipeline",
    [
        "direct",
        "backward_difference",
        "causal_low_pass",
        "causal_polynomial",
        "alpha_beta_gamma",
        "centered_savitzky_golay",
    ],
)
def test_exp_0003_sensing_pipelines_are_seeded_and_finite(pipeline):
    result = process_causal_sensing(
        _truth(),
        pipeline=pipeline,
        sample_rate_hz=200.0,
        noise=MeasurementNoise(5.0e-5, 0.002, 0.1, 0.02),
        pipeline_settings=_settings(),
        random_seed=1601,
        timestamp_offsets_s={"displacement": 0.0025, "force": -0.001},
    )

    assert result.measurements.time_s.size == 601
    assert all(
        np.all(np.isfinite(channel))
        for channel in (
            result.measurements.displacement_m,
            result.measurements.velocity_m_per_s,
            result.measurements.acceleration_m_per_s2,
            result.measurements.contact_force_n,
        )
    )
    assert result.is_causal == (pipeline != "centered_savitzky_golay")
    assert result.required_lookahead_s > 0.0 if not result.is_causal else result.required_lookahead_s == 0.0


def test_ols_tls_and_iv_recover_perfect_full_rank_data():
    truth = _truth()
    response = truth.response
    measurements = SyntheticMeasurements(
        time_s=response.time_s,
        displacement_m=response.displacement_m,
        velocity_m_per_s=response.velocity_m_per_s,
        acceleration_m_per_s2=response.acceleration_m_per_s2,
        contact_force_n=truth.contact_force_n,
        noise=MeasurementNoise(),
        random_seed=1602,
    )
    instruments = delayed_input_instruments(
        truth.contact_force_n,
        response.time_s,
        [0.0, 0.025, 0.05, 0.075, 0.10],
    )

    ols = ordinary_least_squares_eiv(measurements)
    tls = total_least_squares(measurements)
    iv = instrumental_variables(measurements, instruments)

    np.testing.assert_allclose(ols.parameters, [300.0, 5.0, 1.0], rtol=1.0e-11)
    np.testing.assert_allclose(tls.parameters, [300.0, 5.0, 1.0], rtol=1.0e-10)
    np.testing.assert_allclose(iv.parameters, [300.0, 5.0, 1.0], rtol=1.0e-10)


def test_exp_0003_reduced_development_matrix_completes_without_validation_claim():
    with CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["validation_seeds"] = [1601, 1602]
    config["targets"] = config["targets"][:1]
    config["chirp_frequency_bands"] = config["chirp_frequency_bands"][2:4]
    config["pipelines"] = [
        {"name": "causal_polynomial", "causal": True},
        {"name": "centered_savitzky_golay", "causal": False},
    ]
    config["timing_study"]["offset_values_ms"] = [-2.5, 0.0, 2.5]
    config["timing_study"]["group_delay_values_ms"] = [0.0, 2.5]
    config["timing_study"]["profiles"] = config["timing_study"]["profiles"][:2]
    config["integrity_acceptance"]["minimum_monte_carlo_seeds"] = 2
    config["representative_raw"] = {
        "target": config["targets"][0]["name"],
        "band": config["chirp_frequency_bands"][0]["name"],
        "seed": 1601,
    }

    result = run_causal_eiv_identifiability(config)

    assert result.metrics["timing_trial_count"] == 22
    assert result.metrics["timing_profile_trial_count"] == 8
    assert result.metrics["identification_trial_count"] == 24
    assert result.acceptance_checks["complete_identification_matrix"]
    assert not result.acceptance_checks["new_untouched_validation_seed_partition"]
    assert result.representative_raw
