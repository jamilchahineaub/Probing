from pathlib import Path

import numpy as np
import pytest
import yaml

from probeing.estimators import dynamic_ratio_least_squares
from probeing.experiments import run_practical_identifiability
from probeing.measurements import (
    causal_low_pass,
    finite_difference,
    process_practical_sensing,
    savitzky_golay,
)
from probeing.models import (
    InteractionParameters,
    MassSpringDamperModel,
    simulate_contact_interaction,
)
from probeing.probing import sinusoid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "exp_0002_practical_identifiability.yaml"
)


def contact_truth(mode="bilateral"):
    model = MassSpringDamperModel(InteractionParameters(300.0, 5.0, 1.2))
    probe = sinusoid(1.0, 2.0, 2.0, 0.002)
    return simulate_contact_interaction(
        model, probe.time_s, probe.force_n, contact_mode=mode
    )


def pipeline_settings():
    return {
        "low_pass_cutoff_hz": 10.0,
        "savgol_window_duration_s": 0.12,
        "savgol_polynomial_order": 3,
        "complementary_time_constant_s": 0.2,
        "sensorless_force_cutoff_hz": 10.0,
    }


def imperfection():
    return {
        "sample_rate_hz": 100.0,
        "position_sample_rate_hz": 25.0,
        "displacement_std_m": 1.0e-5,
        "velocity_std_m_per_s": 1.0e-3,
        "acceleration_std_m_per_s2": 0.05,
        "force_std_n": 0.01,
        "position_latency_s": 0.005,
        "acceleration_latency_s": 0.002,
        "force_latency_s": 0.006,
        "timestamp_mismatch_std_s": 0.001,
        "sensorless_command_latency_s": 0.01,
    }


def test_unilateral_contact_cannot_pull_and_records_contact_loss():
    bilateral = contact_truth("bilateral")
    unilateral = contact_truth("unilateral")

    assert np.min(bilateral.contact_force_n) < 0.0
    assert np.min(unilateral.contact_force_n) == 0.0
    assert np.all(unilateral.contact_force_n >= 0.0)
    assert np.any(~unilateral.contact_active)
    assert unilateral.contact_loss_count == 4
    np.testing.assert_allclose(
        unilateral.contact_force_n,
        np.maximum(unilateral.commanded_force_n, 0.0),
        atol=1.0e-15,
    )


def test_finite_difference_is_exact_for_quadratic_interior():
    time = np.linspace(0.0, 1.0, 101)
    values = 2.0 * time**2 - 3.0 * time + 1.0

    velocity = finite_difference(values, time, derivative_order=1)
    acceleration = finite_difference(values, time, derivative_order=2)

    np.testing.assert_allclose(velocity, 4.0 * time - 3.0, atol=2.0e-12)
    np.testing.assert_allclose(acceleration[2:-2], 4.0, atol=2.0e-10)


def test_savitzky_golay_recovers_cubic_and_derivatives():
    time = np.linspace(0.0, 2.0, 201)
    values = 0.5 * time**3 - 2.0 * time**2 + time + 4.0

    smooth = savitzky_golay(
        values, 0.01, window_duration_s=0.11, polynomial_order=3
    )
    velocity = savitzky_golay(
        values,
        0.01,
        window_duration_s=0.11,
        polynomial_order=3,
        derivative_order=1,
    )
    acceleration = savitzky_golay(
        values,
        0.01,
        window_duration_s=0.11,
        polynomial_order=3,
        derivative_order=2,
    )

    np.testing.assert_allclose(smooth, values, atol=2.0e-11)
    np.testing.assert_allclose(velocity, 1.5 * time**2 - 4.0 * time + 1.0, atol=2.0e-10)
    np.testing.assert_allclose(acceleration, 3.0 * time - 4.0, atol=2.0e-9)


def test_causal_low_pass_is_bounded_and_delays_a_step():
    values = np.concatenate((np.zeros(20), np.ones(80)))
    filtered = causal_low_pass(values, 0.01, 5.0)

    assert np.all((0.0 <= filtered) & (filtered <= 1.0))
    assert 0.0 < filtered[20] < 1.0
    assert filtered[-1] == pytest.approx(1.0, rel=1.0e-8)


@pytest.mark.parametrize(
    "regime,pipeline",
    [
        ("optimistic_reference", "direct"),
        ("no_direct_velocity", "finite_difference"),
        ("no_direct_acceleration", "savitzky_golay"),
        ("imu_like", "low_pass"),
        ("sensorless_force_exploratory", "savitzky_golay"),
    ],
)
def test_practical_sensing_regimes_are_seeded_finite_and_complete(regime, pipeline):
    first = process_practical_sensing(
        contact_truth("unilateral"),
        regime=regime,
        pipeline=pipeline,
        imperfection=imperfection(),
        pipeline_settings=pipeline_settings(),
        random_seed=1501,
    )
    second = process_practical_sensing(
        contact_truth("unilateral"),
        regime=regime,
        pipeline=pipeline,
        imperfection=imperfection(),
        pipeline_settings=pipeline_settings(),
        random_seed=1501,
    )

    measurements = first.measurements
    assert len(measurements.time_s) == 201
    assert all(
        np.all(np.isfinite(values))
        for values in (
            measurements.displacement_m,
            measurements.velocity_m_per_s,
            measurements.acceleration_m_per_s2,
            measurements.contact_force_n,
        )
    )
    np.testing.assert_array_equal(
        measurements.displacement_m, second.measurements.displacement_m
    )
    assert first.force_is_inferred == (regime == "sensorless_force_exploratory")
    if regime in {"imu_like", "sensorless_force_exploratory"}:
        assert first.position_sample_rate_hz == 25.0


def test_exploratory_sensorless_force_is_not_the_true_unilateral_force():
    result = process_practical_sensing(
        contact_truth("unilateral"),
        regime="sensorless_force_exploratory",
        pipeline="savitzky_golay",
        imperfection=imperfection(),
        pipeline_settings=pipeline_settings(),
        random_seed=1502,
    )

    assert np.sqrt(
        np.mean((result.measurements.contact_force_n - result.true_contact_force_n) ** 2)
    ) > 0.05


def test_dynamic_ratio_estimator_recovers_modal_and_physical_parameters_perfectly():
    truth = contact_truth("bilateral")
    sensing = process_practical_sensing(
        truth,
        regime="optimistic_reference",
        pipeline="direct",
        imperfection={**imperfection(), **{
            "displacement_std_m": 0.0,
            "velocity_std_m_per_s": 0.0,
            "acceleration_std_m_per_s2": 0.0,
            "force_std_n": 0.0,
            "position_latency_s": 0.0,
            "acceleration_latency_s": 0.0,
            "force_latency_s": 0.0,
            "timestamp_mismatch_std_s": 0.0,
        }},
        pipeline_settings=pipeline_settings(),
        random_seed=1503,
    )

    result = dynamic_ratio_least_squares(sensing.measurements)

    assert result.valid_physical_parameters
    np.testing.assert_allclose(result.parameters, [300.0, 5.0, 1.2], rtol=1.0e-12)
    assert result.natural_frequency_rad_per_s == pytest.approx(np.sqrt(250.0))


def test_exp_0002_development_smoke_matrix_runs_without_validation_interpretation():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["validation_seeds"] = [1501, 1502]
    config["targets"] = config["targets"][:2]
    config["probes"] = [config["probes"][0], config["probes"][3]]
    config["imperfection_severities"] = [config["imperfection_severities"][1]]
    config["sensing_regimes"] = [
        {
            "name": "no_direct_acceleration",
            "pipelines": ["savitzky_golay"],
        },
        {"name": "imu_like", "pipelines": ["savitzky_golay"]},
        {
            "name": "sensorless_force_exploratory",
            "pipelines": ["savitzky_golay"],
        },
    ]
    config["integrity_acceptance"]["minimum_monte_carlo_seeds"] = 2

    result = run_practical_identifiability(config, repository_root=REPOSITORY_ROOT)

    assert result.metrics["trial_count"] == 48
    assert result.acceptance_checks["complete_trial_matrix"]
    assert not result.acceptance_checks["untouched_validation_seed_partition"]
    assert result.go_candidates
    assert result.representative_raw
    assert result.ramp_failure_analysis["worst_case_effective_mass_relative_error"] == pytest.approx(
        -0.6267813182732596
    )
