from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from probeing.experiments.decision_sufficiency import (
    FEATURE_SETS,
    TargetCase,
    _inverse_outcome,
    _maneuver_signal,
    _risk_class,
    _transform_outcome,
    evaluate_future_response,
    generate_target_cases,
    run_decision_sufficiency,
    simulate_population,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "exp_0005_decision_sufficiency.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_exp_0005_uses_fixed_causal_probe_and_task_features_exclude_c_and_m() -> None:
    config = _config()
    assert config["probe"]["type"] == "chirp"
    assert config["probe"]["adaptive"] is False
    assert config["probe"]["start_frequency_hz"] == 0.5
    assert config["probe"]["end_frequency_hz"] == 5.0
    assert config["sensing"]["primary_pipeline"] == "causal_low_pass"
    assert config["sensing"]["no_noncausal_primary_result"] is True
    assert "estimated_damping_signed_log" not in FEATURE_SETS["combined_task"]
    assert "estimated_mass_signed_log" not in FEATURE_SETS["combined_task"]


def test_target_partitions_are_reproducible_and_disjoint() -> None:
    bounds = _config()["target_population"]
    first = generate_target_cases([11, 12], 12, bounds, partition="training")
    repeated = generate_target_cases([11, 12], 12, bounds, partition="training")
    validation = generate_target_cases([13], 12, bounds, partition="validation")
    assert first == repeated
    assert not ({case.target_id for case in first} & {case.target_id for case in validation})
    assert all(bounds["stiffness_n_per_m"][0] <= case.stiffness_n_per_m <= bounds["stiffness_n_per_m"][1] for case in first)


def test_zero_valued_future_metrics_have_stable_regression_transform() -> None:
    for outcome in ("dominant_response_frequency_hz", "hold_settling_time_s"):
        assert _transform_outcome(outcome, 0.0) == 0.0
        assert _inverse_outcome(outcome, _transform_outcome(outcome, 0.0)) == 0.0
        value = 2.75
        assert np.isclose(_inverse_outcome(outcome, _transform_outcome(outcome, value)), value)


def test_risk_labels_are_derived_from_future_response_metrics() -> None:
    config = _config()
    safe = {
        "peak_displacement_m": 0.005,
        "peak_velocity_m_per_s": 0.02,
        "late_hold_oscillation_rms_m": 0.0002,
        "hold_settling_time_s": 0.5,
        "contact_loss_proxy": False,
    }
    caution = {**safe, "peak_displacement_m": 0.015}
    unsafe = {**safe, "peak_displacement_m": 0.021}
    assert _risk_class(safe, config) == "SAFE"
    assert _risk_class(caution, config) == "CAUTION"
    assert _risk_class(unsafe, config) == "UNSAFE"
    assert _risk_class({**safe, "contact_loss_proxy": True}, config) == "UNSAFE"


def test_future_outcomes_come_from_hidden_maneuver_response() -> None:
    config = _config()
    cases = (
        TargetCase("soft", "test", 1, 0, 100.0, 3.0, 1.0),
        TargetCase("stiff", "test", 1, 1, 1800.0, 20.0, 1.0),
    )
    time, force = _maneuver_signal(config)
    population = simulate_population(cases, time, force, contact_mode="unilateral")
    outcomes = evaluate_future_response(population, cases, config)
    by_id = {row["target_id"]: row for row in outcomes}
    assert by_id["soft"]["peak_displacement_m"] > by_id["stiff"]["peak_displacement_m"]
    assert by_id["soft"]["force_to_displacement_gain_m_per_n"] > by_id["stiff"]["force_to_displacement_gain_m_per_n"]
    assert by_id["soft"]["risk_class"] != by_id["stiff"]["risk_class"]


def test_small_workflow_creates_predictions_before_hidden_validation_join() -> None:
    config = _config()
    config["seed_partitions"]["training"] = [1801, 1802]
    config["seed_partitions"]["calibration"] = [1821]
    config["seed_partitions"]["validation"] = [1831, 1832, 1833, 1834, 1835]
    config["integrity_acceptance"]["minimum_validation_seeds"] = 5
    config["integrity_acceptance"]["minimum_validation_cases"] = 60
    config["integrity_acceptance"]["forbidden_validation_seeds_before_final_run"] = [
        1831,
        1832,
        1833,
        1834,
        1835,
    ]
    result = run_decision_sufficiency(config)
    assert result.summary["validation_predictions_created_before_future_maneuver"] is True
    assert result.acceptance_checks["validation_predictions_precede_future_maneuver"] is True
    assert result.acceptance_checks["combined_task_excludes_damping_and_mass_estimates"] is True
    assert result.metrics["validation_case_count"] == 60
    assert len(result.validation_rows) == 60 * 3 * 11
    assert {row["risk_class"] for row in result.class_distribution} == {
        "SAFE",
        "CAUTION",
        "UNSAFE",
    }
