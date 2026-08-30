from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from probeing.experiments.passive_ringdown import (
    _early_stop,
    _extended_probe_signal,
    _legacy_cases,
    ringdown_features,
    run_passive_ringdown,
)
from probeing.measurements import MeasurementNoise, SyntheticMeasurements


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "exp_0006_passive_ringdown.yaml"
EXP5_CONFIG_PATH = ROOT / "configs" / "experiments" / "exp_0005_decision_sufficiency.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_extended_probe_commands_exactly_zero_force_after_chirp() -> None:
    config = _config()
    time, force = _extended_probe_signal(config)
    probe_end = float(config["probe"]["duration_s"])
    assert np.max(np.abs(force)) <= float(config["probe"]["amplitude_n"]) + 1.0e-12
    assert np.all(force[time > probe_end] == 0.0)
    assert np.any(np.abs(force[time <= probe_end]) > 0.0)


def test_exp_0006_preserves_exp_0005_target_and_risk_definitions() -> None:
    current = _config()
    previous = yaml.safe_load(EXP5_CONFIG_PATH.read_text(encoding="utf-8"))
    assert current["target_population"] == {
        **previous["target_population"],
        "policy": current["target_population"]["policy"],
    }
    assert current["risk_envelope"] == {
        **previous["risk_envelope"],
        "class_rule": current["risk_envelope"]["class_rule"],
    }
    assert current["future_contact_maneuver"]["contact_force_n"] == previous[
        "future_contact_maneuver"
    ]["contact_force_n"]


def test_ringdown_features_use_only_available_prefix() -> None:
    config = _config()
    time = np.arange(0.0, 6.0001, 0.005)
    relative = np.maximum(time - 3.0, 0.0)
    displacement = np.where(
        time >= 3.0,
        0.004 * np.exp(-1.5 * relative) * np.sin(2.0 * np.pi * 2.0 * relative),
        0.0,
    )
    velocity = np.gradient(displacement, time)
    common = dict(
        time_s=time,
        velocity_m_per_s=velocity,
        acceleration_m_per_s2=np.gradient(velocity, time),
        contact_force_n=np.zeros_like(time),
        noise=MeasurementNoise(displacement_std_m=1.0e-5),
        random_seed=1,
    )
    first = SyntheticMeasurements(displacement_m=displacement, **common)
    changed = displacement.copy()
    changed[time > 3.5] = 100.0
    second = SyntheticMeasurements(displacement_m=changed, **common)
    first_features = ringdown_features(first, 3.0, 0.5, config)
    second_features = ringdown_features(second, 3.0, 0.5, config)
    assert first_features == second_features
    assert first_features["rd_valid"] == 1.0
    assert first_features["rd_dominant_frequency_hz"] > 0.0


def test_locked_exp_0005_false_safe_cases_reproduce_exact_target_ids() -> None:
    config = _config()
    cases = _legacy_cases(config)
    assert [case.target_id for case in cases] == config[
        "legacy_exp_0005_false_safe_audit"
    ]["target_ids"]


def test_early_stopping_does_not_use_ground_truth_fields() -> None:
    config = _config()
    rows = []
    for duration in config["passive_observation"]["windows_s"]:
        rows.append(
            {
                "target_id": "case",
                "noise_regime": "nominal",
                "feature_set": "chirp_ringdown",
                "observation_duration_s": float(duration),
                "predicted_risk_class": "UNSAFE",
                "decision_margin": 0.5,
                "upper_hold_settling_time_s": 3.0,
                "rd_decay_fit_r2": 0.8,
                "actual_risk_class": "SAFE",
                "classification_correct": False,
                "false_safe": False,
                "false_unsafe": True,
            }
        )
    original, _ = _early_stop(rows, config)
    altered = [
        {
            **row,
            "actual_risk_class": "UNSAFE",
            "classification_correct": True,
            "false_unsafe": False,
        }
        for row in rows
    ]
    changed, _ = _early_stop(altered, config)
    assert original[0]["early_stop_duration_s"] == changed[0]["early_stop_duration_s"]
    assert original[0]["predicted_risk_class"] == changed[0]["predicted_risk_class"]


def test_small_passive_ringdown_workflow_has_matched_energy_and_hidden_join() -> None:
    config = _config()
    config["seed_partitions"]["training"] = [1901, 1902]
    config["seed_partitions"]["calibration"] = [1911]
    config["seed_partitions"]["validation"] = [1921, 1922, 1923, 1924, 1925]
    config["integrity_acceptance"]["minimum_validation_seeds"] = 5
    config["integrity_acceptance"]["minimum_validation_cases"] = 60
    result = run_passive_ringdown(config)
    assert result.metrics["validation_case_count"] == 60
    assert result.acceptance_checks["validation_predictions_precede_future_maneuver"]
    assert result.acceptance_checks["passive_force_is_zero"]
    assert result.acceptance_checks["no_additional_probe_energy"]
    assert len(result.validation_rows) == 60 * 3 * 7 * 4
    nominal_dose = {
        row["median_probe_force_squared_dose_n2_s"]
        for row in result.duration_summary
        if row["noise_regime"] == "nominal"
    }
    assert len(nominal_dose) == 1
