from pathlib import Path

import numpy as np
import yaml

from probeing.experiments import run_interaction_identification


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "exp_0001_interaction_identification.yaml"
)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_exp_0001_runs_complete_matrix_and_meets_predeclared_acceptance():
    result = run_interaction_identification(load_config())

    assert result.success
    assert all(result.acceptance_checks.values())
    assert result.safety_events == ()
    assert result.metrics["case_count"] == 60
    assert result.metrics["perfect_case_count"] == 30
    assert result.metrics["near_ideal_case_count"] == 30
    assert result.metrics["minimum_regression_rank_perfect"] == 3
    assert result.metrics["maximum_perfect_batch_relative_error_rms"] <= 1.0e-9
    assert result.metrics["maximum_perfect_rls_relative_error_rms"] <= 1.0e-6
    assert result.metrics["maximum_abs_parameter_correlation_perfect"] > 0.95
    assert len(result.case_metrics) == 60
    assert len(result.raw["time_s"]) == result.metrics["sample_count"]
    assert set(np.unique(result.raw["probe"])) == {
        "ramp",
        "half_sine",
        "sinusoid",
        "chirp",
        "multisine",
    }


def test_exp_0001_reports_bounded_probe_violation_without_hiding_results():
    config = load_config()
    config["safety_limits"]["max_abs_probe_force_n"] = 0.9

    result = run_interaction_identification(config)

    assert not result.success
    assert not result.acceptance_checks["safety_limits_respected"]
    assert any(
        event["metric"] == "peak_abs_probe_force_n"
        for event in result.safety_events
    )
