from pathlib import Path

import yaml

from probeing.experiments.reduced_kelvin_voigt import run_kelvin_voigt_validation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_exp_0001_config():
    with (REPOSITORY_ROOT / "configs/experiments/exp_0001_kelvin_voigt_validation.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def test_exp_0001_meets_predeclared_acceptance_criteria():
    result = run_kelvin_voigt_validation(load_exp_0001_config())

    assert result.success
    assert all(result.acceptance_checks.values())
    assert result.safety_events == ()
    assert result.metrics["force_rmse_n"] == 0.0
    assert result.metrics["peak_displacement_m"] <= 0.005
    assert result.metrics["relative_energy_balance_error"] <= 1.0e-5


def test_safety_violation_fails_required_acceptance():
    config = load_exp_0001_config()
    config["safety_limits"]["max_abs_force_n"] = 1.0

    result = run_kelvin_voigt_validation(config)

    assert not result.success
    assert not result.acceptance_checks["safety_limits_respected"]
    assert any(event["metric"] == "peak_abs_force_n" for event in result.safety_events)

