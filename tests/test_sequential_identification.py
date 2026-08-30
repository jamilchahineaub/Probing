from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

from probeing.experiments.sequential_identification import (
    _make_probe,
    _select_next_probe,
    run_sequential_identification,
)
from probeing.models import (
    InteractionParameters,
    MassSpringDamperModel,
    simulate_contact_interaction,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "exp_0004_sequential_active_identification.yaml"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contact_wrapper_carries_initial_state_between_probe_segments() -> None:
    model = MassSpringDamperModel(InteractionParameters(300.0, 5.0, 1.0))
    time = np.linspace(0.0, 0.2, 101)
    first = simulate_contact_interaction(
        model, time, np.ones_like(time), contact_mode="unilateral"
    )
    second = simulate_contact_interaction(
        model,
        time,
        np.zeros_like(time),
        contact_mode="unilateral",
        initial_displacement_m=float(first.response.displacement_m[-1]),
        initial_velocity_m_per_s=float(first.response.velocity_m_per_s[-1]),
    )
    assert second.response.displacement_m[0] == first.response.displacement_m[-1]
    assert second.response.velocity_m_per_s[0] == first.response.velocity_m_per_s[-1]
    assert np.any(np.abs(second.response.displacement_m[1:] - second.response.displacement_m[0]) > 0.0)


def test_exp0004_candidate_library_is_bounded_and_budgetable() -> None:
    config = _config()
    doses = []
    for candidate in config["probe_library"]["candidates"]:
        signal = _make_probe(candidate, config)
        assert np.max(np.abs(signal.force_n)) <= 1.0 + 1.0e-12
        doses.append(float(np.trapz(signal.force_n**2, signal.time_s)))
    assert max(doses) * 3.0 <= float(
        config["disturbance_budget"]["maximum_command_force_squared_dose_n2_s"]
    ) + 1.0e-9


def test_adaptive_selector_has_no_ground_truth_argument() -> None:
    parameters = inspect.signature(_select_next_probe).parameters
    assert "truth" not in parameters
    assert "target" not in parameters
    assert set(parameters) == {"estimate", "current_state", "cumulative", "config"}


def test_exp0004_development_smoke_matrix() -> None:
    config = _config()
    config["validation_seeds"] = [1701]
    config["targets"] = [
        next(target for target in config["targets"] if target["name"] == "high_effective_mass")
    ]
    config["representative_raw"]["seed"] = 1701
    result = run_sequential_identification(config)

    names = {row["strategy"] for row in result.strategy_summary}
    assert names == {
        "single_fixed_chirp",
        "single_multisine",
        "repeated_identical_chirp",
        "predefined_multistage",
        "uncertainty_driven",
    }
    assert max(row["cumulative_probe_count"] for row in result.trial_rows) <= 3
    assert max(
        row["cumulative_force_squared_dose_n2_s"] for row in result.trial_rows
    ) <= float(
        config["disturbance_budget"]["maximum_command_force_squared_dose_n2_s"]
    ) + 1.0e-9
    assert not result.safety_events
    assert result.acceptance_checks["adaptive_selection_does_not_receive_truth"]
    assert not result.acceptance_checks["minimum_monte_carlo_seeds"]
    assert result.representative_raw

    fixed = next(
        row for row in result.strategy_summary if row["strategy"] == "single_fixed_chirp"
    )
    for field in (
        "stiffness_relative_rmse",
        "damping_relative_rmse",
        "effective_mass_relative_rmse",
        "median_duration_s",
        "median_absolute_input_energy_j",
        "maximum_peak_force_n",
        "maximum_peak_target_displacement_m",
        "maximum_peak_target_velocity_m_per_s",
        "maximum_peak_target_acceleration_m_per_s2",
    ):
        assert np.isfinite(float(fixed[field]))
