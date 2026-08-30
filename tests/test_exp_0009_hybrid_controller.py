from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml

from probeing.contact import TargetParameters
from probeing.controllers import ContactObservation, ContactPhase, HybridContactController
from probeing.experiments.coupled_uav_contact import locked_probe_force
from probeing.experiments.hybrid_contact_delivery import (
    hybrid_delivery_metrics,
    hybrid_future_outcome,
    simulate_hybrid_contact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/exp_0009_hybrid_contact_delivery.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _observation(
    time_s: float,
    *,
    force_n: float = 0.0,
    penetration_m: float = 0.0,
    relative_velocity_m_per_s: float = 0.0,
) -> ContactObservation:
    return ContactObservation(
        time_s=time_s,
        contact_force_n=force_n,
        penetration_m=penetration_m,
        relative_normal_velocity_m_per_s=relative_velocity_m_per_s,
        target_displacement_m=0.0,
        target_velocity_m_per_s=0.0,
        target_acceleration_m_per_s2=0.0,
        attitude_deg=0.0,
    )


def _controller(values: dict | None = None) -> HybridContactController:
    return HybridContactController(
        copy.deepcopy(CONFIG["hybrid_controller"] if values is None else values),
        interface_stiffness_n_per_m=float(CONFIG["contact"]["interface_stiffness_n_per_m"]),
        probe_duration_s=float(CONFIG["locked_probe"]["duration_s"]),
        observation_duration_s=float(CONFIG["locked_probe"]["observation_duration_s"]),
        probe_reference=lambda time_s: 0.5 if time_s <= 3.0 else 0.0,
        vehicle_mass_kg=float(CONFIG["vehicle"]["mass_kg"]),
    )


def test_hybrid_state_machine_reaches_decision_with_logged_transitions() -> None:
    controller = _controller()
    step = 0.01
    time_s = 0.0
    controller.command(_observation(time_s), step)
    time_s += step
    controller.command(_observation(time_s, force_n=0.04, penetration_m=1e-5), step)
    assert controller.phase == ContactPhase.CONTACT_ACQUIRE

    while controller.phase == ContactPhase.CONTACT_ACQUIRE and time_s < 1.0:
        time_s += step
        controller.command(_observation(time_s, force_n=0.05, penetration_m=1e-5), step)
    assert controller.phase == ContactPhase.PRELOAD

    while controller.phase == ContactPhase.PRELOAD and time_s < 4.5:
        time_s += step
        controller.command(_observation(time_s, force_n=0.50, penetration_m=1e-4), step)
    assert controller.phase == ContactPhase.PROBE

    while controller.phase == ContactPhase.PROBE and time_s < 8.0:
        time_s += step
        controller.command(_observation(time_s, force_n=0.50, penetration_m=1e-4), step)
    assert controller.phase == ContactPhase.CONTROLLED_UNLOAD

    unload_references: list[float] = []
    while controller.phase == ContactPhase.CONTROLLED_UNLOAD and time_s < 9.0:
        time_s += step
        output = controller.command(_observation(time_s, force_n=0.20, penetration_m=4e-5), step)
        unload_references.append(output.total_contact_reference_n)
    assert controller.phase == ContactPhase.PASSIVE_OBSERVE
    assert unload_references[0] <= float(CONFIG["hybrid_controller"]["maximum_force_command_n"])
    assert np.all(np.diff(unload_references) <= 1e-12)

    passive_coordinate = controller.commanded_tip_coordinate_m
    while controller.phase == ContactPhase.PASSIVE_OBSERVE and time_s < 10.0:
        time_s += step
        output = controller.command(_observation(time_s), step)
        assert output.commanded_normal_force_n == 0.0
        assert output.probe_variation_reference_n == 0.0
        assert output.desired_tip_velocity_m_per_s == 0.0
        assert output.desired_tip_coordinate_m == passive_coordinate
    assert controller.phase == ContactPhase.DECISION
    assert [item.to_phase for item in controller.transitions] == [
        "CONTACT_ACQUIRE", "PRELOAD", "PROBE", "CONTROLLED_UNLOAD",
        "PASSIVE_OBSERVE", "DECISION",
    ]


def test_approach_velocity_and_preload_are_prospectively_bounded() -> None:
    controller = _controller()
    output = controller.command(_observation(0.0), 0.01)
    assert 0.0 <= output.desired_tip_velocity_m_per_s <= float(
        CONFIG["hybrid_controller"]["maximum_approach_velocity_m_per_s"]
    )
    assert float(CONFIG["hybrid_controller"]["preload_force_n"]) <= float(
        CONFIG["hybrid_controller"]["maximum_force_command_n"]
    )
    assert float(CONFIG["hybrid_controller"]["maximum_force_command_n"]) < float(
        CONFIG["hybrid_controller"]["abort_force_n"]
    )


def test_force_saturation_uses_anti_windup() -> None:
    values = copy.deepcopy(CONFIG["hybrid_controller"])
    values["force_feedforward_error_gain"] = 10.0
    values["force_integral_gain_per_s"] = 3.0
    controller = _controller(values)
    controller.phase = ContactPhase.PROBE
    controller.phase_start_s = 0.0
    output = controller.command(_observation(0.1, force_n=0.10, penetration_m=2e-5), 0.01)
    assert output.commanded_normal_force_n == float(values["maximum_force_command_n"])
    assert controller.integral_force_error_n_s == 0.0


def test_unilateral_force_and_passive_command_in_coupled_simulation() -> None:
    target = TargetParameters(2200.0, 12.0, 1.2)
    trajectory = simulate_hybrid_contact(CONFIG, target)
    assert trajectory.completed
    assert np.all(trajectory.realized_contact_force_n >= 0.0)
    assert np.all(trajectory.realized_contact_force_n[~trajectory.contact_active] == 0.0)
    passive = trajectory.phase == ContactPhase.PASSIVE_OBSERVE.value
    assert np.count_nonzero(passive) > 10
    assert np.all(trajectory.commanded_normal_force_n[passive] == 0.0)
    assert np.all(trajectory.probe_variation_reference_n[passive] == 0.0)
    metrics = hybrid_delivery_metrics(trajectory, CONFIG)
    assert metrics["zero_separation_during_probe"]
    assert metrics["passive_max_commanded_probe_force_n"] == 0.0


def test_frozen_chirp_observation_policy_and_stage_artifacts_are_unchanged() -> None:
    probe = CONFIG["locked_probe"]
    assert (probe["start_frequency_hz"], probe["end_frequency_hz"]) == (0.5, 5.0)
    assert probe["amplitude_n"] == 0.5
    assert probe["duration_s"] == 3.0
    assert probe["observation_duration_s"] == 0.5
    time = np.arange(0.0, 3.501, 0.001)
    force = locked_probe_force(time, CONFIG)
    assert np.max(force) <= 0.5 + 1e-12
    assert np.all(force >= 0.0)
    assert np.all(force[time > 3.0] == 0.0)

    policy_path = ROOT / CONFIG["frozen_references"]["locked_policy_bundle"]
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == CONFIG["frozen_references"]["locked_policy_sha256"]
    exp8_run = ROOT / CONFIG["frozen_references"]["exp_0008_run"]
    assert exp8_run.is_dir()
    assert (ROOT / "lab/experiments/EXP-0008.md").is_file()


def test_aborted_sustained_contact_is_conservatively_unsafe() -> None:
    locked = yaml.safe_load(
        (ROOT / "configs/experiments/exp_0006_passive_ringdown.yaml").read_text(encoding="utf-8")
    )
    target = TargetParameters(1419.3558575701143, 1.864692315277412, 1.26200851333251)
    outcome, trajectory = hybrid_future_outcome(CONFIG, target, locked)
    assert trajectory.aborted
    assert outcome["risk_class"] == "UNSAFE"
    assert outcome["contact_loss_proxy"]
    assert np.isinf(outcome["hold_settling_time_s"])
