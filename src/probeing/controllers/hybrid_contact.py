"""Prospectively specified hybrid contact controller for EXP-0009."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ContactPhase(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_ACQUIRE = "CONTACT_ACQUIRE"
    PRELOAD = "PRELOAD"
    PROBE = "PROBE"
    CONTROLLED_UNLOAD = "CONTROLLED_UNLOAD"
    PASSIVE_OBSERVE = "PASSIVE_OBSERVE"
    DECISION = "DECISION"
    ABORT = "ABORT"


@dataclass(frozen=True)
class ContactObservation:
    time_s: float
    contact_force_n: float
    penetration_m: float
    relative_normal_velocity_m_per_s: float
    target_displacement_m: float
    target_velocity_m_per_s: float
    target_acceleration_m_per_s2: float
    attitude_deg: float
    finite: bool = True


@dataclass(frozen=True)
class HybridControlOutput:
    phase: ContactPhase
    phase_elapsed_s: float
    desired_tip_coordinate_m: float
    desired_tip_velocity_m_per_s: float
    total_contact_reference_n: float
    probe_variation_reference_n: float
    commanded_normal_force_n: float
    near_loss_active: bool


@dataclass(frozen=True)
class StateTransition:
    time_s: float
    from_phase: str
    to_phase: str
    reason: str
    contact_force_n: float
    penetration_m: float
    relative_normal_velocity_m_per_s: float


class HybridContactController:
    """State machine plus bounded force-to-position contact controller.

    The controller uses only interface stiffness, measured force, target
    displacement, and relative velocity. It never uses target k/c/m values.
    """

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        interface_stiffness_n_per_m: float,
        probe_duration_s: float,
        observation_duration_s: float,
        probe_reference,
        vehicle_mass_kg: float = 1.0,
    ) -> None:
        self.values = values
        self.interface_stiffness_n_per_m = float(interface_stiffness_n_per_m)
        self.probe_duration_s = float(probe_duration_s)
        self.observation_duration_s = float(observation_duration_s)
        self.probe_reference = probe_reference
        self.vehicle_mass_kg = float(vehicle_mass_kg)
        self.phase = ContactPhase.APPROACH
        self.phase_start_s = 0.0
        self.stable_dwell_s = 0.0
        self.integral_force_error_n_s = 0.0
        self.commanded_tip_coordinate_m = -float(values["initial_clearance_m"])
        self.commanded_tip_velocity_m_per_s = 0.0
        self.unload_start_force_n = 0.0
        self.last_force_command_n = 0.0
        self.unload_start_tip_coordinate_m = self.commanded_tip_coordinate_m
        self.previous_reference_n = 0.0
        self.filtered_reference_derivative_n_per_s = 0.0
        self.filtered_contact_force_n = 0.0
        self.filtered_target_acceleration_m_per_s2 = 0.0
        self.filtered_relative_velocity_m_per_s = 0.0
        self.transitions: list[StateTransition] = []

    def _transition(self, observation: ContactObservation, phase: ContactPhase, reason: str) -> None:
        if phase == self.phase:
            return
        self.transitions.append(
            StateTransition(
                observation.time_s,
                self.phase.value,
                phase.value,
                reason,
                observation.contact_force_n,
                observation.penetration_m,
                observation.relative_normal_velocity_m_per_s,
            )
        )
        self.phase = phase
        self.phase_start_s = observation.time_s
        self.stable_dwell_s = 0.0
        if phase == ContactPhase.PASSIVE_OBSERVE:
            # The unload trajectory has already moved to the retracted pose.
            # Freeze that prospective pose for the entire observation window;
            # the position controller may stabilize the vehicle, but this
            # state cannot command target-following motion or force excitation.
            self.commanded_tip_coordinate_m = -float(self.values["passive_retraction_m"])
            self.commanded_tip_velocity_m_per_s = 0.0

    def _safety_transition(self, observation: ContactObservation) -> bool:
        if not observation.finite:
            self._transition(observation, ContactPhase.ABORT, "nonfinite_state")
        elif observation.contact_force_n > float(self.values["abort_force_n"]):
            self._transition(observation, ContactPhase.ABORT, "force_limit")
        elif observation.penetration_m > float(self.values["abort_penetration_m"]):
            self._transition(observation, ContactPhase.ABORT, "penetration_limit")
        elif observation.attitude_deg > float(self.values["abort_attitude_deg"]):
            self._transition(observation, ContactPhase.ABORT, "attitude_limit")
        return self.phase == ContactPhase.ABORT

    def _update_phase(self, observation: ContactObservation, step_s: float) -> None:
        if self.phase in {ContactPhase.DECISION, ContactPhase.ABORT}:
            return
        if self._safety_transition(observation):
            return
        elapsed = observation.time_s - self.phase_start_s
        detected = observation.contact_force_n >= float(self.values["contact_detection_force_n"])
        in_contact = observation.contact_force_n > 0.0
        if self.phase == ContactPhase.APPROACH:
            if detected:
                self._transition(observation, ContactPhase.CONTACT_ACQUIRE, "contact_detected")
        elif self.phase == ContactPhase.CONTACT_ACQUIRE:
            stable = (
                elapsed >= float(self.values["acquisition_ramp_duration_s"])
                and
                in_contact
                and abs(observation.contact_force_n - float(self.values["acquisition_force_n"]))
                <= float(self.values["acquisition_force_error_n"])
                and abs(observation.relative_normal_velocity_m_per_s)
                <= float(self.values["acquisition_relative_velocity_m_per_s"])
            )
            self.stable_dwell_s = self.stable_dwell_s + step_s if stable else 0.0
            if self.stable_dwell_s >= float(self.values["acquisition_minimum_dwell_s"]):
                self._transition(observation, ContactPhase.PRELOAD, "stable_contact_acquired")
            elif elapsed >= float(self.values["acquisition_timeout_s"]):
                self._transition(observation, ContactPhase.ABORT, "acquisition_timeout")
        elif self.phase == ContactPhase.PRELOAD:
            ramp = float(self.values["preload_ramp_duration_s"])
            stable = (
                elapsed >= ramp
                and in_contact
                and abs(observation.contact_force_n - float(self.values["preload_force_n"]))
                <= float(self.values["preload_force_error_n"])
                and abs(observation.relative_normal_velocity_m_per_s)
                <= float(self.values["preload_relative_velocity_m_per_s"])
            )
            self.stable_dwell_s = self.stable_dwell_s + step_s if stable else 0.0
            if self.stable_dwell_s >= float(self.values["preload_minimum_dwell_s"]):
                self._transition(observation, ContactPhase.PROBE, "preload_stable")
            elif elapsed >= float(self.values["preload_timeout_s"]):
                self._transition(observation, ContactPhase.ABORT, "preload_timeout")
        elif self.phase == ContactPhase.PROBE and elapsed >= self.probe_duration_s:
            self.unload_start_force_n = self.last_force_command_n
            self.unload_start_tip_coordinate_m = self.commanded_tip_coordinate_m
            self._transition(observation, ContactPhase.CONTROLLED_UNLOAD, "probe_complete")
        elif self.phase == ContactPhase.CONTROLLED_UNLOAD:
            if elapsed >= float(self.values["unload_duration_s"]):
                self._transition(observation, ContactPhase.PASSIVE_OBSERVE, "unload_complete")
        elif self.phase == ContactPhase.PASSIVE_OBSERVE:
            if elapsed >= self.observation_duration_s:
                self._transition(observation, ContactPhase.DECISION, "observation_complete")

    def _force_reference(self, phase: ContactPhase, elapsed: float) -> tuple[float, float]:
        acquire = float(self.values["acquisition_force_n"])
        preload = float(self.values["preload_force_n"])
        if phase == ContactPhase.CONTACT_ACQUIRE:
            duration = float(self.values["acquisition_ramp_duration_s"])
            fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
            smooth = 0.5 - 0.5 * np.cos(np.pi * fraction)
            start = float(self.values["contact_detection_force_n"])
            return start + smooth * (acquire - start), 0.0
        if phase == ContactPhase.PRELOAD:
            duration = float(self.values["preload_ramp_duration_s"])
            fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
            smooth = 0.5 - 0.5 * np.cos(np.pi * fraction)
            return acquire + smooth * (preload - acquire), 0.0
        if phase == ContactPhase.PROBE:
            variation = float(self.probe_reference(elapsed))
            return preload + variation, variation
        if phase == ContactPhase.CONTROLLED_UNLOAD:
            duration = float(self.values["unload_duration_s"])
            fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
            smooth_remaining = 0.5 + 0.5 * np.cos(np.pi * fraction)
            return self.unload_start_force_n * smooth_remaining, 0.0
        return 0.0, 0.0

    def command(self, observation: ContactObservation, step_s: float) -> HybridControlOutput:
        self._update_phase(observation, step_s)
        elapsed = max(observation.time_s - self.phase_start_s, 0.0)
        reference, variation = self._force_reference(self.phase, elapsed)
        raw_reference_derivative = (reference - self.previous_reference_n) / max(step_s, 1.0e-9)
        derivative_tau = float(self.values["reference_derivative_filter_time_constant_s"])
        derivative_alpha = step_s / (derivative_tau + step_s)
        self.filtered_reference_derivative_n_per_s += derivative_alpha * (
            raw_reference_derivative - self.filtered_reference_derivative_n_per_s
        )
        self.previous_reference_n = reference
        filter_tau = float(self.values["contact_control_filter_time_constant_s"])
        filter_alpha = step_s / (filter_tau + step_s)
        self.filtered_contact_force_n += filter_alpha * (
            observation.contact_force_n - self.filtered_contact_force_n
        )
        self.filtered_target_acceleration_m_per_s2 += filter_alpha * (
            observation.target_acceleration_m_per_s2
            - self.filtered_target_acceleration_m_per_s2
        )
        self.filtered_relative_velocity_m_per_s += filter_alpha * (
            observation.relative_normal_velocity_m_per_s
            - self.filtered_relative_velocity_m_per_s
        )
        near_loss = False

        if self.phase == ContactPhase.APPROACH:
            desired_tip = min(
                self.commanded_tip_coordinate_m
                + float(self.values["approach_velocity_m_per_s"]) * step_s,
                0.003,
            )
            force_command = 0.0
        elif self.phase in {ContactPhase.CONTACT_ACQUIRE, ContactPhase.PRELOAD, ContactPhase.PROBE}:
            error = reference - self.filtered_contact_force_n
            if observation.contact_force_n <= 0.0:
                # Open contact is recovered kinematically with only a small,
                # prebounded attitude-maintenance force; full force feedback
                # remains disabled until unilateral contact exists.
                desired_tip = self.commanded_tip_coordinate_m + float(
                    self.values["reacquisition_velocity_m_per_s"]
                ) * step_s
                force_command = float(self.values["separated_retention_force_command_n"])
            else:
                near_loss = (
                    observation.contact_force_n < float(self.values["near_loss_force_n"])
                    and observation.relative_normal_velocity_m_per_s
                    < float(self.values["near_loss_closing_speed_m_per_s"])
                )
                desired_velocity = (
                    float(self.values["target_velocity_feedforward_gain"])
                    * observation.target_velocity_m_per_s
                    + float(self.values["force_admittance_velocity_gain_m_per_n_s"])
                    * error
                    + float(self.values["interface_force_derivative_feedforward_gain"])
                    * self.filtered_reference_derivative_n_per_s
                    / self.interface_stiffness_n_per_m
                )
                if self.phase in {ContactPhase.CONTACT_ACQUIRE, ContactPhase.PRELOAD}:
                    desired_velocity += float(self.values["contact_compression_velocity_m_per_s"]) * max(
                        error / max(reference, 1.0e-6), 0.0
                    )
                if near_loss:
                    desired_velocity += float(self.values["retention_position_boost_m"]) / max(step_s, 1.0e-9)
                desired_velocity = float(np.clip(
                    desired_velocity,
                    -float(self.values["maximum_commanded_normal_velocity_m_per_s"]),
                    float(self.values["maximum_commanded_normal_velocity_m_per_s"]),
                ))
                desired_tip = self.commanded_tip_coordinate_m + desired_velocity * step_s
                proposed_integral = self.integral_force_error_n_s + error * step_s
                integral_limit = float(self.values["force_integral_limit_n_s"])
                proposed_integral = float(np.clip(proposed_integral, -integral_limit, integral_limit))
                control_mode = str(self.values.get("force_control_mode", "direct"))
                if control_mode == "acceleration_matching":
                    desired_relative_acceleration = (
                        float(self.values["force_error_acceleration_gain_m_per_s2_per_n"])
                        * error
                        - float(self.values["relative_velocity_acceleration_gain_per_s"])
                        * self.filtered_relative_velocity_m_per_s
                    )
                    desired_normal_acceleration = (
                        float(self.values["target_acceleration_feedforward_gain"])
                        * self.filtered_target_acceleration_m_per_s2
                        + desired_relative_acceleration
                        + float(self.values["force_integral_gain_per_s"]) * proposed_integral
                    )
                    acceleration_limit = float(self.values["maximum_normal_control_acceleration_m_per_s2"])
                    desired_normal_acceleration = float(np.clip(
                        desired_normal_acceleration, -acceleration_limit, acceleration_limit
                    ))
                    unsaturated_force = (
                        self.filtered_contact_force_n
                        + self.vehicle_mass_kg * desired_normal_acceleration
                    )
                elif control_mode == "target_following":
                    desired_normal_acceleration = (
                        float(self.values["target_acceleration_feedforward_gain"])
                        * self.filtered_target_acceleration_m_per_s2
                        - float(self.values["relative_velocity_acceleration_gain_per_s"])
                        * self.filtered_relative_velocity_m_per_s
                        + float(self.values["force_error_acceleration_gain_m_per_s2_per_n"])
                        * error
                    )
                    acceleration_limit = float(self.values["maximum_normal_control_acceleration_m_per_s2"])
                    desired_normal_acceleration = float(np.clip(
                        desired_normal_acceleration, -acceleration_limit, acceleration_limit
                    ))
                    unsaturated_force = reference + self.vehicle_mass_kg * desired_normal_acceleration
                else:
                    unsaturated_force = (
                        reference
                        + float(self.values["force_feedforward_error_gain"]) * error
                        + float(self.values["force_integral_gain_per_s"]) * proposed_integral
                        - float(self.values["force_velocity_damping_n_per_m_per_s"])
                        * observation.relative_normal_velocity_m_per_s
                    )
                if near_loss:
                    unsaturated_force += float(self.values["retention_force_boost_n"])
                force_command = float(np.clip(
                    unsaturated_force,
                    float(self.values["minimum_force_command_n"]),
                    float(self.values["maximum_force_command_n"]),
                ))
                saturated_high = unsaturated_force > float(self.values["maximum_force_command_n"])
                saturated_low = unsaturated_force < float(self.values["minimum_force_command_n"])
                if not ((saturated_high and error > 0.0) or (saturated_low and error < 0.0)):
                    self.integral_force_error_n_s = proposed_integral
        elif self.phase == ContactPhase.CONTROLLED_UNLOAD:
            fraction = float(np.clip(
                elapsed / float(self.values["unload_duration_s"]), 0.0, 1.0
            ))
            smooth = 0.5 - 0.5 * np.cos(np.pi * fraction)
            desired_tip = (
                (1.0 - smooth) * self.unload_start_tip_coordinate_m
                - smooth * float(self.values["passive_retraction_m"])
            )
            force_command = reference
            self.integral_force_error_n_s *= max(1.0 - 5.0 * step_s, 0.0)
        else:
            desired_tip = self.commanded_tip_coordinate_m
            force_command = 0.0
            self.integral_force_error_n_s = 0.0

        max_velocity = (
            float(self.values["maximum_approach_velocity_m_per_s"])
            if self.phase == ContactPhase.APPROACH
            else float(self.values["maximum_commanded_normal_velocity_m_per_s"])
        )
        raw_velocity = (desired_tip - self.commanded_tip_coordinate_m) / max(step_s, np.finfo(float).eps)
        velocity_target = float(np.clip(raw_velocity, -max_velocity, max_velocity))
        max_delta = float(self.values["maximum_commanded_normal_acceleration_m_per_s2"]) * step_s
        velocity = float(np.clip(
            velocity_target,
            self.commanded_tip_velocity_m_per_s - max_delta,
            self.commanded_tip_velocity_m_per_s + max_delta,
        ))
        if self.phase == ContactPhase.APPROACH:
            velocity = min(max(velocity, 0.0), float(self.values["maximum_approach_velocity_m_per_s"]))
        self.commanded_tip_velocity_m_per_s = velocity
        self.commanded_tip_coordinate_m += velocity * step_s
        if self.phase in {ContactPhase.PASSIVE_OBSERVE, ContactPhase.DECISION, ContactPhase.ABORT}:
            reported_velocity = 0.0
        else:
            reported_velocity = velocity
        self.last_force_command_n = force_command
        return HybridControlOutput(
            self.phase,
            elapsed,
            self.commanded_tip_coordinate_m,
            reported_velocity,
            reference,
            variation,
            force_command,
            near_loss,
        )
