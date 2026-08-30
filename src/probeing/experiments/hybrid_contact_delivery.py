"""EXP-0009 hybrid contact-retention simulation and delivery metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from probeing.contact import Target1D, TargetParameters, TargetState
from probeing.controllers import (
    ContactObservation,
    ContactPhase,
    HybridContactController,
    StateTransition,
)
from probeing.experiments.coupled_uav_contact import (
    CoupledTrajectory,
    VehicleTrajectory,
    build_contact,
    build_vehicle,
    frozen_policy_prediction,
    initial_hover_state,
    locked_probe_force,
)
from probeing.experiments.decision_sufficiency import _maneuver_signal, _risk_class
from probeing.measurements.causal import estimate_signal_delay
from probeing.vehicles import QuadrotorState, quat_to_rotation, rotation_to_euler_xyz


@dataclass(frozen=True)
class HybridTrajectory:
    time_s: NDArray[np.float64]
    phase: NDArray[np.str_]
    phase_elapsed_s: NDArray[np.float64]
    probe_variation_reference_n: NDArray[np.float64]
    total_contact_reference_n: NDArray[np.float64]
    commanded_normal_force_n: NDArray[np.float64]
    realized_contact_force_n: NDArray[np.float64]
    contact_active: NDArray[np.bool_]
    contact_penetration_m: NDArray[np.float64]
    contact_force_world_n: NDArray[np.float64]
    contact_torque_body_nm: NDArray[np.float64]
    target_displacement_m: NDArray[np.float64]
    target_velocity_m_per_s: NDArray[np.float64]
    target_acceleration_m_per_s2: NDArray[np.float64]
    near_loss_active: NDArray[np.bool_]
    desired_tip_coordinate_m: NDArray[np.float64]
    vehicle: VehicleTrajectory
    transitions: tuple[StateTransition, ...]

    @property
    def completed(self) -> bool:
        return bool(np.any(self.phase == ContactPhase.DECISION.value))

    @property
    def aborted(self) -> bool:
        return bool(np.any(self.phase == ContactPhase.ABORT.value))


def _controller_vehicle_config(config: Mapping[str, Any]) -> dict[str, Any]:
    translated = dict(config)
    translated["controller"] = config["vehicle_controller"]
    return translated


def _frozen_probe_scalar(config: Mapping[str, Any]) -> Callable[[float], float]:
    def reference(local_time_s: float) -> float:
        return float(locked_probe_force(np.asarray([local_time_s]), config)[0])
    return reference


def simulate_hybrid_contact(
    config: Mapping[str, Any],
    target_parameters: TargetParameters,
    *,
    integration_step_s: float | None = None,
    probe_reference: Callable[[float], float] | None = None,
    probe_duration_s: float | None = None,
    post_decision_duration_s: float = 0.0,
) -> HybridTrajectory:
    """Integrate the unmodified EXP-0008 physics with the EXP-0009 controller."""

    physics_config = _controller_vehicle_config(config)
    model, position_controller = build_vehicle(physics_config)
    probe, contact_model = build_contact(physics_config)
    target = Target1D(target_parameters)
    step = float(config["simulation"]["integration_step_s"] if integration_step_s is None else integration_step_s)
    probe_duration = float(config["locked_probe"]["duration_s"] if probe_duration_s is None else probe_duration_s)
    probe_callback = _frozen_probe_scalar(config) if probe_reference is None else probe_reference
    state_controller = HybridContactController(
        config["hybrid_controller"],
        interface_stiffness_n_per_m=float(config["contact"]["interface_stiffness_n_per_m"]),
        probe_duration_s=probe_duration,
        observation_duration_s=float(config["locked_probe"]["observation_duration_s"]),
        probe_reference=probe_callback,
        vehicle_mass_kg=float(config["vehicle"]["mass_kg"]),
    )

    surface = np.asarray(config["contact"]["surface_origin_world_m"], dtype=float)
    normal = np.asarray(config["contact"]["normal_world"], dtype=float)
    normal /= np.linalg.norm(normal)
    base_vehicle_position = surface - probe.tip_offset_body_m
    initial_position = base_vehicle_position - float(config["hybrid_controller"]["initial_clearance_m"]) * normal
    vehicle_state = initial_hover_state(model, initial_position)
    combined = np.concatenate((vehicle_state.vector(), TargetState().vector()))
    maximum_duration = float(config["simulation"]["maximum_experiment_duration_s"]) + max(post_decision_duration_s, 0.0)
    intervals = int(round(maximum_duration / step))

    time: list[float] = []
    phase: list[str] = []
    phase_elapsed: list[float] = []
    variation_reference: list[float] = []
    total_reference: list[float] = []
    force_command: list[float] = []
    actual_force: list[float] = []
    active: list[bool] = []
    penetration: list[float] = []
    contact_force_world: list[NDArray[np.float64]] = []
    contact_torque_body: list[NDArray[np.float64]] = []
    target_x: list[float] = []
    target_v: list[float] = []
    target_a: list[float] = []
    near_loss: list[bool] = []
    desired_tip_coordinate: list[float] = []
    position: list[NDArray[np.float64]] = []
    velocity: list[NDArray[np.float64]] = []
    quaternion: list[NDArray[np.float64]] = []
    euler: list[NDArray[np.float64]] = []
    angular_velocity: list[NDArray[np.float64]] = []
    rotor_speed: list[NDArray[np.float64]] = []
    rotor_thrust: list[NDArray[np.float64]] = []
    desired_position: list[NDArray[np.float64]] = []
    desired_force: list[NDArray[np.float64]] = []
    desired_torque: list[NDArray[np.float64]] = []
    reserve: list[float] = []
    saturated: list[bool] = []
    decision_time: float | None = None

    def evaluate(values: NDArray[np.float64]):
        local_vehicle = QuadrotorState.from_vector(values[:17])
        local_target = TargetState(float(values[17]), float(values[18]))
        return local_vehicle, local_target, contact_model.evaluate(local_vehicle, probe, local_target)

    for index in range(intervals + 1):
        current_time = index * step
        vehicle_state, target_state, contact = evaluate(combined)
        attitude = rotation_to_euler_xyz(quat_to_rotation(vehicle_state.quaternion_wxyz))
        observation = ContactObservation(
            time_s=current_time,
            contact_force_n=contact.contact_force_n,
            penetration_m=contact.penetration_m,
            relative_normal_velocity_m_per_s=contact.closing_speed_m_per_s,
            target_displacement_m=target_state.displacement_m,
            target_velocity_m_per_s=target_state.velocity_m_per_s,
            target_acceleration_m_per_s2=target.acceleration(target_state, contact.contact_force_n),
            attitude_deg=float(np.max(np.abs(np.rad2deg(attitude)))),
            finite=bool(np.all(np.isfinite(combined))),
        )
        control = state_controller.command(observation, step)
        position_reference = base_vehicle_position + control.desired_tip_coordinate_m * normal
        velocity_reference = control.desired_tip_velocity_m_per_s * normal
        if (
            contact.active
            and control.phase
            in {ContactPhase.CONTACT_ACQUIRE, ContactPhase.PRELOAD, ContactPhase.PROBE}
            and str(config["hybrid_controller"].get("force_control_mode", "direct"))
            in {"acceleration_matching", "target_following"}
        ):
            # The normal force loop owns the constrained axis while contact is
            # active. Null its position/velocity errors so the outer position
            # loop cannot fight the acceleration-matching force command.
            position_reference = position_reference + normal * float(
                (vehicle_state.position_world_m - position_reference) @ normal
            )
            velocity_reference = velocity_reference + normal * float(
                (vehicle_state.velocity_world_m_per_s - velocity_reference) @ normal
            )
        output = position_controller.command(
            vehicle_state,
            position_reference,
            velocity_reference,
            desired_yaw_rad=0.0,
            feedforward_force_world_n=control.commanded_normal_force_n * normal,
            measured_external_torque_body_nm=contact.vehicle_torque_body_nm,
        )

        time.append(current_time); phase.append(control.phase.value); phase_elapsed.append(control.phase_elapsed_s)
        variation_reference.append(control.probe_variation_reference_n)
        total_reference.append(control.total_contact_reference_n); force_command.append(control.commanded_normal_force_n)
        actual_force.append(contact.contact_force_n); active.append(contact.active); penetration.append(contact.penetration_m)
        contact_force_world.append(contact.vehicle_force_world_n); contact_torque_body.append(contact.vehicle_torque_body_nm)
        target_x.append(target_state.displacement_m); target_v.append(target_state.velocity_m_per_s)
        target_a.append(target.acceleration(target_state, contact.contact_force_n)); near_loss.append(control.near_loss_active)
        desired_tip_coordinate.append(control.desired_tip_coordinate_m)
        position.append(vehicle_state.position_world_m); velocity.append(vehicle_state.velocity_world_m_per_s)
        quaternion.append(vehicle_state.quaternion_wxyz); euler.append(attitude)
        angular_velocity.append(vehicle_state.angular_velocity_body_rad_s); rotor_speed.append(vehicle_state.rotor_speed_rad_s)
        rotor_thrust.append(model.rotors.wrench(vehicle_state.rotor_speed_rad_s)[0])
        desired_position.append(position_reference); desired_force.append(output.desired_force_world_n)
        desired_torque.append(output.desired_torque_body_nm); reserve.append(output.actuator_reserve); saturated.append(output.saturated)

        if control.phase == ContactPhase.DECISION and decision_time is None:
            decision_time = current_time
        if decision_time is not None and current_time >= decision_time + post_decision_duration_s:
            break
        if control.phase == ContactPhase.ABORT:
            break
        if index == intervals:
            break
        rotor_command = output.commanded_rotor_speed_rad_s

        def derivative(values: NDArray[np.float64]) -> NDArray[np.float64]:
            local_vehicle, local_target, local_contact = evaluate(values)
            vehicle_derivative = model.derivative(
                local_vehicle.vector(),
                rotor_command,
                external_force_world_n=local_contact.vehicle_force_world_n,
                external_torque_body_nm=local_contact.vehicle_torque_body_nm,
            )
            target_derivative = target.derivative(local_target.vector(), local_contact.contact_force_n)
            return np.concatenate((vehicle_derivative, target_derivative))

        k1 = derivative(combined); k2 = derivative(combined + 0.5 * step * k1)
        k3 = derivative(combined + 0.5 * step * k2); k4 = derivative(combined + step * k3)
        combined = combined + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        combined[6:10] /= np.linalg.norm(combined[6:10])
        combined[13:17] = model.rotors.clip_speed(combined[13:17])

    vehicle = VehicleTrajectory(
        np.asarray(time), np.asarray(position), np.asarray(velocity), np.asarray(quaternion),
        np.asarray(euler), np.asarray(angular_velocity), np.asarray(rotor_speed),
        np.asarray(rotor_thrust), np.asarray(desired_position), np.asarray(desired_force),
        np.asarray(desired_torque), np.asarray(reserve), np.asarray(saturated, dtype=bool),
    )
    return HybridTrajectory(
        np.asarray(time), np.asarray(phase), np.asarray(phase_elapsed),
        np.asarray(variation_reference), np.asarray(total_reference), np.asarray(force_command),
        np.asarray(actual_force), np.asarray(active, dtype=bool), np.asarray(penetration),
        np.asarray(contact_force_world), np.asarray(contact_torque_body), np.asarray(target_x),
        np.asarray(target_v), np.asarray(target_a), np.asarray(near_loss, dtype=bool),
        np.asarray(desired_tip_coordinate), vehicle, tuple(state_controller.transitions),
    )


def _transition_indices(contact: NDArray[np.bool_]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    losses = np.flatnonzero(contact[:-1] & ~contact[1:]) + 1
    gains = np.flatnonzero(~contact[:-1] & contact[1:]) + 1
    return losses, gains


def hybrid_delivery_metrics(trajectory: HybridTrajectory, config: Mapping[str, Any]) -> Mapping[str, float | bool]:
    probe = trajectory.phase == ContactPhase.PROBE.value
    unload = trajectory.phase == ContactPhase.CONTROLLED_UNLOAD.value
    passive = trajectory.phase == ContactPhase.PASSIVE_OBSERVE.value
    step = float(np.median(np.diff(trajectory.time_s)))
    first_contact_indices = np.flatnonzero(trajectory.contact_active)
    first_index = int(first_contact_indices[0]) if first_contact_indices.size else 0
    impact_end = min(first_index + int(round(0.10 / step)) + 1, trajectory.time_s.size)
    first_peak = float(np.max(trajectory.realized_contact_force_n[first_index:impact_end])) if first_contact_indices.size else 0.0
    first_impulse = float(np.trapz(trajectory.realized_contact_force_n[first_index:impact_end], dx=step)) if first_contact_indices.size else 0.0
    _, all_gains = _transition_indices(trajectory.contact_active)
    recontact_gains = all_gains[all_gains > first_index + 1]
    recontact_peaks = []
    for gain_index in recontact_gains:
        stop = min(int(gain_index) + int(round(0.05 / step)) + 1, trajectory.time_s.size)
        recontact_peaks.append(float(np.max(trajectory.realized_contact_force_n[gain_index:stop])))
    passive_force = trajectory.realized_contact_force_n[passive]
    unload_force = trajectory.realized_contact_force_n[unload]
    attitude = np.rad2deg(trajectory.vehicle.euler_xyz_rad)
    initial_position = trajectory.vehicle.position_world_m[0]
    common: dict[str, float | bool] = {
        "completed": trajectory.completed,
        "aborted": trajectory.aborted,
        "first_contact_peak_force_n": first_peak,
        "first_contact_impulse_n_s": first_impulse,
        "recontact_peak_force_n": float(max(recontact_peaks, default=0.0)),
        "total_recontact_count": float(recontact_gains.size),
        "unload_impulse_n_s": float(np.trapz(unload_force, dx=step)) if unload_force.size else np.inf,
        "passive_residual_force_rms_n": float(np.sqrt(np.mean(passive_force**2))) if passive_force.size else np.inf,
        "passive_force_squared_energy_n2_s": float(np.trapz(passive_force**2, dx=step)) if passive_force.size else np.inf,
        "passive_recontact": bool(np.any(passive_force > float(config["hybrid_controller"]["contact_detection_force_n"]))) if passive_force.size else True,
        "passive_max_commanded_probe_force_n": float(np.max(np.abs(trajectory.commanded_normal_force_n[passive]))) if passive_force.size else np.inf,
        "peak_contact_force_n": float(np.max(trajectory.realized_contact_force_n)),
        "peak_target_displacement_m": float(np.max(np.abs(trajectory.target_displacement_m))),
        "peak_target_velocity_m_per_s": float(np.max(np.abs(trajectory.target_velocity_m_per_s))),
        "peak_target_acceleration_m_per_s2": float(np.max(np.abs(trajectory.target_acceleration_m_per_s2))),
        "peak_attitude_deg": float(np.max(np.abs(attitude))),
        "peak_angular_rate_rad_s": float(np.max(np.abs(trajectory.vehicle.angular_velocity_body_rad_s))),
        "peak_vehicle_position_disturbance_m": float(np.max(np.linalg.norm(trajectory.vehicle.position_world_m - initial_position, axis=1))),
        "motor_saturation_fraction": float(np.mean(trajectory.vehicle.motor_saturated)),
        "minimum_actuator_reserve": float(np.min(trajectory.vehicle.actuator_reserve)),
    }
    if np.count_nonzero(probe) < 10:
        return {
            **common,
            "probe_rms_tracking_error_n": np.inf,
            "probe_normalized_rms_tracking_error": np.inf,
            "probe_peak_tracking_error_n": np.inf,
            "probe_cross_correlation_lag_s": np.inf,
            "probe_weighted_phase_lag_rad": np.inf,
            "probe_delivery_bandwidth_hz": 0.0,
            "probe_correlation": 0.0,
            "zero_separation_during_probe": False,
            "probe_contact_fraction": 0.0,
            "probe_separation_count": np.inf,
            "probe_recontact_count": np.inf,
            "near_loss_count": 0.0,
        }
    reference = trajectory.total_contact_reference_n[probe]
    variation = trajectory.probe_variation_reference_n[probe]
    actual = trajectory.realized_contact_force_n[probe]
    error = actual - reference
    centered_actual = actual - np.mean(actual); centered_variation = variation - np.mean(variation)
    lag = estimate_signal_delay(variation, actual, step, maximum_delay_s=0.25)
    frequency = np.fft.rfftfreq(reference.size, step)
    ref_fft = np.fft.rfft(centered_variation); actual_fft = np.fft.rfft(centered_actual)
    valid = (frequency >= 0.5) & (frequency <= 5.0) & (np.abs(ref_fft) >= 0.1 * np.max(np.abs(ref_fft)))
    transfer = actual_fft[valid] / ref_fft[valid]
    gain = np.abs(transfer); phase = np.unwrap(np.angle(transfer))
    faithful_values = (gain >= 10 ** (-3 / 20)) & (gain <= 10 ** (3 / 20)) & (np.abs(phase) <= np.pi / 4)
    faithful_bandwidth = float(np.max(frequency[valid][faithful_values])) if np.any(faithful_values) else 0.0
    probe_contact = trajectory.contact_active[probe]
    losses, gains = _transition_indices(probe_contact)
    return {
        **common,
        "probe_rms_tracking_error_n": float(np.sqrt(np.mean(error**2))),
        "probe_normalized_rms_tracking_error": float(np.sqrt(np.mean(error**2)) / max(np.sqrt(np.mean(variation**2)), np.finfo(float).eps)),
        "probe_peak_tracking_error_n": float(np.max(np.abs(error))),
        "probe_cross_correlation_lag_s": float(lag),
        "probe_weighted_phase_lag_rad": float(-np.average(phase, weights=np.abs(ref_fft[valid]) ** 2)) if np.any(valid) else 0.0,
        "probe_delivery_bandwidth_hz": faithful_bandwidth,
        "probe_correlation": float(np.corrcoef(variation, actual)[0, 1]) if np.std(actual) > 0 else 0.0,
        "zero_separation_during_probe": bool(np.all(probe_contact)),
        "probe_contact_fraction": float(np.mean(probe_contact)),
        "probe_separation_count": float(losses.size),
        "probe_recontact_count": float(gains.size),
        "near_loss_count": float(np.count_nonzero(trajectory.near_loss_active[probe])),
    }


def policy_view(trajectory: HybridTrajectory, config: Mapping[str, Any]) -> CoupledTrajectory:
    """Retain frozen 3 s probe and 0.5 s passive feature windows.

    The physical unload interval is intentionally excluded from the frozen
    feature clock. Causal sensing is then run on the probe samples followed by
    the actual passive samples; no feature or policy coefficient is changed.
    """

    step = float(config["simulation"]["integration_step_s"])
    probe_duration = float(config["locked_probe"]["duration_s"])
    observation = float(config["locked_probe"]["observation_duration_s"])
    probe_time = np.arange(int(round(probe_duration / step)) + 1) * step
    passive_time = np.arange(1, int(round(observation / step)) + 1) * step
    full_time = np.concatenate((probe_time, probe_duration + passive_time))
    probe_mask = trajectory.phase == ContactPhase.PROBE.value
    passive_mask = trajectory.phase == ContactPhase.PASSIVE_OBSERVE.value
    if np.count_nonzero(probe_mask) < 3 or np.count_nonzero(passive_mask) < 3:
        raise ValueError("hybrid trajectory did not complete probe and passive phases")

    def extract(values: NDArray[Any]) -> NDArray[Any]:
        source_probe_t = trajectory.phase_elapsed_s[probe_mask]
        source_passive_t = trajectory.phase_elapsed_s[passive_mask]
        values = np.asarray(values)
        if values.ndim == 1:
            left = np.interp(probe_time, source_probe_t, values[probe_mask])
            right = np.interp(passive_time, source_passive_t, values[passive_mask])
            return np.concatenate((left, right))
        columns = [extract(values[:, column]) for column in range(values.shape[1])]
        return np.column_stack(columns)

    variation = extract(trajectory.probe_variation_reference_n)
    variation[full_time > probe_duration] = 0.0
    vehicle = VehicleTrajectory(
        full_time,
        extract(trajectory.vehicle.position_world_m), extract(trajectory.vehicle.velocity_world_m_per_s),
        extract(trajectory.vehicle.quaternion_wxyz), extract(trajectory.vehicle.euler_xyz_rad),
        extract(trajectory.vehicle.angular_velocity_body_rad_s), extract(trajectory.vehicle.rotor_speed_rad_s),
        extract(trajectory.vehicle.rotor_thrust_n), extract(trajectory.vehicle.desired_position_world_m),
        extract(trajectory.vehicle.desired_force_world_n), extract(trajectory.vehicle.desired_torque_body_nm),
        extract(trajectory.vehicle.actuator_reserve), extract(trajectory.vehicle.motor_saturated).astype(bool),
    )
    return CoupledTrajectory(
        full_time, variation, extract(trajectory.realized_contact_force_n),
        extract(trajectory.contact_active).astype(bool), extract(trajectory.contact_penetration_m),
        extract(trajectory.contact_force_world_n), extract(trajectory.contact_torque_body_nm),
        extract(trajectory.target_displacement_m), extract(trajectory.target_velocity_m_per_s),
        extract(trajectory.target_acceleration_m_per_s2), vehicle,
    )


def hybrid_frozen_policy_prediction(
    trajectory: HybridTrajectory,
    target_parameters: TargetParameters,
    *,
    config: Mapping[str, Any],
    target_id: str,
    source_seed: int,
    case_index: int,
    noise_name: str,
    noise_multiplier: float,
    measurement_seed: int,
    repository_root,
    replication_config: Mapping[str, Any],
    policy_cache,
) -> Mapping[str, Any]:
    return frozen_policy_prediction(
        policy_view(trajectory, config), target_parameters,
        target_id=target_id, source_seed=source_seed, case_index=case_index,
        noise_name=noise_name, noise_multiplier=noise_multiplier,
        measurement_seed=measurement_seed, repository_root=repository_root,
        replication_config=replication_config, policy_cache=policy_cache,
    )


def hybrid_future_outcome(
    config: Mapping[str, Any], target_parameters: TargetParameters,
    locked_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], HybridTrajectory]:
    maneuver_time, maneuver_force = _maneuver_signal(locked_config)
    preload = float(config["hybrid_controller"]["preload_force_n"])
    probe_end = float(locked_config["future_contact_maneuver"]["ramp_down_end_s"])

    def variation(local_time_s: float) -> float:
        total = float(np.interp(local_time_s, maneuver_time, maneuver_force))
        return total - preload

    trajectory = simulate_hybrid_contact(
        config, target_parameters, probe_reference=variation,
        probe_duration_s=probe_end, post_decision_duration_s=2.5,
    )
    probe_transition = next((item for item in trajectory.transitions if item.to_phase == ContactPhase.PROBE.value), None)
    if probe_transition is None or trajectory.aborted:
        metrics = {
            "peak_displacement_m": np.inf, "peak_velocity_m_per_s": np.inf,
            "late_hold_oscillation_rms_m": np.inf, "hold_settling_time_s": np.inf,
            "contact_loss_proxy": True, "realized_sustained_force_rms_error_n": np.inf,
            "risk_class": "UNSAFE",
        }
        return metrics, trajectory
    relative_time = trajectory.time_s - probe_transition.time_s
    values = locked_config["future_contact_maneuver"]; envelope = locked_config["risk_envelope"]
    ramp_end = float(values["ramp_up_s"]); hold_end = float(values["hold_end_s"])
    forced = (relative_time >= 0.0) & (relative_time <= hold_end)
    late = (relative_time >= hold_end - 1.0) & (relative_time <= hold_end)
    displacement = trajectory.target_displacement_m; velocity = trajectory.target_velocity_m_per_s
    peak_displacement = float(np.max(np.abs(displacement[forced])))
    peak_velocity = float(np.max(np.abs(velocity[forced])))
    late_center = float(np.mean(displacement[late]))
    oscillation = float(np.sqrt(np.mean((displacement[late] - late_center) ** 2)))
    tolerance = max(float(envelope["settling_displacement_fraction"]) * abs(late_center), float(envelope["settling_displacement_floor_m"]))
    eligible = np.flatnonzero((relative_time >= ramp_end) & (relative_time <= hold_end))
    settled = (np.abs(displacement - late_center) <= tolerance) & (np.abs(velocity) <= float(envelope["settling_velocity_m_per_s"]))
    settling_time = hold_end - ramp_end
    for sample in eligible:
        if bool(np.all(settled[sample:eligible[-1] + 1])):
            settling_time = float(relative_time[sample] - ramp_end)
            break
    probe_mask = (relative_time >= 0.0) & (relative_time <= probe_end)
    desired = np.interp(relative_time[probe_mask], maneuver_time, maneuver_force)
    metrics: dict[str, Any] = {
        "peak_displacement_m": peak_displacement,
        "peak_velocity_m_per_s": peak_velocity,
        "late_hold_oscillation_rms_m": oscillation,
        "hold_settling_time_s": float(settling_time),
        "contact_loss_proxy": bool(peak_displacement > float(envelope["contact_tracking_displacement_m"])),
        "realized_sustained_force_rms_error_n": float(np.sqrt(np.mean((trajectory.realized_contact_force_n[probe_mask] - desired) ** 2))),
    }
    metrics["risk_class"] = _risk_class(metrics, locked_config)
    return metrics, trajectory
