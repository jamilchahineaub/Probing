"""EXP-0008 modular 6-DoF vehicle and coupled-contact simulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.contact import (
    ProbeGeometry, Target1D, TargetParameters, TargetState,
    UnilateralPenaltyContact,
)
from probeing.models import ContactInteractionSimulation, InteractionSimulation
from probeing.measurements import process_causal_sensing
from probeing.measurements.causal import estimate_signal_delay
from probeing.experiments.decision_sufficiency import (
    TargetCase, _noise, _maneuver_signal, _risk_class,
)
from probeing.experiments.passive_ringdown import (
    FEATURE_SETS, _chirp_features_and_diagnostics, _predict, ringdown_features,
)
from probeing.experiments.locked_policy_replication import (
    _binary_metrics, _load_locked_policy,
)
from probeing.controllers import CascadedController, CascadedControllerGains
from probeing.vehicles import (
    QuadrotorModel, QuadrotorParameters, QuadrotorState, RotorGeometry,
    RotorModel, RotorParameters, quat_to_rotation, rotation_to_euler_xyz,
    quat_from_axis_angle,
)


@dataclass(frozen=True)
class VehicleTrajectory:
    time_s: NDArray[np.float64]
    position_world_m: NDArray[np.float64]
    velocity_world_m_per_s: NDArray[np.float64]
    quaternion_wxyz: NDArray[np.float64]
    euler_xyz_rad: NDArray[np.float64]
    angular_velocity_body_rad_s: NDArray[np.float64]
    rotor_speed_rad_s: NDArray[np.float64]
    rotor_thrust_n: NDArray[np.float64]
    desired_position_world_m: NDArray[np.float64]
    desired_force_world_n: NDArray[np.float64]
    desired_torque_body_nm: NDArray[np.float64]
    actuator_reserve: NDArray[np.float64]
    motor_saturated: NDArray[np.bool_]


@dataclass(frozen=True)
class CoupledTrajectory:
    time_s: NDArray[np.float64]
    desired_probe_force_n: NDArray[np.float64]
    realized_contact_force_n: NDArray[np.float64]
    contact_active: NDArray[np.bool_]
    contact_penetration_m: NDArray[np.float64]
    contact_force_world_n: NDArray[np.float64]
    contact_torque_body_nm: NDArray[np.float64]
    target_displacement_m: NDArray[np.float64]
    target_velocity_m_per_s: NDArray[np.float64]
    target_acceleration_m_per_s2: NDArray[np.float64]
    vehicle: VehicleTrajectory


def build_contact(config: Mapping[str, Any]) -> tuple[ProbeGeometry, UnilateralPenaltyContact]:
    probe_values = config["probe_geometry"]
    contact_values = config["contact"]
    probe = ProbeGeometry(
        np.asarray(probe_values["tip_offset_body_m"], dtype=float)
        - np.asarray(config["vehicle"]["center_of_mass_body_m"], dtype=float),
        probe_mass_kg=float(probe_values["probe_mass_kg"]),
    )
    contact = UnilateralPenaltyContact(
        surface_origin_world_m=np.asarray(contact_values["surface_origin_world_m"], dtype=float),
        normal_world=np.asarray(contact_values["normal_world"], dtype=float),
        interface_stiffness_n_per_m=float(contact_values["interface_stiffness_n_per_m"]),
        interface_damping_n_s_per_m=float(contact_values["interface_damping_n_s_per_m"]),
    )
    return probe, contact


def locked_probe_force(time_s: ArrayLike, config: Mapping[str, Any]) -> NDArray[np.float64]:
    values = config["locked_probe"]
    time = np.asarray(time_s, dtype=float)
    duration = float(values["duration_s"])
    slope = (float(values["end_frequency_hz"]) - float(values["start_frequency_hz"])) / duration
    phase = 2.0 * np.pi * (float(values["start_frequency_hz"]) * time + 0.5 * slope * time**2)
    force = np.zeros_like(time)
    active = time <= duration
    signed = float(values["amplitude_n"]) * np.sin(phase[active])
    force[active] = np.maximum(signed, 0.0)
    force[np.abs(force) < 1.0e-15] = 0.0
    return force


def build_vehicle(config: Mapping[str, Any]) -> tuple[QuadrotorModel, CascadedController]:
    values = config["vehicle"]
    center_of_mass = np.asarray(values["center_of_mass_body_m"], dtype=float)
    geometry = RotorGeometry(
        positions_body_m=np.asarray(values["rotor_positions_body_m"], dtype=float) - center_of_mass,
        spin_directions=np.asarray(values["rotor_spin_directions"], dtype=float),
    )
    rotors = RotorModel(
        RotorParameters(
            thrust_coefficient_n_per_rad2=float(values["thrust_coefficient_n_per_rad2"]),
            torque_coefficient_nm_per_rad2=float(values["reaction_torque_coefficient_nm_per_rad2"]),
            motor_time_constant_s=float(values["motor_time_constant_s"]),
            minimum_speed_rad_s=float(values["minimum_rotor_speed_rad_s"]),
            maximum_speed_rad_s=float(values["maximum_rotor_speed_rad_s"]),
        ), geometry,
    )
    model = QuadrotorModel(
        QuadrotorParameters(
            mass_kg=float(values["mass_kg"]),
            inertia_body_kg_m2=np.asarray(values["inertia_body_kg_m2"], dtype=float),
            center_of_mass_body_m=center_of_mass,
        ), rotors,
    )
    gains = config["controller"]
    controller = CascadedController(
        model,
        CascadedControllerGains(
            position_p=np.asarray(gains["position_p"], dtype=float),
            velocity_d=np.asarray(gains["velocity_d"], dtype=float),
            attitude_p=np.asarray(gains["attitude_p"], dtype=float),
            body_rate_d=np.asarray(gains["body_rate_d"], dtype=float),
            maximum_acceleration_m_per_s2=float(gains["maximum_acceleration_m_per_s2"]),
        ),
    )
    return model, controller


def initial_hover_state(model: QuadrotorModel, position_world_m: ArrayLike) -> QuadrotorState:
    return QuadrotorState(
        position_world_m=np.asarray(position_world_m, dtype=float),
        velocity_world_m_per_s=np.zeros(3),
        quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body_rad_s=np.zeros(3),
        rotor_speed_rad_s=np.full(4, model.hover_speed_rad_s()),
    )


ReferenceCallback = Callable[[float, QuadrotorState], tuple[NDArray[np.float64], NDArray[np.float64], float]]


def simulate_no_contact(
    model: QuadrotorModel, controller: CascadedController,
    initial_state: QuadrotorState, *, duration_s: float, step_s: float,
    reference: ReferenceCallback,
) -> VehicleTrajectory:
    intervals = int(round(duration_s / step_s))
    time = np.arange(intervals + 1, dtype=float) * step_s
    position = np.zeros((time.size, 3)); velocity = np.zeros_like(position)
    quaternion = np.zeros((time.size, 4)); euler = np.zeros_like(position)
    angular_velocity = np.zeros_like(position); rotor_speed = np.zeros((time.size, 4))
    rotor_thrust = np.zeros_like(rotor_speed); desired_position = np.zeros_like(position)
    desired_force = np.zeros_like(position); desired_torque = np.zeros_like(position)
    reserve = np.zeros(time.size); saturated = np.zeros(time.size, dtype=bool)
    state = initial_state
    for index, current_time in enumerate(time):
        target_position, target_velocity, target_yaw = reference(float(current_time), state)
        output = controller.command(state, target_position, target_velocity, desired_yaw_rad=target_yaw)
        position[index] = state.position_world_m; velocity[index] = state.velocity_world_m_per_s
        quaternion[index] = state.quaternion_wxyz
        euler[index] = rotation_to_euler_xyz(quat_to_rotation(state.quaternion_wxyz))
        angular_velocity[index] = state.angular_velocity_body_rad_s
        rotor_speed[index] = state.rotor_speed_rad_s
        rotor_thrust[index] = model.rotors.wrench(state.rotor_speed_rad_s)[0]
        desired_position[index] = target_position; desired_force[index] = output.desired_force_world_n
        desired_torque[index] = output.desired_torque_body_nm
        reserve[index] = output.actuator_reserve; saturated[index] = output.saturated
        if index < intervals:
            state = model.rk4_step(state, output.commanded_rotor_speed_rad_s, step_s)
    return VehicleTrajectory(
        time, position, velocity, quaternion, euler, angular_velocity, rotor_speed,
        rotor_thrust, desired_position, desired_force, desired_torque, reserve, saturated,
    )


def simulate_attitude_step(
    model: QuadrotorModel, controller: CascadedController,
    initial_state: QuadrotorState, *, duration_s: float, step_s: float,
    roll_step_rad: float,
) -> VehicleTrajectory:
    """Exercise only the inner attitude/rate loop at nominal hover thrust."""

    intervals = int(round(duration_s / step_s)); time = np.arange(intervals + 1) * step_s
    position = np.zeros((time.size, 3)); velocity = np.zeros_like(position)
    quaternion = np.zeros((time.size, 4)); euler = np.zeros_like(position)
    angular_velocity = np.zeros_like(position); rotor_speed = np.zeros((time.size, 4))
    rotor_thrust = np.zeros_like(rotor_speed); desired_position = np.zeros_like(position)
    desired_force = np.zeros_like(position); desired_torque = np.zeros_like(position)
    reserve = np.zeros(time.size); saturated = np.zeros(time.size, dtype=bool)
    state = initial_state; hover_force = np.asarray([0.0, 0.0, model.parameters.mass_kg * model.parameters.gravity_m_per_s2])
    for index, current_time in enumerate(time):
        desired_rotation = np.eye(3) if current_time < 0.5 else quat_to_rotation(quat_from_axis_angle([1.0, 0.0, 0.0], roll_step_rad))
        output = controller.attitude_command(state, hover_force, desired_rotation)
        position[index] = state.position_world_m; velocity[index] = state.velocity_world_m_per_s
        quaternion[index] = state.quaternion_wxyz; euler[index] = rotation_to_euler_xyz(quat_to_rotation(state.quaternion_wxyz))
        angular_velocity[index] = state.angular_velocity_body_rad_s; rotor_speed[index] = state.rotor_speed_rad_s
        rotor_thrust[index] = model.rotors.wrench(state.rotor_speed_rad_s)[0]
        desired_position[index] = initial_state.position_world_m; desired_force[index] = hover_force
        desired_torque[index] = output.desired_torque_body_nm; reserve[index] = output.actuator_reserve
        saturated[index] = output.saturated
        if index < intervals:
            state = model.rk4_step(state, output.commanded_rotor_speed_rad_s, step_s)
    return VehicleTrajectory(time, position, velocity, quaternion, euler, angular_velocity, rotor_speed, rotor_thrust, desired_position, desired_force, desired_torque, reserve, saturated)


def simulate_coupled_contact(
    config: Mapping[str, Any], target_parameters: TargetParameters, *,
    integration_step_s: float | None = None,
    force_time_s: ArrayLike | None = None,
    desired_force_n: ArrayLike | None = None,
    passive_start_s: float | None = None,
    normal_force_limit_n: float | None = None,
) -> CoupledTrajectory:
    """Integrate vehicle, lagged rotors, interface, and target as one state."""

    model, controller = build_vehicle(config)
    probe, contact_model = build_contact(config)
    target = Target1D(target_parameters)
    step = float(config["simulation"]["integration_step_s"] if integration_step_s is None else integration_step_s)
    probe_values = config["locked_probe"]
    if (force_time_s is None) != (desired_force_n is None):
        raise ValueError("force_time_s and desired_force_n must be provided together")
    duration = (
        float(probe_values["duration_s"]) + float(probe_values["observation_duration_s"])
        if force_time_s is None else float(np.asarray(force_time_s, dtype=float)[-1])
    )
    intervals = int(round(duration / step)); time = np.arange(intervals + 1) * step
    force_reference = (
        locked_probe_force(time, config)
        if force_time_s is None
        else np.interp(time, np.asarray(force_time_s, dtype=float), np.asarray(desired_force_n, dtype=float))
    )
    passive_start = float(probe_values["duration_s"] if passive_start_s is None else passive_start_s)
    surface = np.asarray(config["contact"]["surface_origin_world_m"], dtype=float)
    normal = np.asarray(config["contact"]["normal_world"], dtype=float); normal /= np.linalg.norm(normal)
    initial_position = surface - probe.tip_offset_body_m
    vehicle_state = initial_hover_state(model, initial_position)
    combined = np.concatenate((vehicle_state.vector(), TargetState().vector()))

    n = time.size
    position = np.zeros((n, 3)); velocity = np.zeros_like(position); quaternion = np.zeros((n, 4))
    euler = np.zeros_like(position); angular_velocity = np.zeros_like(position)
    rotor_speed = np.zeros((n, 4)); rotor_thrust = np.zeros_like(rotor_speed)
    desired_position = np.zeros_like(position); desired_force = np.zeros_like(position)
    desired_torque = np.zeros_like(position); reserve = np.zeros(n); saturated = np.zeros(n, dtype=bool)
    actual_force = np.zeros(n); active = np.zeros(n, dtype=bool); penetration = np.zeros(n)
    contact_force_world = np.zeros_like(position); contact_torque_body = np.zeros_like(position)
    target_x = np.zeros(n); target_v = np.zeros(n); target_a = np.zeros(n)
    initial_vehicle_position = initial_position.copy()
    control_values = config["controller"]
    normal_position_offset = 0.0

    def evaluate_contact(values: NDArray[np.float64]):
        vehicle = QuadrotorState.from_vector(values[:17])
        target_state = TargetState(float(values[17]), float(values[18]))
        return vehicle, target_state, contact_model.evaluate(vehicle, probe, target_state)

    for index, current_time in enumerate(time):
        vehicle_state, target_state, contact = evaluate_contact(combined)
        reference_force = float(force_reference[index])
        passive = current_time > passive_start
        force_error = reference_force - contact.contact_force_n
        if index > 0:
            # Anti-windup: feedforward creates compression. The admittance is
            # reserved for unloading excess force and never integrates a
            # positive error into deeper penetration during contact onset.
            normal_position_offset += float(control_values["force_admittance_gain_m_per_n_s"]) * min(force_error, 0.0) * step
            position_limit = float(control_values["force_admittance_position_limit_m"])
            normal_position_offset = float(np.clip(normal_position_offset, -position_limit, position_limit))
        position_reference = initial_vehicle_position + normal_position_offset * normal
        if passive:
            normal_position_offset = min(normal_position_offset, -float(control_values["passive_retraction_m"]))
            position_reference = initial_vehicle_position + normal_position_offset * normal
        normal_feedforward = reference_force + float(control_values["force_feedback_gain"]) * force_error
        force_limit = float(control_values["maximum_normal_feedforward_force_n"] if normal_force_limit_n is None else normal_force_limit_n)
        normal_feedforward = float(np.clip(normal_feedforward, -force_limit, force_limit))
        if passive:
            normal_feedforward = 0.0
        output = controller.command(
            vehicle_state, position_reference, np.zeros(3), desired_yaw_rad=0.0,
            feedforward_force_world_n=normal_feedforward * normal,
            measured_external_torque_body_nm=contact.vehicle_torque_body_nm,
        )
        position[index] = vehicle_state.position_world_m; velocity[index] = vehicle_state.velocity_world_m_per_s
        quaternion[index] = vehicle_state.quaternion_wxyz; euler[index] = rotation_to_euler_xyz(quat_to_rotation(vehicle_state.quaternion_wxyz))
        angular_velocity[index] = vehicle_state.angular_velocity_body_rad_s
        rotor_speed[index] = vehicle_state.rotor_speed_rad_s; rotor_thrust[index] = model.rotors.wrench(vehicle_state.rotor_speed_rad_s)[0]
        desired_position[index] = position_reference; desired_force[index] = output.desired_force_world_n
        desired_torque[index] = output.desired_torque_body_nm; reserve[index] = output.actuator_reserve; saturated[index] = output.saturated
        actual_force[index] = contact.contact_force_n; active[index] = contact.active; penetration[index] = contact.penetration_m
        contact_force_world[index] = contact.vehicle_force_world_n; contact_torque_body[index] = contact.vehicle_torque_body_nm
        target_x[index] = target_state.displacement_m; target_v[index] = target_state.velocity_m_per_s
        target_a[index] = target.acceleration(target_state, contact.contact_force_n)
        if index >= intervals:
            continue
        rotor_command = output.commanded_rotor_speed_rad_s

        def derivative(values: NDArray[np.float64]) -> NDArray[np.float64]:
            local_vehicle, local_target, local_contact = evaluate_contact(values)
            vehicle_derivative = model.derivative(
                local_vehicle.vector(), rotor_command,
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

    vehicle = VehicleTrajectory(time, position, velocity, quaternion, euler, angular_velocity, rotor_speed, rotor_thrust, desired_position, desired_force, desired_torque, reserve, saturated)
    return CoupledTrajectory(
        time, force_reference, actual_force, active, penetration, contact_force_world,
        contact_torque_body, target_x, target_v, target_a, vehicle,
    )


def probe_tracking_metrics(trajectory: CoupledTrajectory, config: Mapping[str, Any]) -> Mapping[str, float]:
    duration = float(config["locked_probe"]["duration_s"])
    mask = trajectory.time_s <= duration
    reference = trajectory.desired_probe_force_n[mask]; actual = trajectory.realized_contact_force_n[mask]
    error = actual - reference; step = float(trajectory.time_s[1] - trajectory.time_s[0])
    centered_reference = reference - np.mean(reference); centered_actual = actual - np.mean(actual)
    lag_s = estimate_signal_delay(reference, actual, step, maximum_delay_s=0.25)
    frequency = np.fft.rfftfreq(reference.size, step)
    ref_fft = np.fft.rfft(centered_reference); actual_fft = np.fft.rfft(centered_actual)
    valid = (frequency >= 0.5) & (frequency <= 5.0) & (np.abs(ref_fft) >= 0.1 * np.max(np.abs(ref_fft)))
    transfer = actual_fft[valid] / ref_fft[valid]
    gain = np.abs(transfer); phase = np.unwrap(np.angle(transfer))
    faithful = valid.copy()
    faithful[valid] = (gain >= 10 ** (-3 / 20)) & (gain <= 10 ** (3 / 20)) & (np.abs(phase) <= np.pi / 4)
    bandwidth = float(np.max(frequency[faithful])) if np.any(faithful) else 0.0
    return {
        "probe_rms_tracking_error_n": float(np.sqrt(np.mean(error**2))),
        "probe_peak_tracking_error_n": float(np.max(np.abs(error))),
        "probe_relative_rms_tracking_error": float(np.sqrt(np.mean(error**2)) / max(np.sqrt(np.mean(reference**2)), np.finfo(float).eps)),
        "probe_cross_correlation_lag_s": float(lag_s),
        "probe_weighted_phase_lag_rad": float(-np.average(phase, weights=np.abs(ref_fft[valid]) ** 2)) if np.any(valid) else 0.0,
        "probe_delivery_bandwidth_hz": bandwidth,
        "contact_fraction": float(np.mean(trajectory.contact_active[mask])),
        "contact_loss_count": float(np.count_nonzero(trajectory.contact_active[:-1] & ~trajectory.contact_active[1:])),
        "peak_contact_force_n": float(np.max(actual)),
        "peak_penetration_m": float(np.max(trajectory.contact_penetration_m)),
        "peak_target_displacement_m": float(np.max(np.abs(trajectory.target_displacement_m))),
        "peak_target_velocity_m_per_s": float(np.max(np.abs(trajectory.target_velocity_m_per_s))),
        "peak_target_acceleration_m_per_s2": float(np.max(np.abs(trajectory.target_acceleration_m_per_s2))),
        "peak_attitude_deg": float(np.max(np.abs(np.rad2deg(trajectory.vehicle.euler_xyz_rad)))),
        "peak_angular_rate_rad_s": float(np.max(np.abs(trajectory.vehicle.angular_velocity_body_rad_s))),
        "motor_saturation_fraction": float(np.mean(trajectory.vehicle.motor_saturated)),
        "minimum_actuator_reserve": float(np.min(trajectory.vehicle.actuator_reserve)),
        "peak_contact_torque_nm": float(np.max(np.linalg.norm(trajectory.contact_torque_body_nm, axis=1))),
    }


def contact_mechanics_validation(
    config: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, float | str | bool], ...], Mapping[str, NDArray[np.float64]]]:
    """Validate static contact, offset wrench, separation, ring-down, and energy."""

    model, _ = build_vehicle(config); probe, contact_model = build_contact(config)
    surface = np.asarray(config["contact"]["surface_origin_world_m"], dtype=float)
    base_position = surface - probe.tip_offset_body_m
    vehicle = initial_hover_state(model, base_position)
    through_com = ProbeGeometry(np.asarray([0.30, 0.0, 0.0]))
    through_state = initial_hover_state(model, surface - through_com.tip_offset_body_m)
    through_result = contact_model.evaluate(
        QuadrotorState(
            through_state.position_world_m + np.asarray([0.001, 0.0, 0.0]),
            through_state.velocity_world_m_per_s, through_state.quaternion_wxyz,
            through_state.angular_velocity_body_rad_s, through_state.rotor_speed_rad_s,
        ), through_com, TargetState(),
    )
    offset_state = QuadrotorState(
        vehicle.position_world_m + np.asarray([0.001, 0.0, 0.0]), vehicle.velocity_world_m_per_s,
        vehicle.quaternion_wxyz, vehicle.angular_velocity_body_rad_s, vehicle.rotor_speed_rad_s,
    )
    offset_result = contact_model.evaluate(offset_state, probe, TargetState())
    cross_expected = np.cross(probe.tip_offset_body_m, offset_result.vehicle_force_world_n)
    above_probe = ProbeGeometry(np.asarray([0.30, 0.0, 0.08]))
    above_vehicle = initial_hover_state(model, surface - above_probe.tip_offset_body_m + np.asarray([0.001, 0.0, 0.0]))
    above_result = contact_model.evaluate(above_vehicle, above_probe, TargetState())
    separated_state = QuadrotorState(
        vehicle.position_world_m - np.asarray([0.001, 0.0, 0.0]), vehicle.velocity_world_m_per_s,
        vehicle.quaternion_wxyz, vehicle.angular_velocity_body_rad_s, vehicle.rotor_speed_rad_s,
    )
    separated = contact_model.evaluate(separated_state, probe, TargetState())

    target = Target1D(TargetParameters(300.0, 4.0, 1.2))
    interface_k = float(config["contact"]["interface_stiffness_n_per_m"])
    imposed = 0.004
    fixed_vehicle = QuadrotorState(
        vehicle.position_world_m + imposed * contact_model.normal_world,
        np.zeros(3), vehicle.quaternion_wxyz, np.zeros(3), vehicle.rotor_speed_rad_s,
    )
    step = 0.0005; time = np.arange(int(round(5.0 / step)) + 1) * step
    state_vector = TargetState().vector(); displacement = np.zeros(time.size)
    force = np.zeros(time.size); total_energy = np.zeros(time.size)

    def contact_force(values: NDArray[np.float64]) -> float:
        return contact_model.evaluate(fixed_vehicle, probe, TargetState(float(values[0]), float(values[1]))).contact_force_n

    for index in range(time.size):
        target_state = TargetState(float(state_vector[0]), float(state_vector[1]))
        result = contact_model.evaluate(fixed_vehicle, probe, target_state)
        displacement[index] = target_state.displacement_m; force[index] = result.contact_force_n
        total_energy[index] = target.mechanical_energy_j(target_state) + result.interface_elastic_energy_j
        if index + 1 == time.size:
            continue
        def derivative(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return target.derivative(values, contact_force(values))
        k1 = derivative(state_vector); k2 = derivative(state_vector + 0.5 * step * k1)
        k3 = derivative(state_vector + 0.5 * step * k2); k4 = derivative(state_vector + step * k3)
        state_vector = state_vector + step * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    expected_force = imposed / (1.0 / target.parameters.stiffness_n_per_m + 1.0 / interface_k)
    equilibrium_error = float(abs(force[-1] - expected_force))
    energy_growth = float(np.max(np.diff(total_energy)))

    ring_target = Target1D(TargetParameters(180.0, 2.0, 1.5))
    ring_time = np.arange(int(round(3.0 / 0.001)) + 1) * 0.001
    ring_state = TargetState(0.01, 0.0).vector(); ring_numeric = np.zeros(ring_time.size)
    for index in range(ring_time.size):
        ring_numeric[index] = ring_state[0]
        if index + 1 == ring_time.size: continue
        derivative = lambda values: ring_target.derivative(values, 0.0)
        h = 0.001; k1 = derivative(ring_state); k2 = derivative(ring_state + 0.5*h*k1)
        k3 = derivative(ring_state + 0.5*h*k2); k4 = derivative(ring_state + h*k3)
        ring_state = ring_state + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    p = ring_target.parameters; wn = np.sqrt(p.stiffness_n_per_m / p.effective_mass_kg)
    decay = p.damping_n_s_per_m / (2.0 * p.effective_mass_kg); wd = np.sqrt(wn**2 - decay**2)
    ring_analytic = 0.01 * np.exp(-decay * ring_time) * (np.cos(wd * ring_time) + decay / wd * np.sin(wd * ring_time))
    ring_error = float(np.max(np.abs(ring_numeric - ring_analytic)))
    rows: tuple[Mapping[str, float | str | bool], ...] = (
        {"test": "zero_force_before_contact", "metric": "force_n", "value": separated.contact_force_n, "limit": 0.0, "passed": separated.contact_force_n == 0.0},
        {"test": "normal_force_direction", "metric": "vehicle_force_x_n", "value": offset_result.vehicle_force_world_n[0], "limit": 0.0, "passed": offset_result.vehicle_force_world_n[0] < 0.0},
        {"test": "through_com_torque", "metric": "torque_norm_nm", "value": float(np.linalg.norm(through_result.vehicle_torque_body_nm)), "limit": 1e-12, "passed": float(np.linalg.norm(through_result.vehicle_torque_body_nm)) < 1e-12},
        {"test": "offset_cross_product", "metric": "torque_error_nm", "value": float(np.linalg.norm(offset_result.vehicle_torque_body_nm - cross_expected)), "limit": 1e-12, "passed": np.allclose(offset_result.vehicle_torque_body_nm, cross_expected, atol=1e-12)},
        {"test": "offset_pitch_sign", "metric": "pitch_torque_nm", "value": offset_result.vehicle_torque_body_nm[1], "limit": 0.0, "passed": offset_result.vehicle_torque_body_nm[1] > 0.0},
        {"test": "above_below_pitch_sign", "metric": "torque_product_nm2", "value": float(offset_result.vehicle_torque_body_nm[1] * above_result.vehicle_torque_body_nm[1]), "limit": 0.0, "passed": offset_result.vehicle_torque_body_nm[1] * above_result.vehicle_torque_body_nm[1] < 0.0},
        {"test": "stationary_press_equilibrium", "metric": "force_error_n", "value": equilibrium_error, "limit": 1e-6, "passed": equilibrium_error < 1e-6},
        {"test": "contact_energy", "metric": "maximum_step_energy_growth_j", "value": energy_growth, "limit": 1e-10, "passed": energy_growth < 1e-10},
        {"test": "release_ringdown", "metric": "max_analytic_error_m", "value": ring_error, "limit": 1e-9, "passed": ring_error < 1e-9},
    )
    raw = {"press_time_s": time, "press_displacement_m": displacement, "press_force_n": force, "press_total_energy_j": total_energy, "ring_time_s": ring_time, "ring_numeric_m": ring_numeric, "ring_analytic_m": ring_analytic}
    return rows, raw


def trajectory_as_stage1_interaction(
    trajectory: CoupledTrajectory, target_parameters: TargetParameters,
) -> ContactInteractionSimulation:
    response = InteractionSimulation(
        time_s=trajectory.time_s,
        applied_force_n=trajectory.realized_contact_force_n,
        displacement_m=trajectory.target_displacement_m,
        velocity_m_per_s=trajectory.target_velocity_m_per_s,
        acceleration_m_per_s2=trajectory.target_acceleration_m_per_s2,
        kinetic_energy_j=0.5 * target_parameters.effective_mass_kg * trajectory.target_velocity_m_per_s**2,
        elastic_energy_j=0.5 * target_parameters.stiffness_n_per_m * trajectory.target_displacement_m**2,
    )
    return ContactInteractionSimulation(
        response=response,
        commanded_force_n=trajectory.desired_probe_force_n,
        contact_force_n=trajectory.realized_contact_force_n,
        contact_active=trajectory.contact_active,
        contact_mode="unilateral",
        contact_loss_count=int(np.count_nonzero(trajectory.contact_active[:-1] & ~trajectory.contact_active[1:])),
    )


def frozen_policy_prediction(
    trajectory: CoupledTrajectory, target_parameters: TargetParameters, *,
    target_id: str, source_seed: int, case_index: int, noise_name: str,
    noise_multiplier: float, measurement_seed: int, repository_root: Any,
    replication_config: Mapping[str, Any],
    policy_cache: tuple[Mapping[str, Any], Any, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Apply the immutable EXP-0006 policy without fitting or threshold changes."""

    locked_config, model, bundle = (
        _load_locked_policy(replication_config, repository_root)
        if policy_cache is None else policy_cache
    )
    truth = trajectory_as_stage1_interaction(trajectory, target_parameters)
    case = TargetCase(target_id, "stage3a", int(source_seed), int(case_index), target_parameters.stiffness_n_per_m, target_parameters.damping_n_s_per_m, target_parameters.effective_mass_kg)
    common = dict(
        sample_rate_hz=float(locked_config["sensing"]["sample_rate_hz"]),
        noise=_noise(locked_config, float(noise_multiplier)),
        pipeline_settings=locked_config["sensing"]["pipeline_settings"],
        random_seed=int(measurement_seed),
        timestamp_offsets_s={"displacement": 0.0, "velocity": 0.0, "acceleration": 0.0, "force": 0.0},
    )
    primary = process_causal_sensing(truth, pipeline=str(locked_config["sensing"]["primary_pipeline"]), **common)
    stiffness = process_causal_sensing(truth, pipeline="alpha_beta_gamma", **common)
    chirp = _chirp_features_and_diagnostics(primary, stiffness, case, locked_config)
    observation = float(bundle["observation_duration_s"])
    ring = ringdown_features(primary.measurements, float(locked_config["probe"]["duration_s"]), observation, locked_config)
    feature = {
        "target_id": target_id, "partition": "stage3a", "source_seed": int(source_seed),
        "case_index": int(case_index), "noise_regime": noise_name,
        "noise_multiplier": float(noise_multiplier), "observation_duration_s": observation,
        "true_stiffness_n_per_m": target_parameters.stiffness_n_per_m,
        "true_damping_n_s_per_m": target_parameters.damping_n_s_per_m,
        "true_effective_mass_kg": target_parameters.effective_mass_kg,
        "probe_force_squared_dose_n2_s": float(np.trapz(trajectory.realized_contact_force_n**2, trajectory.time_s)),
        **chirp, **ring,
    }
    prediction = _predict(model, feature, FEATURE_SETS[str(bundle["feature_set"])], locked_config)
    return {**feature, **prediction}


def coupled_future_outcome(
    config: Mapping[str, Any], target_parameters: TargetParameters,
    locked_config: Mapping[str, Any], *, integration_step_s: float | None = None,
) -> tuple[Mapping[str, Any], CoupledTrajectory]:
    maneuver_time, maneuver_force = _maneuver_signal(locked_config)
    trajectory = simulate_coupled_contact(
        config, target_parameters, integration_step_s=integration_step_s,
        force_time_s=maneuver_time, desired_force_n=maneuver_force,
        passive_start_s=float(locked_config["future_contact_maneuver"]["ramp_down_end_s"]),
        normal_force_limit_n=3.0,
    )
    values = locked_config["future_contact_maneuver"]; envelope = locked_config["risk_envelope"]
    time = trajectory.time_s; displacement = trajectory.target_displacement_m; velocity = trajectory.target_velocity_m_per_s
    ramp_end = float(values["ramp_up_s"]); hold_end = float(values["hold_end_s"])
    forced = time <= hold_end; late = (time >= hold_end - 1.0) & (time <= hold_end)
    peak_displacement = float(np.max(np.abs(displacement[forced]))); peak_velocity = float(np.max(np.abs(velocity[forced])))
    late_center = float(np.mean(displacement[late])); oscillation = float(np.sqrt(np.mean((displacement[late] - late_center) ** 2)))
    tolerance = max(float(envelope["settling_displacement_fraction"]) * abs(late_center), float(envelope["settling_displacement_floor_m"]))
    eligible = np.flatnonzero((time >= ramp_end) & (time <= hold_end))
    settled = (np.abs(displacement - late_center) <= tolerance) & (np.abs(velocity) <= float(envelope["settling_velocity_m_per_s"]))
    settling_time = hold_end - ramp_end
    for sample in eligible:
        if bool(np.all(settled[sample:eligible[-1] + 1])):
            settling_time = float(time[sample] - ramp_end); break
    contact_loss = bool(peak_displacement > float(envelope["contact_tracking_displacement_m"]))
    metrics: dict[str, Any] = {
        "peak_displacement_m": peak_displacement, "peak_velocity_m_per_s": peak_velocity,
        "late_hold_oscillation_rms_m": oscillation, "hold_settling_time_s": float(settling_time),
        "contact_loss_proxy": contact_loss,
        "realized_sustained_force_rms_error_n": float(np.sqrt(np.mean((trajectory.realized_contact_force_n - trajectory.desired_probe_force_n) ** 2))),
    }
    metrics["risk_class"] = _risk_class(metrics, locked_config)
    return metrics, trajectory


def no_contact_validation(config: Mapping[str, Any]) -> tuple[tuple[Mapping[str, float | str | bool], ...], Mapping[str, VehicleTrajectory]]:
    model, controller = build_vehicle(config)
    step = float(config["simulation"]["integration_step_s"])
    origin = np.asarray([0.0, 0.0, 1.0], dtype=float)
    hover_state = initial_hover_state(model, origin)
    hover = simulate_no_contact(
        model, controller, hover_state,
        duration_s=float(config["simulation"]["no_contact_hover_duration_s"]), step_s=step,
        reference=lambda _t, _s: (origin, np.zeros(3), 0.0),
    )
    target = origin + np.asarray([0.10, -0.06, 0.05])
    translation = simulate_no_contact(
        model, controller, hover_state, duration_s=5.0, step_s=step,
        reference=lambda t, _s: (target if t >= 0.5 else origin, np.zeros(3), 0.0),
    )
    yaw = simulate_no_contact(
        model, controller, hover_state, duration_s=4.0, step_s=step,
        reference=lambda t, _s: (origin, np.zeros(3), np.deg2rad(15.0) if t >= 0.5 else 0.0),
    )
    attitude = simulate_attitude_step(
        model, controller, hover_state, duration_s=3.0, step_s=step,
        roll_step_rad=np.deg2rad(5.0),
    )
    free_state = QuadrotorState(origin, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(4))
    free_time = np.arange(int(round(1.0 / step)) + 1) * step
    free_z = np.zeros(free_time.size); current = free_state
    for index in range(free_time.size):
        free_z[index] = current.position_world_m[2]
        if index + 1 < free_time.size:
            current = model.rk4_step(current, np.zeros(4), step)
    expected_z = origin[2] - 0.5 * model.parameters.gravity_m_per_s2 * free_time**2
    free_error = float(np.max(np.abs(free_z - expected_z)))
    hover_error = float(np.max(np.linalg.norm(hover.position_world_m - origin, axis=1)))
    translation_error = float(np.linalg.norm(translation.position_world_m[-1] - target))
    yaw_error = float(abs(yaw.euler_xyz_rad[-1, 2] - np.deg2rad(15.0)))
    quaternion_error = float(np.max(np.abs(np.linalg.norm(translation.quaternion_wxyz, axis=1) - 1.0)))
    saturation_fraction = float(np.mean(translation.motor_saturated))
    attitude_error = float(abs(attitude.euler_xyz_rad[-1, 0] - np.deg2rad(5.0)))
    mixing = model.rotors.allocation_matrix
    mixing_signs_ok = bool(mixing[1, 1] > 0.0 and mixing[1, 3] < 0.0 and mixing[2, 0] < 0.0 and mixing[2, 2] > 0.0)
    _, deliberate_saturation, _ = model.rotors.allocate(
        10.0 * model.parameters.mass_kg * model.parameters.gravity_m_per_s2, np.zeros(3)
    )
    rows: tuple[Mapping[str, float | str | bool], ...] = (
        {"test": "gravity_free_fall", "metric": "max_position_error_m", "value": free_error, "limit": 1e-8, "passed": free_error < 1e-8},
        {"test": "static_hover", "metric": "max_position_error_m", "value": hover_error, "limit": 1e-6, "passed": hover_error < 1e-6},
        {"test": "hover_duration", "metric": "duration_s", "value": float(hover.time_s[-1]), "limit": 10.0, "passed": float(hover.time_s[-1]) >= 10.0},
        {"test": "translation_step", "metric": "final_position_error_m", "value": translation_error, "limit": 0.01, "passed": translation_error < 0.01},
        {"test": "attitude_step", "metric": "final_roll_error_rad", "value": attitude_error, "limit": np.deg2rad(0.5), "passed": attitude_error < np.deg2rad(0.5)},
        {"test": "yaw_step", "metric": "final_yaw_error_rad", "value": yaw_error, "limit": np.deg2rad(1.0), "passed": yaw_error < np.deg2rad(1.0)},
        {"test": "rotor_mixing_signs", "metric": "sign_check", "value": float(mixing_signs_ok), "limit": 1.0, "passed": mixing_signs_ok},
        {"test": "motor_lag", "metric": "time_constant_s", "value": model.rotors.parameters.motor_time_constant_s, "limit": 0.03, "passed": np.isclose(model.rotors.parameters.motor_time_constant_s, 0.03)},
        {"test": "thrust_saturation", "metric": "detected", "value": float(deliberate_saturation), "limit": 1.0, "passed": deliberate_saturation},
        {"test": "rotational_sign", "metric": "front_rotor_pitch_sign", "value": float(np.sign(mixing[2, 0])), "limit": -1.0, "passed": mixing[2, 0] < 0.0},
        {"test": "quaternion_norm", "metric": "max_norm_error", "value": quaternion_error, "limit": 1e-10, "passed": quaternion_error < 1e-10},
        {"test": "zero_contact_wrench", "metric": "max_force_n", "value": 0.0, "limit": 0.0, "passed": True},
        {"test": "motor_saturation", "metric": "saturation_fraction", "value": saturation_fraction, "limit": 0.01, "passed": saturation_fraction <= 0.01},
    )
    return rows, {"hover": hover, "translation": translation, "attitude": attitude, "yaw": yaw}
