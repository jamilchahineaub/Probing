from __future__ import annotations

import numpy as np

from probeing.controllers import CascadedController, CascadedControllerGains
from probeing.vehicles import (
    QuadrotorModel,
    QuadrotorParameters,
    QuadrotorState,
    RotorGeometry,
    RotorModel,
    RotorParameters,
    quat_from_axis_angle,
    quat_to_rotation,
)


def vehicle() -> QuadrotorModel:
    rotors = RotorModel(RotorParameters(), RotorGeometry.plus_configuration(0.23))
    parameters = QuadrotorParameters(
        mass_kg=1.50,
        inertia_body_kg_m2=np.diag([0.029, 0.029, 0.055]),
        center_of_mass_body_m=np.zeros(3),
    )
    return QuadrotorModel(parameters, rotors)


def state(model: QuadrotorModel, *, rotor_speed: float = 0.0) -> QuadrotorState:
    return QuadrotorState(
        position_world_m=np.zeros(3),
        velocity_world_m_per_s=np.zeros(3),
        quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body_rad_s=np.zeros(3),
        rotor_speed_rad_s=np.full(4, rotor_speed),
    )


def test_gravity_sign_and_free_fall() -> None:
    model = vehicle()
    derivative = model.derivative(state(model).vector(), np.zeros(4))
    assert np.allclose(derivative[3:6], [0.0, 0.0, -model.parameters.gravity_m_per_s2])


def test_static_hover_equilibrium() -> None:
    model = vehicle()
    hover = model.hover_speed_rad_s()
    derivative = model.derivative(state(model, rotor_speed=hover).vector(), np.full(4, hover))
    assert np.linalg.norm(derivative[3:6]) < 1.0e-12
    assert np.linalg.norm(derivative[10:13]) < 1.0e-12
    assert np.linalg.norm(derivative[13:17]) < 1.0e-12


def test_individual_rotor_force_and_torque_signs() -> None:
    model = vehicle()
    speed = np.asarray([500.0, 0.0, 0.0, 0.0])
    thrust, force, torque = model.rotors.wrench(speed)
    assert thrust[0] > 0.0 and np.allclose(thrust[1:], 0.0)
    assert force[2] > 0.0
    assert torque[1] < 0.0  # front rotor pitches nose down
    assert torque[2] > 0.0  # rotor 1 documented positive yaw reaction


def test_mixer_generates_requested_small_wrench() -> None:
    model = vehicle()
    requested_thrust = model.parameters.mass_kg * model.parameters.gravity_m_per_s2
    requested_torque = np.asarray([0.01, -0.012, 0.004])
    speed, saturated, reserve = model.rotors.allocate(requested_thrust, requested_torque)
    _, force, torque = model.rotors.wrench(speed)
    assert not saturated
    assert reserve > 0.0
    assert np.isclose(force[2], requested_thrust)
    assert np.allclose(torque, requested_torque)


def test_motor_lag_is_first_order_and_bounded() -> None:
    model = vehicle()
    initial = np.full(4, 100.0)
    command = np.full(4, 500.0)
    derivative = model.rotors.speed_derivative(initial, command)
    expected = (command - initial) / model.rotors.parameters.motor_time_constant_s
    assert np.allclose(derivative, expected)
    clipped = model.rotors.clip_speed(np.full(4, 2_000.0))
    assert np.all(clipped == model.rotors.parameters.maximum_speed_rad_s)


def test_rotational_dynamics_sign() -> None:
    model = vehicle()
    base = state(model, rotor_speed=model.hover_speed_rad_s())
    derivative = model.derivative(
        base.vector(),
        base.rotor_speed_rad_s,
        external_torque_body_nm=np.asarray([0.01, 0.0, 0.0]),
    )
    assert derivative[10] > 0.0
    assert np.isclose(derivative[10], 0.01 / model.parameters.inertia_body_kg_m2[0, 0])


def test_quaternion_norm_preserved_over_rotation() -> None:
    model = vehicle()
    current = state(model, rotor_speed=model.hover_speed_rad_s())
    current = QuadrotorState(
        current.position_world_m,
        current.velocity_world_m_per_s,
        current.quaternion_wxyz,
        np.asarray([0.2, -0.1, 0.15]),
        current.rotor_speed_rad_s,
    )
    for _ in range(2_000):
        current = model.rk4_step(current, current.rotor_speed_rad_s, 0.001)
    assert abs(np.linalg.norm(current.quaternion_wxyz) - 1.0) < 1.0e-12
    rotation = quat_to_rotation(current.quaternion_wxyz)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)


def test_cascaded_controller_hover_and_attitude_correction() -> None:
    model = vehicle()
    controller = CascadedController(
        model,
        CascadedControllerGains(
            position_p=np.asarray([4.0, 4.0, 6.0]),
            velocity_d=np.asarray([3.0, 3.0, 4.5]),
            attitude_p=np.asarray([0.35, 0.35, 0.20]),
            body_rate_d=np.asarray([0.08, 0.08, 0.06]),
        ),
    )
    hover = state(model, rotor_speed=model.hover_speed_rad_s())
    output = controller.command(hover, np.zeros(3), np.zeros(3))
    assert not output.saturated
    assert np.allclose(output.commanded_rotor_speed_rad_s, model.hover_speed_rad_s())
    tilted = QuadrotorState(
        hover.position_world_m,
        hover.velocity_world_m_per_s,
        quat_from_axis_angle([1.0, 0.0, 0.0], np.deg2rad(5.0)),
        hover.angular_velocity_body_rad_s,
        hover.rotor_speed_rad_s,
    )
    correcting = controller.command(tilted, np.zeros(3), np.zeros(3))
    assert correcting.desired_torque_body_nm[0] < 0.0
