from __future__ import annotations

import numpy as np

from probeing.contact import (
    ProbeGeometry,
    Target1D,
    TargetParameters,
    TargetState,
    UnilateralPenaltyContact,
)
from probeing.vehicles import QuadrotorState, quat_from_axis_angle


def vehicle_state(x: float, velocity_x: float = 0.0) -> QuadrotorState:
    return QuadrotorState(
        position_world_m=np.asarray([x, 0.0, 1.0]),
        velocity_world_m_per_s=np.asarray([velocity_x, 0.0, 0.0]),
        quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body_rad_s=np.zeros(3),
        rotor_speed_rad_s=np.zeros(4),
    )


def interface() -> UnilateralPenaltyContact:
    return UnilateralPenaltyContact(
        surface_origin_world_m=np.asarray([0.0, 0.0, 1.0]),
        normal_world=np.asarray([1.0, 0.0, 0.0]),
        interface_stiffness_n_per_m=5_000.0,
        interface_damping_n_s_per_m=10.0,
    )


def test_zero_force_before_contact_and_after_separation() -> None:
    probe = ProbeGeometry(np.asarray([0.30, 0.0, 0.0]))
    no_contact = interface().evaluate(vehicle_state(-0.31), probe, TargetState())
    separation = interface().evaluate(vehicle_state(-0.301, -0.1), probe, TargetState())
    assert not no_contact.active and no_contact.contact_force_n == 0.0
    assert not separation.active and separation.contact_force_n == 0.0


def test_contact_force_direction_and_action_reaction() -> None:
    probe = ProbeGeometry(np.asarray([0.30, 0.0, 0.0]))
    result = interface().evaluate(vehicle_state(-0.299), probe, TargetState())
    assert result.active and result.contact_force_n > 0.0
    assert result.target_force_world_n[0] > 0.0
    assert result.vehicle_force_world_n[0] < 0.0
    assert np.allclose(result.target_force_world_n, -result.vehicle_force_world_n)


def test_offset_contact_torque_is_exact_cross_product() -> None:
    probe = ProbeGeometry(np.asarray([0.30, 0.0, -0.08]))
    state = vehicle_state(-0.299)
    result = interface().evaluate(state, probe, TargetState(displacement_m=-0.08))
    expected = np.cross(probe.tip_offset_body_m, result.vehicle_force_world_n)
    assert np.allclose(result.vehicle_torque_body_nm, expected)
    assert result.vehicle_torque_body_nm[1] > 0.0


def test_contact_torque_rotates_force_to_body_frame() -> None:
    probe = ProbeGeometry(np.asarray([0.30, 0.0, -0.08]))
    state = vehicle_state(-0.299)
    state = QuadrotorState(
        state.position_world_m,
        state.velocity_world_m_per_s,
        quat_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0),
        state.angular_velocity_body_rad_s,
        state.rotor_speed_rad_s,
    )
    force_world = np.asarray([-1.0, 0.0, 0.0])
    torque = probe.torque_body_from_world_force(state, force_world)
    expected_body_force = np.asarray([0.0, 1.0, 0.0])
    assert np.allclose(torque, np.cross(probe.tip_offset_body_m, expected_body_force), atol=1e-12)


def test_target_equilibrium_and_dissipation_sign() -> None:
    target = Target1D(TargetParameters(200.0, 5.0, 1.0))
    equilibrium = TargetState(displacement_m=0.01, velocity_m_per_s=0.0)
    assert abs(target.acceleration(equilibrium, 2.0)) < 1.0e-12
    moving = TargetState(displacement_m=0.01, velocity_m_per_s=-0.2)
    assert target.dissipation_power_w(moving) > 0.0
    assert target.mechanical_energy_j(moving) > 0.0


def test_interface_does_not_create_tensile_force() -> None:
    probe = ProbeGeometry(np.asarray([0.30, 0.0, 0.0]))
    result = interface().evaluate(
        vehicle_state(-0.299, -10.0), probe, TargetState(velocity_m_per_s=0.0)
    )
    assert result.contact_force_n == 0.0
    assert not result.active
