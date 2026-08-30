"""Transparent cascaded position and geometric attitude controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.vehicles import QuadrotorModel, QuadrotorState, quat_to_rotation


def _vee(skew: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


@dataclass(frozen=True)
class CascadedControllerGains:
    position_p: NDArray[np.float64]
    velocity_d: NDArray[np.float64]
    attitude_p: NDArray[np.float64]
    body_rate_d: NDArray[np.float64]
    maximum_acceleration_m_per_s2: float = 8.0

    def __post_init__(self) -> None:
        for name in ("position_p", "velocity_d", "attitude_p", "body_rate_d"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three positive gains")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ControllerOutput:
    commanded_rotor_speed_rad_s: NDArray[np.float64]
    desired_force_world_n: NDArray[np.float64]
    desired_rotation_body_to_world: NDArray[np.float64]
    desired_torque_body_nm: NDArray[np.float64]
    saturated: bool
    actuator_reserve: float


class CascadedController:
    def __init__(self, vehicle: QuadrotorModel, gains: CascadedControllerGains) -> None:
        self.vehicle = vehicle
        self.gains = gains

    @staticmethod
    def desired_rotation(force_world_n: ArrayLike, yaw_rad: float) -> NDArray[np.float64]:
        force = np.asarray(force_world_n, dtype=float)
        magnitude = float(np.linalg.norm(force))
        if magnitude <= np.finfo(float).tiny:
            raise ValueError("desired force cannot be zero")
        z_body = force / magnitude
        x_heading = np.asarray([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=float)
        y_body = np.cross(z_body, x_heading)
        if np.linalg.norm(y_body) < 1.0e-8:
            x_heading = np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=float)
            y_body = np.cross(z_body, x_heading)
        y_body /= np.linalg.norm(y_body)
        x_body = np.cross(y_body, z_body)
        return np.column_stack((x_body, y_body, z_body))

    def command(
        self,
        state: QuadrotorState,
        desired_position_world_m: ArrayLike,
        desired_velocity_world_m_per_s: ArrayLike,
        *,
        desired_yaw_rad: float = 0.0,
        feedforward_force_world_n: ArrayLike | None = None,
        measured_external_torque_body_nm: ArrayLike | None = None,
    ) -> ControllerOutput:
        position_error = np.asarray(desired_position_world_m, dtype=float) - state.position_world_m
        velocity_error = np.asarray(desired_velocity_world_m_per_s, dtype=float) - state.velocity_world_m_per_s
        acceleration = self.gains.position_p * position_error + self.gains.velocity_d * velocity_error
        norm = float(np.linalg.norm(acceleration))
        if norm > self.gains.maximum_acceleration_m_per_s2:
            acceleration *= self.gains.maximum_acceleration_m_per_s2 / norm
        gravity_compensation = np.asarray(
            [0.0, 0.0, self.vehicle.parameters.gravity_m_per_s2], dtype=float
        )
        force = self.vehicle.parameters.mass_kg * (gravity_compensation + acceleration)
        if feedforward_force_world_n is not None:
            force += np.asarray(feedforward_force_world_n, dtype=float)
        desired_rotation = self.desired_rotation(force, desired_yaw_rad)
        return self.attitude_command(
            state, force, desired_rotation,
            measured_external_torque_body_nm=measured_external_torque_body_nm,
        )

    def attitude_command(
        self,
        state: QuadrotorState,
        desired_force_world_n: ArrayLike,
        desired_rotation_body_to_world: ArrayLike,
        *, measured_external_torque_body_nm: ArrayLike | None = None,
    ) -> ControllerOutput:
        """Run the inner attitude/body-rate loop for explicit validation cases."""

        force = np.asarray(desired_force_world_n, dtype=float)
        desired_rotation = np.asarray(desired_rotation_body_to_world, dtype=float)
        if force.shape != (3,) or desired_rotation.shape != (3, 3):
            raise ValueError("desired force and rotation have invalid shape")
        rotation = quat_to_rotation(state.quaternion_wxyz)
        attitude_skew = 0.5 * (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = _vee(attitude_skew)
        torque = (
            -self.gains.attitude_p * attitude_error
            - self.gains.body_rate_d * state.angular_velocity_body_rad_s
            + np.cross(
                state.angular_velocity_body_rad_s,
                self.vehicle.parameters.inertia_body_kg_m2 @ state.angular_velocity_body_rad_s,
            )
        )
        if measured_external_torque_body_nm is not None:
            torque -= np.asarray(measured_external_torque_body_nm, dtype=float)
        total_thrust = max(float(force @ rotation[:, 2]), 0.0)
        speed, saturated, reserve = self.vehicle.rotors.allocate(total_thrust, torque)
        return ControllerOutput(
            commanded_rotor_speed_rad_s=speed,
            desired_force_world_n=force,
            desired_rotation_body_to_world=desired_rotation,
            desired_torque_body_nm=torque,
            saturated=saturated,
            actuator_reserve=reserve,
        )
