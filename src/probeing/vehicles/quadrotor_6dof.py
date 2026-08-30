"""Singularity-safe 6-DoF rigid-body quadrotor dynamics.

World is ENU (+z up). Body is FLU (+x forward, +y left, +z up). The
scalar-first Hamilton quaternion q_WB rotates body vectors into world vectors.
Angular velocity and applied torque are expressed in the body frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotor_model import RotorModel


def quat_multiply(left: ArrayLike, right: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_normalize(quaternion: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("quaternion must have nonzero finite norm")
    return q / norm


def quat_from_axis_angle(axis: ArrayLike, angle_rad: float) -> NDArray[np.float64]:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or norm <= np.finfo(float).tiny:
        raise ValueError("rotation axis must be nonzero")
    half = 0.5 * float(angle_rad)
    return np.asarray([np.cos(half), *(np.sin(half) * vector / norm)], dtype=float)


def quat_to_rotation(quaternion: ArrayLike) -> NDArray[np.float64]:
    w, x, y, z = quat_normalize(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_to_euler_xyz(rotation: ArrayLike) -> NDArray[np.float64]:
    """Return roll, pitch, yaw for visualization only."""

    matrix = np.asarray(rotation, dtype=float)
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.asarray([roll, pitch, yaw], dtype=float)


@dataclass(frozen=True)
class QuadrotorParameters:
    mass_kg: float
    inertia_body_kg_m2: NDArray[np.float64]
    center_of_mass_body_m: NDArray[np.float64]
    gravity_m_per_s2: float = 9.80665

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_body_kg_m2, dtype=float)
        center = np.asarray(self.center_of_mass_body_m, dtype=float)
        if self.mass_kg <= 0.0 or self.gravity_m_per_s2 <= 0.0:
            raise ValueError("vehicle mass and gravity must be positive")
        if inertia.shape != (3, 3) or not np.allclose(inertia, inertia.T, atol=1.0e-12):
            raise ValueError("inertia must be a symmetric 3x3 tensor")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia tensor must be positive definite")
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center of mass must be a finite vector")
        object.__setattr__(self, "inertia_body_kg_m2", inertia)
        object.__setattr__(self, "center_of_mass_body_m", center)


@dataclass(frozen=True)
class QuadrotorState:
    position_world_m: NDArray[np.float64]
    velocity_world_m_per_s: NDArray[np.float64]
    quaternion_wxyz: NDArray[np.float64]
    angular_velocity_body_rad_s: NDArray[np.float64]
    rotor_speed_rad_s: NDArray[np.float64]

    def vector(self) -> NDArray[np.float64]:
        return np.concatenate(
            (
                self.position_world_m,
                self.velocity_world_m_per_s,
                self.quaternion_wxyz,
                self.angular_velocity_body_rad_s,
                self.rotor_speed_rad_s,
            )
        ).astype(float)

    @classmethod
    def from_vector(cls, vector: ArrayLike, *, normalize: bool = True) -> "QuadrotorState":
        values = np.asarray(vector, dtype=float)
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise ValueError("quadrotor state must contain 17 finite values")
        quaternion = quat_normalize(values[6:10]) if normalize else values[6:10].copy()
        return cls(
            position_world_m=values[0:3].copy(),
            velocity_world_m_per_s=values[3:6].copy(),
            quaternion_wxyz=quaternion,
            angular_velocity_body_rad_s=values[10:13].copy(),
            rotor_speed_rad_s=values[13:17].copy(),
        )


class QuadrotorModel:
    def __init__(self, parameters: QuadrotorParameters, rotors: RotorModel) -> None:
        self.parameters = parameters
        self.rotors = rotors
        self._inertia_inverse = np.linalg.inv(parameters.inertia_body_kg_m2)

    def hover_speed_rad_s(self) -> float:
        coefficient = self.rotors.parameters.thrust_coefficient_n_per_rad2
        return float(np.sqrt(self.parameters.mass_kg * self.parameters.gravity_m_per_s2 / (4.0 * coefficient)))

    def derivative(
        self,
        vector: ArrayLike,
        commanded_rotor_speed_rad_s: ArrayLike,
        *,
        external_force_world_n: ArrayLike | None = None,
        external_torque_body_nm: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        state = QuadrotorState.from_vector(vector, normalize=False)
        q = state.quaternion_wxyz
        q_norm = float(np.linalg.norm(q))
        if q_norm <= np.finfo(float).tiny:
            raise ValueError("quaternion norm vanished")
        rotation = quat_to_rotation(q)
        _, rotor_force_body, rotor_torque_body = self.rotors.wrench(state.rotor_speed_rad_s)
        force_external = np.zeros(3) if external_force_world_n is None else np.asarray(external_force_world_n, dtype=float)
        torque_external = np.zeros(3) if external_torque_body_nm is None else np.asarray(external_torque_body_nm, dtype=float)
        gravity = np.asarray([0.0, 0.0, -self.parameters.gravity_m_per_s2], dtype=float)
        acceleration = gravity + (rotation @ rotor_force_body + force_external) / self.parameters.mass_kg
        omega = state.angular_velocity_body_rad_s
        inertia_omega = self.parameters.inertia_body_kg_m2 @ omega
        angular_acceleration = self._inertia_inverse @ (
            rotor_torque_body + torque_external - np.cross(omega, inertia_omega)
        )
        quaternion_derivative = 0.5 * quat_multiply(q, np.asarray([0.0, *omega]))
        rotor_derivative = self.rotors.speed_derivative(
            state.rotor_speed_rad_s, commanded_rotor_speed_rad_s
        )
        return np.concatenate(
            (
                state.velocity_world_m_per_s,
                acceleration,
                quaternion_derivative,
                angular_acceleration,
                rotor_derivative,
            )
        )

    def rk4_step(
        self,
        state: QuadrotorState,
        commanded_rotor_speed_rad_s: ArrayLike,
        step_s: float,
        *,
        external_force_world_n: ArrayLike | None = None,
        external_torque_body_nm: ArrayLike | None = None,
    ) -> QuadrotorState:
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive")
        initial = state.vector()
        arguments = {
            "external_force_world_n": external_force_world_n,
            "external_torque_body_nm": external_torque_body_nm,
        }
        k1 = self.derivative(initial, commanded_rotor_speed_rad_s, **arguments)
        k2 = self.derivative(initial + 0.5 * step_s * k1, commanded_rotor_speed_rad_s, **arguments)
        k3 = self.derivative(initial + 0.5 * step_s * k2, commanded_rotor_speed_rad_s, **arguments)
        k4 = self.derivative(initial + step_s * k3, commanded_rotor_speed_rad_s, **arguments)
        result = initial + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        result[13:17] = self.rotors.clip_speed(result[13:17])
        return QuadrotorState.from_vector(result, normalize=True)
