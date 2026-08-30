"""Four-rotor thrust, reaction-torque, saturation, and motor-lag model.

The body frame is right-handed FLU: +x forward, +y left, +z up. Every rotor
produces force along +body-z. ``spin_direction`` is +1 for a positive body-z
reaction torque and -1 for a negative one. Rotor speed is in rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RotorParameters:
    thrust_coefficient_n_per_rad2: float = 1.90e-5
    torque_coefficient_nm_per_rad2: float = 2.60e-7
    motor_time_constant_s: float = 0.030
    minimum_speed_rad_s: float = 0.0
    maximum_speed_rad_s: float = 900.0

    def __post_init__(self) -> None:
        values = (
            self.thrust_coefficient_n_per_rad2,
            self.torque_coefficient_nm_per_rad2,
            self.motor_time_constant_s,
            self.maximum_speed_rad_s,
        )
        if not all(np.isfinite(values)) or any(value <= 0.0 for value in values):
            raise ValueError("rotor coefficients, lag, and maximum speed must be positive")
        if self.minimum_speed_rad_s < 0.0 or self.minimum_speed_rad_s >= self.maximum_speed_rad_s:
            raise ValueError("rotor speed limits are invalid")


@dataclass(frozen=True)
class RotorGeometry:
    positions_body_m: NDArray[np.float64]
    spin_directions: NDArray[np.float64]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_body_m, dtype=float)
        directions = np.asarray(self.spin_directions, dtype=float)
        if positions.shape != (4, 3) or directions.shape != (4,):
            raise ValueError("a quadrotor requires four 3-D positions and four spin directions")
        if not np.all(np.isfinite(positions)) or not np.all(np.isin(directions, (-1.0, 1.0))):
            raise ValueError("rotor geometry must be finite with +/-1 spin directions")
        object.__setattr__(self, "positions_body_m", positions)
        object.__setattr__(self, "spin_directions", directions)

    @classmethod
    def plus_configuration(cls, arm_length_m: float = 0.23) -> "RotorGeometry":
        """Front, left, rear, right rotors with alternating yaw reaction signs."""

        if not np.isfinite(arm_length_m) or arm_length_m <= 0.0:
            raise ValueError("arm_length_m must be positive")
        return cls(
            positions_body_m=np.asarray(
                [
                    [arm_length_m, 0.0, 0.0],
                    [0.0, arm_length_m, 0.0],
                    [-arm_length_m, 0.0, 0.0],
                    [0.0, -arm_length_m, 0.0],
                ],
                dtype=float,
            ),
            spin_directions=np.asarray([1.0, -1.0, 1.0, -1.0]),
        )


class RotorModel:
    """Independent rotors and exact static allocation for the documented geometry."""

    def __init__(self, parameters: RotorParameters, geometry: RotorGeometry) -> None:
        self.parameters = parameters
        self.geometry = geometry
        self._allocation = self._build_allocation()
        if np.linalg.matrix_rank(self._allocation) != 4:
            raise ValueError("rotor allocation matrix is singular")

    def _build_allocation(self) -> NDArray[np.float64]:
        p = self.parameters
        columns: list[NDArray[np.float64]] = []
        for position, direction in zip(
            self.geometry.positions_body_m, self.geometry.spin_directions
        ):
            thrust_per_speed2 = np.asarray(
                [0.0, 0.0, p.thrust_coefficient_n_per_rad2], dtype=float
            )
            moment = np.cross(position, thrust_per_speed2)
            moment[2] += direction * p.torque_coefficient_nm_per_rad2
            columns.append(np.asarray([p.thrust_coefficient_n_per_rad2, *moment]))
        return np.column_stack(columns)

    @property
    def allocation_matrix(self) -> NDArray[np.float64]:
        return self._allocation.copy()

    def clip_speed(self, speed_rad_s: ArrayLike) -> NDArray[np.float64]:
        speed = np.asarray(speed_rad_s, dtype=float)
        if speed.shape != (4,) or not np.all(np.isfinite(speed)):
            raise ValueError("rotor speed must contain four finite values")
        return np.clip(
            speed,
            self.parameters.minimum_speed_rad_s,
            self.parameters.maximum_speed_rad_s,
        )

    def speed_derivative(
        self, speed_rad_s: ArrayLike, commanded_speed_rad_s: ArrayLike
    ) -> NDArray[np.float64]:
        speed = self.clip_speed(speed_rad_s)
        command = self.clip_speed(commanded_speed_rad_s)
        derivative = (command - speed) / self.parameters.motor_time_constant_s
        at_min = (speed <= self.parameters.minimum_speed_rad_s) & (derivative < 0.0)
        at_max = (speed >= self.parameters.maximum_speed_rad_s) & (derivative > 0.0)
        derivative[at_min | at_max] = 0.0
        return derivative

    def wrench(
        self, speed_rad_s: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        speed = self.clip_speed(speed_rad_s)
        squared = speed**2
        thrust = self.parameters.thrust_coefficient_n_per_rad2 * squared
        wrench = self._allocation @ squared
        return (
            thrust,
            np.asarray([0.0, 0.0, wrench[0]], dtype=float),
            np.asarray(wrench[1:4], dtype=float),
        )

    def allocate(
        self, total_thrust_n: float, torque_body_nm: ArrayLike
    ) -> tuple[NDArray[np.float64], bool, float]:
        desired = np.asarray([total_thrust_n, *np.asarray(torque_body_nm, dtype=float)], dtype=float)
        if desired.shape != (4,) or not np.all(np.isfinite(desired)):
            raise ValueError("desired wrench must be finite")
        squared = np.linalg.solve(self._allocation, desired)
        lower = self.parameters.minimum_speed_rad_s**2
        upper = self.parameters.maximum_speed_rad_s**2
        clipped = np.clip(squared, lower, upper)
        saturated = bool(np.any(np.abs(clipped - squared) > 1.0e-10))
        speed = np.sqrt(clipped)
        reserve = float(
            np.min((upper - clipped) / max(upper - lower, np.finfo(float).eps))
        )
        return speed, saturated, reserve
