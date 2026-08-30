"""Rigid probe kinematics and contact-wrench transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.vehicles import QuadrotorState, quat_to_rotation


@dataclass(frozen=True)
class ProbeGeometry:
    """A massless rigid probe fixed at ``tip_offset_body_m`` from vehicle CoM."""

    tip_offset_body_m: NDArray[np.float64]
    probe_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        offset = np.asarray(self.tip_offset_body_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("probe offset must be a finite body-frame vector")
        if self.probe_mass_kg < 0.0 or not np.isfinite(self.probe_mass_kg):
            raise ValueError("probe mass cannot be negative")
        object.__setattr__(self, "tip_offset_body_m", offset)

    def tip_kinematics(
        self, state: QuadrotorState
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rotation = quat_to_rotation(state.quaternion_wxyz)
        offset_world = rotation @ self.tip_offset_body_m
        position = state.position_world_m + offset_world
        velocity = state.velocity_world_m_per_s + rotation @ np.cross(
            state.angular_velocity_body_rad_s, self.tip_offset_body_m
        )
        return position, velocity

    def torque_body_from_world_force(
        self, state: QuadrotorState, force_world_n: ArrayLike
    ) -> NDArray[np.float64]:
        rotation = quat_to_rotation(state.quaternion_wxyz)
        force_body = rotation.T @ np.asarray(force_world_n, dtype=float)
        return np.cross(self.tip_offset_body_m, force_body)
