"""Unilateral normal penalty interface between a rigid probe and moving target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probeing.vehicles import QuadrotorState

from .probe_geometry import ProbeGeometry
from .target_1d import TargetState


@dataclass(frozen=True)
class ContactResult:
    active: bool
    penetration_m: float
    closing_speed_m_per_s: float
    contact_force_n: float
    target_force_world_n: NDArray[np.float64]
    vehicle_force_world_n: NDArray[np.float64]
    vehicle_torque_body_nm: NDArray[np.float64]
    interface_elastic_energy_j: float


class UnilateralPenaltyContact:
    """No-tension Kelvin-Voigt interface used only to enforce physical contact.

    The unknown structure remains the separate Stage 1 mass-spring-damper
    target. The interface stiffness is deliberately much larger than the
    target range and is not an identified target parameter.
    """

    def __init__(
        self,
        *,
        surface_origin_world_m: ArrayLike,
        normal_world: ArrayLike,
        interface_stiffness_n_per_m: float,
        interface_damping_n_s_per_m: float,
    ) -> None:
        origin = np.asarray(surface_origin_world_m, dtype=float)
        normal = np.asarray(normal_world, dtype=float)
        norm = float(np.linalg.norm(normal))
        if origin.shape != (3,) or normal.shape != (3,) or norm <= np.finfo(float).tiny:
            raise ValueError("surface origin and nonzero normal must be 3-D")
        if interface_stiffness_n_per_m <= 0.0 or interface_damping_n_s_per_m < 0.0:
            raise ValueError("contact interface coefficients are invalid")
        self.surface_origin_world_m = origin
        self.normal_world = normal / norm
        self.interface_stiffness_n_per_m = float(interface_stiffness_n_per_m)
        self.interface_damping_n_s_per_m = float(interface_damping_n_s_per_m)

    def evaluate(
        self,
        vehicle_state: QuadrotorState,
        probe: ProbeGeometry,
        target_state: TargetState,
    ) -> ContactResult:
        tip_position, tip_velocity = probe.tip_kinematics(vehicle_state)
        tip_coordinate = float(
            (tip_position - self.surface_origin_world_m) @ self.normal_world
        )
        tip_normal_velocity = float(tip_velocity @ self.normal_world)
        penetration = tip_coordinate - target_state.displacement_m
        closing_speed = tip_normal_velocity - target_state.velocity_m_per_s
        raw_force = (
            self.interface_stiffness_n_per_m * penetration
            + self.interface_damping_n_s_per_m * closing_speed
        )
        force = float(max(raw_force, 0.0)) if penetration > 0.0 else 0.0
        active = force > 0.0
        target_force = force * self.normal_world
        vehicle_force = -target_force
        torque_body = probe.torque_body_from_world_force(vehicle_state, vehicle_force)
        energy = 0.5 * self.interface_stiffness_n_per_m * max(penetration, 0.0) ** 2
        return ContactResult(
            active=active,
            penetration_m=float(max(penetration, 0.0)),
            closing_speed_m_per_s=closing_speed,
            contact_force_n=force,
            target_force_world_n=target_force,
            vehicle_force_world_n=vehicle_force,
            vehicle_torque_body_nm=torque_body,
            interface_elastic_energy_j=float(energy),
        )
