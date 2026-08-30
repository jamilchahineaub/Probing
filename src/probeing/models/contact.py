"""Bilateral and command-gated unilateral interaction wrappers for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mass_spring_damper import InteractionSimulation, MassSpringDamperModel


@dataclass(frozen=True)
class ContactInteractionSimulation:
    """Target response plus commanded/contact-force and contact-state histories."""

    response: InteractionSimulation
    commanded_force_n: NDArray[np.float64]
    contact_force_n: NDArray[np.float64]
    contact_active: NDArray[np.bool_]
    contact_mode: Literal["bilateral", "unilateral"]
    contact_loss_count: int


def simulate_contact_interaction(
    model: MassSpringDamperModel,
    time_s: ArrayLike,
    commanded_force_n: ArrayLike,
    *,
    contact_mode: Literal["bilateral", "unilateral"],
    initial_displacement_m: float = 0.0,
    initial_velocity_m_per_s: float = 0.0,
) -> ContactInteractionSimulation:
    """Simulate signed bilateral or no-tension unilateral forcing.

    The unilateral Stage 1 abstraction has no probe-body dynamics. A positive
    commanded force denotes compression. Non-positive commands denote
    separation, so the contact force is zero and the target follows its free
    response until compressive forcing resumes.
    """

    time = np.asarray(time_s, dtype=float)
    command = np.asarray(commanded_force_n, dtype=float)
    if time.ndim != 1 or command.ndim != 1 or time.shape != command.shape:
        raise ValueError("time and commanded force must be one-dimensional and equal length")
    if not np.all(np.isfinite(command)):
        raise ValueError("commanded force must be finite")
    if contact_mode == "bilateral":
        contact_force = command.copy()
        active = np.ones(command.shape, dtype=bool)
    elif contact_mode == "unilateral":
        contact_force = np.maximum(command, 0.0)
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(command)))
        )
        contact_force[contact_force <= tolerance] = 0.0
        active = contact_force > 0.0
    else:
        raise ValueError("contact_mode must be 'bilateral' or 'unilateral'")

    response = model.simulate(
        time,
        contact_force,
        initial_displacement_m=initial_displacement_m,
        initial_velocity_m_per_s=initial_velocity_m_per_s,
    )
    loss_count = int(np.count_nonzero(active[:-1] & ~active[1:]))
    return ContactInteractionSimulation(
        response=response,
        commanded_force_n=command,
        contact_force_n=contact_force,
        contact_active=active,
        contact_mode=contact_mode,
        contact_loss_count=loss_count,
    )
