"""One-dimensional Kelvin-Voigt contact model.

Positive indentation and positive reaction force denote compression.  The
bilateral mode implements the constitutive equation exactly.  The optional
unilateral mode clips tensile reaction and removes contact at negative
indentation; it is provided for later contact-loss studies but is not used to
claim validation of the bilateral energy identity in EXP-0001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class KelvinVoigtParameters:
    """Constitutive parameters for ``F = k*x + c*x_dot``."""

    stiffness_n_per_m: float
    damping_n_s_per_m: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.stiffness_n_per_m, self.damping_n_s_per_m], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Kelvin-Voigt parameters must be finite")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("stiffness_n_per_m must be strictly positive")
        if self.damping_n_s_per_m < 0.0:
            raise ValueError("damping_n_s_per_m must be non-negative")


@dataclass(frozen=True)
class KelvinVoigtResponse:
    """Force decomposition evaluated at one or more trajectory samples."""

    force_n: NDArray[np.float64]
    raw_force_n: NDArray[np.float64]
    elastic_force_n: NDArray[np.float64]
    damping_force_n: NDArray[np.float64]
    contact_active: NDArray[np.bool_]


class KelvinVoigtModel:
    """Evaluate a Kelvin-Voigt element under prescribed indentation."""

    def __init__(
        self,
        parameters: KelvinVoigtParameters,
        contact_mode: Literal["bilateral", "unilateral"] = "bilateral",
    ) -> None:
        if contact_mode not in {"bilateral", "unilateral"}:
            raise ValueError("contact_mode must be 'bilateral' or 'unilateral'")
        self.parameters = parameters
        self.contact_mode = contact_mode

    def response(
        self, indentation_m: ArrayLike, indentation_velocity_m_per_s: ArrayLike
    ) -> KelvinVoigtResponse:
        """Return reaction force and its elastic/viscous decomposition."""

        indentation, velocity = np.broadcast_arrays(
            np.asarray(indentation_m, dtype=float),
            np.asarray(indentation_velocity_m_per_s, dtype=float),
        )
        if not np.all(np.isfinite(indentation)) or not np.all(np.isfinite(velocity)):
            raise ValueError("indentation and velocity samples must be finite")

        elastic = self.parameters.stiffness_n_per_m * indentation
        damping = self.parameters.damping_n_s_per_m * velocity
        raw_force = elastic + damping

        if self.contact_mode == "unilateral":
            contact_eligible = (indentation > 0.0) | (
                (indentation == 0.0) & (velocity > 0.0)
            )
            force = np.where(contact_eligible, np.maximum(raw_force, 0.0), 0.0)
            contact_active = contact_eligible & (force > 0.0)
        else:
            force = raw_force.copy()
            contact_active = np.ones(force.shape, dtype=bool)

        return KelvinVoigtResponse(
            force_n=np.asarray(force, dtype=float),
            raw_force_n=np.asarray(raw_force, dtype=float),
            elastic_force_n=np.asarray(elastic, dtype=float),
            damping_force_n=np.asarray(damping, dtype=float),
            contact_active=np.asarray(contact_active, dtype=bool),
        )

