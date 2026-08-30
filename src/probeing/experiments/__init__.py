"""Configured research experiment implementations."""

from .interaction_identification import (
    MilestoneAExperimentResult,
    run_interaction_identification,
)
from .practical_identifiability import (
    PracticalIdentifiabilityResult,
    analyze_exp0001_ramp_failure,
    run_practical_identifiability,
)
from .causal_eiv_identifiability import (
    CausalEIVIdentifiabilityResult,
    run_causal_eiv_identifiability,
)
from .sequential_identification import (
    SequentialIdentificationResult,
    run_sequential_identification,
)
from .decision_sufficiency import (
    DecisionSufficiencyResult,
    run_decision_sufficiency,
)
from .passive_ringdown import PassiveRingdownResult, run_passive_ringdown
from .locked_policy_replication import LockedReplicationResult, run_locked_policy_replication
from .coupled_uav_contact import build_vehicle, no_contact_validation, simulate_no_contact

__all__ = [
    "MilestoneAExperimentResult",
    "PracticalIdentifiabilityResult",
    "analyze_exp0001_ramp_failure",
    "run_interaction_identification",
    "run_practical_identifiability",
    "CausalEIVIdentifiabilityResult",
    "run_causal_eiv_identifiability",
    "SequentialIdentificationResult",
    "run_sequential_identification",
    "DecisionSufficiencyResult",
    "run_decision_sufficiency",
    "PassiveRingdownResult",
    "run_passive_ringdown",
    "build_vehicle",
    "no_contact_validation",
    "simulate_no_contact",
]
