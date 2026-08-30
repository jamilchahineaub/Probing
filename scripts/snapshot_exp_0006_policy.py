#!/usr/bin/env python3
"""Print the exact fitted EXP-0006 0.5 s primary policy as canonical JSON.

This utility reads only the frozen EXP-0006 configuration and its original
training/calibration seed partitions. It does not read or generate EXP-0007
targets. The printed bundle is checked into EXP-0007 and hash-locked before
the independent replication is executed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from probeing.experiments.decision_sufficiency import generate_target_cases
from probeing.experiments.passive_ringdown import (
    FEATURE_SETS,
    _extract_partition_features,
    _fit_models,
    _simulate_outcomes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXP6_CONFIG = REPOSITORY_ROOT / "configs/experiments/exp_0006_passive_ringdown.yaml"
EXP6_SOURCE = REPOSITORY_ROOT / "src/probeing/experiments/passive_ringdown.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle() -> Mapping[str, Any]:
    config = yaml.safe_load(EXP6_CONFIG.read_text(encoding="utf-8"))
    policy_config = copy.deepcopy(config)
    policy_config["passive_observation"]["windows_s"] = [0.5]
    partitions = policy_config["seed_partitions"]
    count = int(partitions["cases_per_seed"])
    training = generate_target_cases(
        partitions["training"], count, policy_config["target_population"], partition="training"
    )
    calibration = generate_target_cases(
        partitions["calibration"], count, policy_config["target_population"], partition="calibration"
    )
    state = [0]
    total = 3 * (len(training) + len(calibration))
    training_features, _, _ = _extract_partition_features(
        training,
        policy_config,
        progress_callback=None,
        progress_state=state,
        progress_total=total,
    )
    training_outcomes = _simulate_outcomes(training, policy_config)
    calibration_features, _, _ = _extract_partition_features(
        calibration,
        policy_config,
        progress_callback=None,
        progress_state=state,
        progress_total=total,
    )
    calibration_outcomes = _simulate_outcomes(calibration, policy_config)
    fitted = _fit_models(
        training_features,
        training_outcomes,
        calibration_features,
        calibration_outcomes,
        policy_config,
    )[(0.5, "chirp_ringdown")]
    first_model = next(iter(fitted.regressors.values()))
    if any(
        model.mean.tolist() != first_model.mean.tolist()
        or model.scale.tolist() != first_model.scale.tolist()
        for model in fitted.regressors.values()
    ):
        raise RuntimeError("EXP-0006 outcome models unexpectedly use different normalization")
    regressors = {
        outcome: {
            "coefficients": model.coefficients.tolist(),
            "upper_residual": float(fitted.upper_residuals[outcome]),
        }
        for outcome, model in fitted.regressors.items()
    }
    return {
        "schema_version": 1,
        "policy_id": "EXP-0006-primary-0p5s-chirp-ringdown",
        "source_experiment": "EXP-0006",
        "source_run_id": "EXP-0006_20260829T194621.991365Z_s8101_2966d3d2",
        "source_config_sha256": _sha256(EXP6_CONFIG),
        "source_policy_module_sha256": _sha256(EXP6_SOURCE),
        "feature_set": "chirp_ringdown",
        "feature_names": list(FEATURE_SETS["chirp_ringdown"]),
        "observation_duration_s": 0.5,
        "locked_config_path": "configs/experiments/exp_0006_passive_ringdown.yaml",
        "normalization": {
            "mean": first_model.mean.tolist(),
            "scale": first_model.scale.tolist(),
        },
        "regressors": regressors,
        "fit_provenance": {
            "training_seeds": list(partitions["training"]),
            "calibration_seeds": list(partitions["calibration"]),
            "cases_per_seed": count,
            "note": "Snapshot provenance only; EXP-0007 performs no fitting or calibration.",
        },
    }


def main() -> int:
    print(json.dumps(build_bundle(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
