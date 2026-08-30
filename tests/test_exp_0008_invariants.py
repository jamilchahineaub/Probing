from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from probeing.experiments.coupled_uav_contact import locked_probe_force


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/experiments/exp_0008_coupled_uav_contact.yaml").read_text())


def _fingerprint(experiment_id: str) -> str:
    selected = []
    underscored = experiment_id.lower().replace("-", "_")
    for root_name in ("configs/experiments", "runs", "results"):
        for path in (ROOT / root_name).rglob("*"):
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                if underscored in relative.lower() or f"/{experiment_id}_" in f"/{relative}":
                    selected.append(path)
    aggregate = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.relative_to(ROOT).as_posix()):
        aggregate.update(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n".encode())
    return aggregate.hexdigest()


def test_locked_chirp_and_passive_duration_are_unchanged() -> None:
    probe = CONFIG["locked_probe"]
    assert probe["start_frequency_hz"] == 0.5
    assert probe["end_frequency_hz"] == 5.0
    assert probe["amplitude_n"] == 0.5
    assert probe["duration_s"] == 3.0
    assert probe["observation_duration_s"] == 0.5
    time = np.arange(0.0, 3.501, 0.001)
    force = locked_probe_force(time, CONFIG)
    assert np.max(force) <= 0.5 + 1e-12
    assert np.all(force[time > 3.0] == 0.0)
    assert np.all(force >= 0.0)


def test_locked_policy_bundle_hash_is_unchanged() -> None:
    path = ROOT / CONFIG["frozen_baseline"]["locked_policy_bundle"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CONFIG["frozen_baseline"]["locked_policy_sha256"]


def test_stage1_artifact_fingerprints_are_unchanged() -> None:
    expected = CONFIG["frozen_baseline"]["stage1_fingerprints"]
    actual = {
        f"exp_{index:04d}": _fingerprint(f"EXP-{index:04d}")
        for index in range(1, 8)
    }
    assert actual == expected
