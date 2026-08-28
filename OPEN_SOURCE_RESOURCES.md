# Open-source resource ledger

Audit date: 2026-08-28. Activity dates below are the newest commits or pushes
observed during this audit, not release pins. Every adopted dependency must be
pinned and re-audited before use.

| Resource | URL | License | Activity observed | Relevant areas | Reuse candidate | Risks/dependencies | Used now? |
|---|---|---|---|---|---|---|---|
| RotorPy | https://github.com/spencerfolk/rotorpy | MIT | 2026-04-22 commit `9b760b8`; package metadata reported version 2.1.2 | `rotorpy/vehicles/`, vehicle parameters, controllers, `examples/basic_usage.py` | Later Stage E independent 6-DoF/motor/aerodynamic reference and fast Monte Carlo scaffold | No target-contact identification model; dynamics/contact conventions must be cross-checked; optional learning stack is unnecessary | No |
| PX4 Autopilot | https://github.com/PX4/PX4-Autopilot | BSD-3-Clause | Exact latest commit not captured because the commit-page request was rate-limited | SITL, multicopter flight stack, Gazebo/ROS 2 integration | Primary later end-to-end flight-controller baseline | Large build/toolchain; version and ROS/Gazebo compatibility must be pinned experimentally | No |
| PX4 Gazebo models | https://github.com/PX4/PX4-gazebo-models | BSD-3-Clause | 2026-08-09 commit `5577035` | `models/`, `worlds/`, `simulation-gazebo` | Reuse vehicle/world structure and launch workflow in integrated simulation | Repository instructions still mention a Gazebo Garden prerequisite in one path; must verify against the planned Harmonic stack before adoption | No |
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0 | GitHub metadata showed push on 2026-08-27 | `model/`, `python/`, `sample/`, MJCF contact/actuation | Independent fast contact-dynamics cross-validation | Contact softness/constraint parameters do not directly equal physical `k,c`; calibration and timestep sensitivity are required | No |
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie | Apache-2.0 for top-level content; per-model assets vary | Active repository observed; exact last commit not recorded | Per-model MJCF/XML and meshes | Possible later industrial/robot asset source | Each model has separate license; no candidate aerial platform or target has been selected | No |
| acados | https://github.com/acados/acados | BSD-2-Clause | 2026-08-12 commit `21376cb` | `examples/`, Python interface, NMPC, MHE, sensitivities | Later nominal/chance-constrained MPC and MHE benchmark | Native build, generated code, CasADi and solver dependencies; premature for Stage A | No |

## Audit notes

- No external source code or asset was copied into this repository.
- Current work uses only the locally installed Python dependencies recorded in
  each run manifest.
- Stars were not used as a selection criterion; technical fit, license, and
  reproducibility matter more for the current milestone.

