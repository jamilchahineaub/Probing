# Stage 2 — independent MATLAB/Simulink validation

## Decision

**PASS.** The independently reconstructed MATLAB/Simulink experiment supports
the locked Stage 1 conclusion. Stage 3 is authorized by the scientific gate
but was not started.

## Frozen experiment

- Run: `STAGE2_20260829T221139.099Z_s20101`
- MATLAB: R2026a Update 5; Simulink fixed-step `ode4`
- Policy: immutable `EXP-0006-primary-0p5s-chirp-ringdown` bundle, SHA-256
  `7cc1669f897ea1c917b5f3b289412bdbbc0187ac9a3979ca38240078f1d8c015`
- Probe: unilateral bounded 0.5 N, 0.5–5 Hz chirp for 3 s, then 0.5 s at zero
  commanded force
- Population: 960 independent broad MATLAB targets plus 400 independently
  generated boundary targets; seeds 20101–20300, disjoint from Stage 1
- Predictions were completed before final sustained-contact outcomes were
  generated and joined. No fitting, calibration, feature, or threshold change
  occurred.

## Verification

Seven MATLAB tests passed. Maximum analytical displacement errors were
`1.52e-16 m` (free), `1.39e-15 m` (forced), `4.19e-15 m` (undamped), and
`3.92e-16 m` (overdamped). Equilibrium displacement error was `5.74e-13 m`;
undamped energy drift was `2.34e-16 J`.

The Simulink model contains separate Probe Command, Target Dynamics, Sensors
and Noise, Causal Signal Processing, Feature Extraction, SAFE/NON-SAFE
Decision, and Sustained-Contact Ground Truth subsystems. Across six targets,
MATLAB/Simulink maximum differences were `7.96e-17 m`, `2.18e-15 m/s`, and
`1.65e-13 m/s^2`; all six resulting binary decisions matched. Post-hoc matched
Python/MATLAB differences were also at
floating-point scale (`1.13e-16 m`, `3.32e-15 m/s`, `2.04e-13 m/s^2`).

## Primary results

| Noise | False-safe | One-sided 95% upper | Accuracy | SAFE precision | NON-SAFE recall |
|---|---:|---:|---:|---:|---:|
| Low | 0/510 (0%) | 0.586% | 87.94% | 100% | 100% |
| Nominal | 0/510 (0%) | 0.586% | 88.38% | 100% | 100% |
| High | 0/510 (0%) | 0.586% | 87.65% | 100% | 100% |

The boundary subset contained 200 SAFE and 200 NON-SAFE cases. Nominal/high
boundary false-safe rates were both 0%, with a 1.487% one-sided upper bound;
binary accuracy was 85.5%/85.0%. There were zero probe safety violations.

## Secondary results and interpretation

Nominal three-class accuracy was 81.91%. Median/p95 peak-displacement relative
error was 4.12%/15.89%, and settling absolute error was 0.079/1.051 s. Separate
nominal `k/c/m_eff` diagnostic median errors were 3.76%/21.91%/10.52%; p95
errors were 7.56%/330.5%/58.43%. Damping and effective-mass tails remained
poor while the binary decision remained reliable, independently supporting
the Stage 1 task-sufficiency conclusion.

Compared with EXP-0007, MATLAB nominal accuracy was 0.37 percentage point
lower and high-noise accuracy 1.40 points lower. MATLAB observed zero
false-safe cases versus one nominal and two high-noise Python cases. MATLAB's
population had 510 NON-SAFE targets versus 521 in Python. These differences
are consistent with the independent random population and sensor-noise stream;
matched trajectories rule out a material numerical-integration discrepancy.

## Artifacts

Artifacts are under `runs/STAGE2_20260829T221139.099Z_s20101/`, including the
manifest, raw workspace, predictions, metrics, tests, cross-checks, and figures.
Figures and key tables are also indexed under `results/`.
