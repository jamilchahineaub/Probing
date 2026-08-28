# Verified achievements

Only evidence-backed results belong in this ledger.

## 2026-08-28 — Kelvin–Voigt evaluator passed EXP-0001

- Scope: one-dimensional bilateral Kelvin–Voigt response under one bounded
  raised-cosine ramp at `k = 500 N/m`, `c = 20 N·s/m`, and `dt = 0.001 s`.
- Evidence: force RMSE `0 N`, maximum force error `0 N`, relative
  energy-balance error `7.48585e-7`, displacement overshoot `0 m`, and zero
  safety-limit violations.
- Run ID:
  `EXP-0001_20260828T170017.246126Z_s1101_1bc4b363`.
- Tests: 13 tests passed with no warnings in the final verification command.
- Plot:
  [Kelvin–Voigt kinematics, force decomposition, and energy balance](results/figures/EXP-0001_20260828T170017.246126Z_s1101_1bc4b363__kelvin_voigt_validation.png).
- Table:
  [EXP-0001 metrics](results/tables/EXP-0001_20260828T170017.246126Z_s1101_1bc4b363__metrics.csv).
- Limitation: this does not verify a UAV model, an estimator, identifiability,
  uncertainty calibration, unilateral contact, or decision safety.

