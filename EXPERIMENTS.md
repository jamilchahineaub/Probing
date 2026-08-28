# Experiment index

## EXP-0001 — Kelvin–Voigt analytical validation under a bounded smooth ramp

### Hypothesis

A Kelvin–Voigt evaluator driven by a prescribed smooth ramp will reproduce the
analytical force decomposition and mechanical energy balance within configured
numerical tolerances while respecting the probe safety bounds.

### Configuration

- Configuration:
  [exp_0001_kelvin_voigt_validation.yaml](configs/experiments/exp_0001_kelvin_voigt_validation.yaml)
- Target: bilateral Kelvin–Voigt, `k = 500 N/m`, `c = 20 N·s/m`.
- Probe: 5 mm raised-cosine ramp over 1 s, 0.25 s hold, `dt = 0.001 s`.
- Seed partition: development; seed 1101. The experiment is deterministic and
  records the seed for schema consistency.
- Safety limits: `|F| <= 3 N`, `x <= 5.1 mm`, `|x_dot| <= 10 mm/s`.
- Acceptance: force RMSE and maximum error `<= 1e-12 N`, relative energy
  residual `<= 1e-5`, displacement overshoot `<= 1e-12 m`, no safety event.

### Run IDs

- `EXP-0001_20260828T170017.246126Z_s1101_1bc4b363` — success.

### Results

- Force RMSE / maximum error: `0 N` / `0 N`.
- Relative energy-balance error: `7.485853698262815e-7`.
- Peak force: `2.5098306749339105 N`.
- Peak displacement: `0.005 m`; peak speed: `0.007853981633974483 m/s`.
- Final work: `0.00686684513464828 J`; stored energy: `0.00625 J`;
  dissipated energy: `0.0006168502750680846 J`.
- Safety violations: 0.
- Plot:
  [response and energy figure](results/figures/EXP-0001_20260828T170017.246126Z_s1101_1bc4b363__kelvin_voigt_validation.png).
- Metrics table:
  [CSV](results/tables/EXP-0001_20260828T170017.246126Z_s1101_1bc4b363__metrics.csv).
- Raw/provenance entry point:
  [manifest](runs/EXP-0001_20260828T170017.246126Z_s1101_1bc4b363/manifest.json).

### Conclusion

The implementation and numerical energy accounting meet the predeclared
criteria for this configuration. The damping component causes the peak force
to exceed the final static force slightly. The experiment provides no evidence
about parameter identifiability because it uses known parameters and noiseless,
prescribed kinematics.

### Next experiment

EXP-0002 will validate the effective-mass extension against analytical free
responses and forced-response convergence before probe-signal identification
comparisons begin.

