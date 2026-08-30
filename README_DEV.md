# Development notes

This repository is intentionally limited to Stage 1: a reduced-order
one-dimensional interaction-identification problem. It does not implement a
UAV or an integrated robotics simulator.

## What currently works

- Force-driven mass-spring-damper target:
  `F = m_eff*x_ddot + c*x_dot + k*x`.
- Fixed-step RK4 integration and analytical free/constant-force solutions.
- Bilateral Kelvin–Voigt constitutive evaluator retained from the pilot.
- Bounded ramp, half-sine, sinusoid, linear chirp, and multisine force probes.
- Perfect or configurable seeded Gaussian displacement, velocity,
  acceleration, and force measurements.
- Batch least squares and recursive least squares for `[k, c, m_eff]`.
- Rank, normalized conditioning, and parameter-correlation diagnostics.
- Immutable runs containing resolved configuration, seeds, compressed raw
  CSV/NPZ data, per-case/summary metrics, acceptance/safety records, manifest,
  and PNG/PDF figures.
- EXP-0001 with six target types, five probes, and two measurement scenarios.
- No-tension unilateral contact with explicit contact-loss intervals.
- Finite-difference, causal low-pass, and Savitzky–Golay derivative pipelines.
- Direct, derivative-based, IMU-like, and exploratory force-proxy sensing with
  configurable rate, latency, timestamp mismatch, and noise.
- Errors-in-variables, modal-parameter, RLS convergence, physical disturbance,
  and heuristic information/disturbance metrics.
- EXP-0002 with 19,800 validation-seed Monte Carlo trials and explicit GO/NO-GO.
- Fractional timestamp and filter-delay sweeps for synchronized chirp sensing.
- Strictly backward differences, causal low-pass derivatives, trailing causal
  polynomial differentiation, and alpha-beta-gamma state tracking.
- Standardized total least squares and input-history instrumental variables.
- Regressor variance/cross-correlation, residualized Fisher information,
  effective delay, noise attenuation, and computation-cost diagnostics.
- EXP-0003 with 17,880 held-out timing/pipeline/EIV/frequency evaluations and
  parameter-specific A/B/C classifications.
- EXP-0004 fixed, repeated, predefined, and uncertainty-driven bounded probing
  with a matched-disturbance kill test.
- EXP-0005 fixed-chirp task features, hidden sustained-contact labels,
  conservative outcome bounds, three-class risk prediction, and safe-force
  diagnostics across a broad continuous target population.
- EXP-0006 zero-force causal ring-down prefixes, decay/persistence features,
  matched-energy safety comparison, locked historical-failure audit, and
  truth-independent early stopping.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

The workstation environment can run the code without an editable install by
setting `PYTHONPATH=src`. The current implementation does not import SciPy.

## Verification

Run all unit and experiment tests:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 -m pytest -q
```

The suite covers analytical free and forced responses, equilibrium,
energy/damping behavior, `c=0`, very high stiffness, probe bounds, sensor
noise, batch LS, RLS, and rank deficiency.

The current suite has 81 tests and also covers unilateral contact, practical
and causal derivatives/filters, sensing regimes, OLS/TLS/IV, force-proxy
mismatch, sequential probing, response-derived risk labels, stable outcome
transforms, prefix-causal ring-down features, truth-independent stopping, and
reduced EXP-0002 through EXP-0006 smoke execution.

## Reproduce EXP-0001

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_experiment.py \
  --config configs/experiments/exp_0001_interaction_identification.yaml
```

The command prints a unique run ID. It never overwrites an existing run.
`runs/<run-id>/manifest.json` is the provenance entry point.

The recorded reference is
`EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69`. See
[EXPERIMENTS.md](EXPERIMENTS.md) for interpretation.

## Reproduce EXP-0002

```bash
PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_exp_0002.py \
  --config configs/experiments/exp_0002_practical_identifiability.yaml
```

This is a full 19,800-trial run. The recorded validation reference is
`EXP-0002_20260828T212540.955871Z_s2101_e8713743`, whose decision is
`CONTINUE_STAGE_1`.

## Reproduce EXP-0003

```bash
PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_exp_0003.py \
  --config configs/experiments/exp_0003_causal_eiv_identifiability.yaml
```

This creates 17,880 timing/profile/identification evaluations using the frozen
held-out seed partition. The recorded reference is
`EXP-0003_20260828T225126.077199Z_s3101_b2da5c53`; its decision is
`CONTINUE_STAGE_1`.

## Reproduce EXP-0004

```bash
PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_exp_0004.py \
  --config configs/experiments/exp_0004_sequential_active_identification.yaml
```

The recorded reference is
`EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2`; its active full-vector kill
criterion fired and its Stage 1 decision is `CONTINUE_STAGE_1`.

## Reproduce EXP-0005

```bash
PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_exp_0005.py \
  --config configs/experiments/exp_0005_decision_sufficiency.yaml
```

The recorded reference is
`EXP-0005_20260829T191551.247939Z_s6101_8de417c0`. Its integrity checks passed,
but the conservative decision-sufficiency kill criterion fired and the Stage 1
decision remains `CONTINUE_STAGE_1`.

## Reproduce EXP-0006

```bash
PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_exp_0006.py \
  --config configs/experiments/exp_0006_passive_ringdown.yaml
```

The recorded reference is
`EXP-0006_20260829T194621.991365Z_s8101_2966d3d2`. It verified zero
nominal/high false-safe decisions after 0.5 s of passive observation, but the
strict three-class/duration gate missed by one high-noise UNSAFE-recall case;
the Stage 1 decision remains `CONTINUE_STAGE_1`.

## Repository structure

```text
configs/              experiment inputs and acceptance criteria
src/probeing/models/  reduced physical models and analytical solutions
src/probeing/probing/ bounded force probes
src/probeing/measurements/ synthetic measurement generation
src/probeing/estimators/ LS/RLS and observability diagnostics
src/probeing/experiments/ configured experiment matrix
src/probeing/plotting/ scientific figures
scripts/              run and log entry points
tests/                deterministic mathematical/experiment tests
lab/                  daily and experiment interpretation records
runs/                 immutable per-run raw artifacts and manifests
results/              indexed figures and metric tables
```

Directories reserved for later integrated simulation are not part of the
current implementation.

## Seed policy

- Development: 1000–1999
- Earlier held-out experiments: 2000–4999
- EXP-0005 training/calibration: 5101–5120 / 5201–5210
- EXP-0005 untouched validation: 6101–6120
- EXP-0006 training/calibration: 7101–7120 / 7201–7210
- EXP-0006 untouched validation: 8101–8120

EXP-0001 uses development seed 1101 plus deterministic per-case offsets.
EXP-0002 uses validation seeds 2101–2110. EXP-0003 uses the untouched held-out
3101–3120 partition after development only on 1601–1610. EXP-0004 uses
4101–4120 after development on 1701–1705. EXP-0005 uses disjoint 5101–5120
training, 5201–5210 calibration, and 6101–6120 validation partitions after
development only in the 1800-series. EXP-0006 uses 7101–7120 training,
7201–7210 calibration, and 8101–8120 validation after development only in the
1900-series.

## Known limitations

- The target is exact, linear, and one-dimensional. The unilateral option
  clamps tensile force but has no probe gap, impact, or probe-body dynamics.
- Savitzky–Golay is centered/offline and is not a causal deployment pipeline.
- Causal low-pass, polynomial, and alpha-beta-gamma settings are common across
  targets; none is a deployment-validated observer.
- The exploratory force proxy is command based, not a UAV wrench observer, and
  failed strongly.
- Ordinary least squares with noisy response regressors is an
  errors-in-variables baseline and can be biased.
- Standardized TLS assumes isotropic errors after scaling and failed generally.
  Weighted TLS is intentionally absent because the colored/correlated derived
  error covariance is not yet defensible.
- IV uses the known chirp/contact-activation history; weak instruments and later
  actuator/contact mismatch remain limitations.
- EXP-0003 classifications are parameter-specific. No single causal estimator
  passed the physical full-vector gate.
- EXP-0005's combined task features improved overall decision accuracy without
  estimated damping or mass, but the fixed chirp missed slow-settling targets,
  exceeded the high-noise false-safe threshold, and did not provide a 95%-
  reliable safe-force lower bound.
- EXP-0006's 0.5 s passive policy removed false-safe decisions in one held-out
  population, but conservatively moved additional SAFE targets to CAUTION and
  missed the strict high-noise UNSAFE-recall gate by one case. It requires
  independent replication and is not a certified classifier.
- The rigid/ramp near-ideal case has weak effective-mass identification; this
  is an experimental result, not a defect to conceal by retuning RLS.
- The reference manifests record dirty worktrees and should be committed with
  their code/artifacts for commit-level provenance.
- No EKF, optimized probe, UAV dynamics, Gazebo, PX4, ROS 2, MPC, Isaac Sim, or
  hardware is implemented or claimed.
