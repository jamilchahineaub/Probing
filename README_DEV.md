# Development notes

This repository currently implements the first reduced-order validation step
for safe aerial probing. It is not yet an aerial-vehicle simulator or a target
identification system.

## What currently works

- One-dimensional Kelvin–Voigt force evaluation, with explicit bilateral and
  unilateral contact modes.
- A bounded raised-cosine ramp with analytical velocity and acceleration.
- Force-error, safety-limit, and mechanical-energy validation metrics.
- Immutable experiment run directories containing resolved configuration, raw
  CSV/NPZ output, metrics, safety events, acceptance checks, a manifest, and
  generated PNG/PDF figures.
- EXP-0001, which validates one Kelvin–Voigt configuration against its
  analytical constitutive response.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

The existing workstation environment can run the current code without an
editable install by setting `PYTHONPATH=src`. Its SciPy 1.8.0 and NumPy 1.26.4
versions are incompatible; current code intentionally does not import SciPy.

## Reproduction

Run the test suite:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/probeing-matplotlib python3 -m pytest -q
```

Create a new, non-overwriting reproduction of EXP-0001:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/probeing-matplotlib \
python3 scripts/run_experiment.py \
  --config configs/experiments/exp_0001_kelvin_voigt_validation.yaml
```

The command prints the unique run ID. The run manifest under `runs/<run-id>/`
is the provenance entry point. Reproductions create new run directories and do
not replace the recorded reference run.

Create a daily log template when work begins on a new date:

```bash
python3 scripts/create_daily_log.py
```

## Repository structure

```text
configs/              versioned experiment inputs and acceptance criteria
src/probeing/         models, probes, metrics, experiments, and plotting code
sim/                  later reduced/integrated simulator adapters
ros2_ws/              reserved for the later ROS 2 integration milestone
scripts/              experiment and research-log entry points
tests/                deterministic mathematical and safety tests
literature/           supporting literature artifacts when licenses permit
lab/                  daily logs, experiment notes, and decision notes
data/                 external/raw datasets; no dataset is present yet
runs/                 immutable per-run artifacts
results/              indexed figures, tables, and reports
docs/                 technical documentation
```

## Seed partition policy

- Development seeds: 1000–1999
- Validation seeds: 2000–2999
- Final test seeds: 3000–3999

EXP-0001 uses development seed 1101. No validation or final-test result exists.

## Known limitations

- The validated model is a prescribed one-dimensional boundary condition; it
  has no UAV, actuator, sensor, contact-normal, or feedback dynamics.
- EXP-0001 uses bilateral contact so that the Kelvin–Voigt energy identity is
  analytically testable. Unilateral contact loss is implemented but has not
  received an experiment-level validation.
- The force oracle and evaluator are algebraic. Identifiability, measurement
  noise, uncertainty calibration, and parameter estimation are untested.
- Only a smooth ramp has an experiment record. Half-sine, sinusoid, chirp,
  multisine, and physically filtered PRBS probes remain to be compared.
- This folder is not a valid Git repository root, so the reference run records
  a null project commit SHA and the reason. This is a reproducibility blocker.

