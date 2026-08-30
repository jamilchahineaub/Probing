# Configurations

Experiment configurations contain hypotheses, seed partitions, physical
parameters, safety limits, and acceptance criteria. Acceptance thresholds must
be declared before a run is interpreted.

The authoritative Milestone A experiment is
`experiments/exp_0001_interaction_identification.yaml`. The older
`exp_0001_kelvin_voigt_validation.yaml` is retained only to reproduce the
historical prescribed-displacement pilot.

The Stage 1 realistic-sensing validation is
`experiments/exp_0002_practical_identifiability.yaml`. It freezes validation
seeds, common sensing-pipeline settings, imperfection severities, the EXP-0001
artifact fingerprint, and GO/NO-GO criteria.

The Stage 1 causal timing/EIV validation is
`experiments/exp_0003_causal_eiv_identifiability.yaml`. It freezes held-out
seeds 3101–3120, predecessor fingerprints, fractional timing sweeps, common
causal processing settings, OLS/TLS/IV, six fixed-amplitude chirp bands,
parameter-specific classifications, and the full-vector Stage 1 gate.

The Stage 1 sequential-probing validation is
`experiments/exp_0004_sequential_active_identification.yaml`. It freezes new
held-out seeds 4101–4120, predecessor fingerprints, the seven-probe library,
parameter-specific causal estimators, measurable uncertainty selector, stopping
criteria, cumulative disturbance budget, matched-dose comparison, kill
criterion, and Stage 1 gate.

The Stage 1 decision-sufficiency validation is
`experiments/exp_0005_decision_sufficiency.yaml`. It freezes disjoint training,
calibration, and untouched validation partitions; a single 0.5 N, 0.5–5 Hz
chirp; causal synchronized sensing; response-derived sustained-contact risk
labels; transparent feature/predictor baselines; conservative false-safe and
response-bound criteria; and the Stage 1 gate.

The Stage 1 passive ring-down validation is
`experiments/exp_0006_passive_ringdown.yaml`. It preserves the EXP-0005 target
and risk definitions, appends zero-force causal observation prefixes, freezes
new 7100/7200/8100-series seed partitions, compares chirp and ring-down feature
sets, defines a truth-independent early-stopping rule, locks an evaluation-only
EXP-0005 failure audit, and predeclares stricter false-safe and duration gates.
