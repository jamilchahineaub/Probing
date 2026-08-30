# Verified achievements

Only evidence-backed results belong in this ledger.

## 2026-08-30 — EXP-0008 validated coupled UAV contact and exposed a transfer failure

- Scope: an explicit 6-DoF quadrotor with four lagged, saturated rotors; a
  rigid offset probe; unilateral contact; and the frozen one-dimensional target
  were exercised in 107 physical configurations and 321 low/nominal/high-noise
  policy evaluations. No Gazebo, PX4, ROS 2, MPC, RL, estimator redesign, or
  policy tuning was used.
- Verified mechanics: all 13 no-contact and nine contact validation checks
  passed, including hover, rigid-body signs, quaternion preservation, unilateral
  separation, the exact `r_probe x F_contact` moment, and energy/ring-down
  checks. The difficult impact case was timestep-converged. An independently
  written MATLAB model and a subsystem-based Simulink model agreed with the
  representative Python coupled trajectory to numerical precision.
- Verified negative transfer result: a `0.5 N` reference produced median
  realized-force RMS error `0.399 N`, median lag `224 ms`, and separation/
  recontact in every physical configuration. On the 94-target primary
  population, frozen nominal/high-noise false-safe rates rose to
  `11.76%/15.69%`; all full-sweep nominal false-safe cases were slow-settling
  failures created by the coupled sustained-contact behavior.
- Verified partial result: nominal/high-noise binary accuracy remained
  `82.98%/81.91%`, passive target ring-down stayed observable, maximum attitude
  excursion was only `2.893 deg`, and no rotor saturated. Thus the Stage 1/2
  signal remains visible but is not safe to transfer unchanged through this
  contact-delivery architecture.
- Reproducibility: 98 Python tests passed; Stage 1 fingerprints remained
  unchanged; the frozen Stage 2 checkpoint is
  `95fcdbb5d08f2ffa46077aced7f5c882fb2627f8` with tag
  `stage2-complete-20260830`.
- Evidence: [experiment record](lab/experiments/EXP-0008.md),
  [run summary](runs/EXP-0008_20260830T123909.142216Z_s13101_f41ef89f/summary.json),
  and [transfer audit](runs/EXP-0008_20260830T123909.142216Z_s13101_f41ef89f/postrun_analysis.json).
- Interpretation boundary: this validates the simulator and a scientifically
  important failure mechanism; it does not validate safe coupled probing or
  authorize Stage 3B.

## 2026-08-29 — EXP-0006 verified a zero-energy passive false-safe reduction

- Scope: one 0.5 N, 0.5–5 Hz chirp followed by exactly zero force; 240
  untouched one-dimensional targets; low, nominal, and high causal sensing;
  unchanged EXP-0005 physical risk labels. No active follow-up, integrated
  vehicle, or advanced estimator was used.
- Verified result: 0.5 s of passive observation reduced the EXP-0006 held-out
  nominal/high false-safe rates from `2.44%/3.66%` to `0.00%/0.00%`, while
  force-squared dose stayed exactly `0.206124 N²s` at every duration.
- Verified mechanism: chirp plus ring-down achieved `83.33%` nominal accuracy
  with zero false-safe decisions, while ring-down alone achieved only `57.08%`.
  Threshold dwell fraction and time-to-threshold were the strongest fitted
  ring-down settling diagnostics.
- Verified parameter result: adding estimated physical diagnostics changed
  nominal accuracy from `83.33%` to `83.75%` and left false-safe at zero.
  Accurate separate damping and effective mass were not required for this
  observed safety benefit.
- Reproducibility: all 12 integrity checks passed; no safety event occurred;
  EXP-0001 through EXP-0005 fingerprints were unchanged; 81 tests passed before
  the held-out run.
- Evidence:
  [manifest](runs/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2/manifest.json),
  [duration table](results/tables/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__duration_summary.csv),
  [false-safe plot](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__false_safe_vs_observation_duration.png),
  [decision](runs/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2/stage1_decision.json).
- Interpretation boundary: the full Stage gate did not pass. High-noise UNSAFE
  recall was `89.58%`, one case below 90%, and a fixed 3 s wait exceeded the
  practical-duration limit. This is not yet a certified classifier or
  authorization to leave Stage 1.

## 2026-08-29 — EXP-0005 isolated the safety-decision blind spot

- Scope: one fixed 0.5 N, 0.5–5 Hz chirp; causal synchronized sensing; 240
  continuous held-out targets; three noise regimes; hidden 2 N sustained-
  contact evaluation. No integrated vehicle or advanced estimator was used.
- Verified partial result: combined task features excluding estimated damping
  and mass reached `86.25%` nominal three-class accuracy, versus `78.33%` for
  the estimated full-parameter baseline. Peak-displacement median/p95 error was
  `4.84%/17.23%`.
- Verified mass result: the worst mass-error quartile retained `80.00%`
  decision accuracy and zero nominal false-safe cases. Accurate separate
  `m_eff` was not necessary for every correct safety decision.
- Verified limitation: high-noise false-safe rate was `6.49%`, above the 5%
  gate, and safe-force lower-bound coverage was only `93.75%` nominally. Every
  nominal false-safe case was an underestimated slow-settling response.
- Reproducibility: all ten integrity checks passed; validation probes had no
  safety events; EXP-0001 through EXP-0004 fingerprints were unchanged; 75
  tests passed before the held-out run.
- Evidence:
  [manifest](runs/EXP-0005_20260829T191551.247939Z_s6101_8de417c0/manifest.json),
  [classification table](results/tables/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__classification_summary.csv),
  [false-safe audit](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__confusion_and_false_safe.png),
  [mass/decision figure](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__parameter_accuracy_vs_decision_accuracy.png).
- Interpretation boundary: the kill criterion fired. This is a verified
  conditional result and failure diagnosis, not a demonstrated safe-contact
  classifier or authorization to leave Stage 1.

## 2026-08-29 — EXP-0004 falsified the sequential full-vector hypothesis

- Scope: 600 held-out Stage 1 target/strategy trials, 1,219 executed probes,
  seven bounded 1 N candidates, five strategy families, and twenty new seeds.
  No vehicle, MATLAB, or integrated simulation was used.
- Reproducibility: all eight integrity checks passed; EXP-0001 through EXP-0003
  fingerprints were unchanged before/after; zero safety events.
- Verified result: uncertainty-driven probing achieved `65.00%` full-vector
  success, below a `80.83%` fixed two-stage comparator at only `5.98%` dose
  mismatch. Adaptive worst-target p95 RMS error was also worse
  (`33.70%` versus `25.67%`). The predeclared kill criterion fired.
- Strongest baseline: one 0.5–5 Hz chirp achieved `90.83%` full-vector success
  with 3 s duration and `1.482 N² s` dose, but failed the strict rigid-damping
  and high-mass tail gates.
- Verified uncertainty limitation: every follow-up decision named damping as
  weakest; only `41.01%` of selected probes reduced its error, and predicted
  variance reduction had `rho = -0.478` with realized improvement.
- Evidence:
  [manifest](runs/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2/manifest.json),
  [strategy summary](results/tables/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__strategy_summary.csv),
  [fixed/adaptive comparison](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__fixed_vs_adaptive.png),
  [decision](runs/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2/stage1_decision.json).
- Tests: 69 passed before the held-out run.
- Interpretation boundary: this is strong negative evidence for the current
  active full-vector architecture, not proof that all active identification or
  decision-sufficient probing is impossible.

## 2026-08-29 — EXP-0003 isolated timing and EIV limits causally

- Scope: 17,880 held-out Stage 1 evaluations across six targets, six fixed-1 N
  chirp bands, timing offsets, four causal pipelines plus an offline reference,
  and OLS/TLS/IV. No vehicle or integrated simulation was used.
- Reproducibility: seeds 3101–3120; all integrity checks passed; EXP-0001 and
  EXP-0002 fingerprints were unchanged before/after; zero safety events.
- Verified timing result: worst-target damping bias became practically
  significant at the first tested 1 ms displacement or acceleration offset;
  mass and damping crossed thresholds at 5 ms force/filter relative delay.
- Verified causal result: synchronizing channels did not remove the causal mass
  error. Causal low-pass retained about 85% median absolute mass error in the
  baseline profile, while centered Savitzky–Golay remained substantially better
  and noncausal.
- Verified EIV result: IV reduced worst-target causal mass p95 from `108.40%`
  under OLS to `34.62%` for the 0.5–5 Hz condition, with five of six targets
  passing; weak high-mass instruments and poor damping prevent a general claim.
  TLS supplied no general improvement.
- Verified classifications: `k` category A, `c` category B, `m_eff` category B;
  no single causal full-vector candidate passed. Stage decision:
  `CONTINUE_STAGE_1`.
- Evidence:
  [manifest](runs/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53/manifest.json),
  [classifications](results/tables/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__parameter_classifications.csv),
  [timing plot](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__timing_bias.png),
  [EIV comparison](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__estimator_comparison.png).
- Tests: 65 passed before the held-out reference run.
- Interpretation boundary: parameter categories are parameter-specific; this is
  not a demonstrated physical full-vector estimator or deployment pipeline.

## 2026-08-29 — EXP-0002 verified the practical identifiability gap

- Scope: 19,800 validation-seed trials across six targets, five probes, two
  contact modes, three sensing severities, and eleven sensing/pipeline
  combinations. No MATLAB, EKF, UAV, or integrated simulation was used.
- Reproducibility: validation seeds 2101–2110; all eight integrity checks
  passed; EXP-0001 artifact fingerprint was unchanged before/after.
- Verified result: 0/30 nominal unilateral practical candidates passed the
  predeclared Stage 1 gate. Stage decision: `CONTINUE_STAGE_1`.
- Best candidate: chirp plus position-derived Savitzky–Golay
  velocity/acceleration. Worst-target p95 errors were `19.35%` for stiffness,
  `32.98%` for damping, and `62.58%` for effective mass.
- Negative evidence: all best-candidate trials were full rank, so the failure
  demonstrates EIV/timing sensitivity beyond structural rank. Sensorless-force
  proxy median/p95 errors were `82.85%`/`305.65%`.
- Ramp mechanism verified: `95.75%` force DC energy, only `0.148%` above target
  natural frequency, inertial contribution `0.696%` of force RMS, explaining
  fragile mass estimation.
- Evidence:
  [manifest](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/manifest.json),
  [decision](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/stage1_decision.json),
  [failure heatmap](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__failure_heatmap.png),
  [EIV shifts](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__errors_in_variables.png).
- Tests: 54 passed before the reference validation run.
- Interpretation boundary: this verifies a Stage 1 negative/practical-limit
  result, not deployable sensing or sensorless force estimation.

## 2026-08-28 — Milestone A development matrix demonstrated conditional identification

- Scope: exact linear, bilateral, force-driven one-dimensional target; six
  target types, five unoptimized bounded probes, and perfect/near-ideal direct
  measurements. This is a development-seed result, not held-out validation.
- Evidence: all 30 perfect-data regressions had rank 3. Maximum perfect batch
  LS and RLS RMS relative errors were `2.23e-13` and `6.50e-12`; near-ideal
  median error was `0.488%` for both.
- Safety: all probes remained within `1 N`; no configured displacement,
  velocity, acceleration, or force limit was exceeded.
- Reference run:
  `EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69`.
- Evidence entry points:
  [manifest](runs/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69/manifest.json),
  [case metrics](results/tables/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__case_metrics.csv),
  [parameter comparison](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__parameter_comparison.png),
  [observability diagnostics](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__observability.png).
- Model verification: numerical free and forced responses are checked against
  analytical solutions; equilibrium, damping/energy, zero damping, and a very
  high stiffness limit are covered by the test suite.
- Tests: 42 passed in the reference verification run.
- Important negative evidence: the near-ideal rigid/ramp case had `36.46%` RMS
  error, mainly a `62.68%` effective-mass underestimate; worst parameter
  correlation was `0.985`. The milestone therefore demonstrates conditional,
  probe-dependent identifiability rather than universal success.
- Exclusions: no EKF, UAV, Gazebo, PX4, ROS 2, MPC, Isaac Sim, optimized probe,
  nonlinear contact, or hardware result is claimed.

## 2026-08-28 — Preliminary Kelvin–Voigt evaluator validation

- Scope: bilateral algebraic `F = k*x + c*x_dot` response under one prescribed
  raised-cosine displacement.
- Evidence: force error was zero and relative numerical energy-balance error
  was `7.49e-7` in run
  `EXP-0001_20260828T170017.246126Z_s1101_1bc4b363`.
- Limitation: this historical pilot did not estimate unknown parameters and is
  not the current EXP-0001 identification result.
## Independent MATLAB/Simulink replication

Stage 2 independently reproduced the decision-sufficient interaction result
on 1,360 new MATLAB targets. Nominal and high-noise false-safe rates were zero
with a 0.586% one-sided 95% upper bound, and MATLAB/Simulink dynamics agreed to
floating-point precision. This verifies that the Stage 1 conclusion is not a
Python-specific numerical artifact.
