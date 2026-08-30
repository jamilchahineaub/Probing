# Project state

## Current hypothesis

The fixed chirp plus passive ring-down signal is valid for the frozen reduced-
order target experiment, but EXP-0008 shows that it does not transfer as a
safe decision policy when a controlled 6-DoF vehicle, offset probe, unilateral
separation/re-contact, and imperfect force delivery are included. Target-only
ring-down remains measurable; it does not predict settling failures created by
the later coupled contact controller.

## Current milestone

EXP-0008 completed the first Stage 3A coupled baseline. The 6-DoF vehicle,
conventional controller, unilateral contact, and offset wrench are validated,
but the frozen safety policy failed its transfer test. Stage 3A therefore
remains **CONTINUE_STAGE_3A**. Stage 3B, Gazebo, PX4, ROS 2, MPC, RL, Isaac
Sim, perception, and hardware remain unauthorized.

## Completed evidence

### EXP-0001

- Exact bilateral model and analytical/numerical validation.
- All perfect cases rank 3.
- Near-ideal median error `0.488%`; chirp best at `0.136%` mean error.
- Ramp exposed weak inertial excitation and `62.68%` mass underestimation.

### EXP-0002

- 19,800 trials on seeds 2101–2110.
- Realistic noise/rate/timing/filtering, unilateral contact, and no-direct-
  acceleration pipelines.
- Best offline chirp candidate worst-target p95:
  `19.35%/32.98%/62.58%` for `k/c/m_eff`.
- 0/30 candidates passed; Stage 1 continued.

### EXP-0003

- Reference run:
  `EXP-0003_20260828T225126.077199Z_s3101_b2da5c53`.
- Twenty new held-out seeds, six chirp bands, four causal pipelines plus the
  centered offline reference, and OLS/TLS/IV.
- 4,680 one-factor timing trials, 2,400 timing-profile trials, and 10,800
  identification trials; zero safety events and all integrity checks passed.
- EXP-0001/0002 fingerprints were unchanged.

### EXP-0004

- Reference run:
  `EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2`.
- Twenty new held-out seeds, five strategy families, seven bounded candidate
  probes, and parameter-specific causal estimators inherited from EXP-0003.
- 600 complete strategy trials and 1,219 executed probes; zero safety events;
  all integrity checks passed; EXP-0001/0002/0003 fingerprints unchanged.
- Single 0.5–5 Hz chirp: `90.83%` full-vector success and `20.92%`
  worst-target p95 RMS error.
- Adaptive sequence: `65.00%` success and `33.70%` worst-target p95 RMS error.
- Matched-dose predefined two-stage comparator: `80.83%` success and `25.67%`
  p95; adaptive success was 15.83 points lower.
- The kill criterion fired: `STOP_ACTIVE_FULL_VECTOR_IDENTIFICATION` for this
  architecture. Stage decision remains `CONTINUE_STAGE_1`.

### EXP-0005

- Reference run:
  `EXP-0005_20260829T191551.247939Z_s6101_8de417c0`.
- Disjoint 240-target training, 120-target calibration, and 240-target
  untouched validation populations; three causal noise regimes and 7,920
  held-out predictor rows.
- Combined task features excluded estimated `c` and `m_eff` and achieved
  `86.25%` nominal accuracy versus `78.33%` for the full-parameter baseline.
- False-safe rates were `2.60%`, `3.90%`, and `6.49%` for low, nominal, and
  high noise. High noise exceeded the predeclared `5%` limit.
- The primary predictor retained `80.00%` accuracy in the worst mass-error
  quartile and `85.29%` when mass error exceeded `30%`; neither subset had a
  nominal false-safe case.
- Nominal peak-displacement median/p95 error was `4.84%/17.23%`, but the safe-
  force lower bound covered only `93.75%` of cases.
- All validation integrity checks passed and validation probes had no safety
  events. The decision-sufficiency kill criterion fired; Stage 1 continues.

### EXP-0006

- Reference run:
  `EXP-0006_20260829T194621.991365Z_s8101_2966d3d2`.
- Disjoint 240-target validation population on seeds 8101–8120, three causal
  noise regimes, seven passive observation prefixes, and 20,160 held-out
  prediction rows.
- The same 0.5 N, 0.5–5 Hz chirp was followed by exactly zero force. Probe dose
  stayed at `0.206124 N²s` for every wait duration.
- At 0.5 s, false-safe rate was zero in low, nominal, and high noise. Nominal
  and high accuracy were `83.33%/82.92%`; UNSAFE recall was
  `91.67%/89.58%`.
- Ring-down-only accuracy was `57.08%` nominally. Chirp plus ring-down reached
  `83.33%` with no false-safe; adding separate `c/m_eff` diagnostics changed
  accuracy by only 0.42 point and did not improve false-safe performance.
- A causal early-stop rule had a 1.5 s median but 3 s p95 wait. The 3 s fixed
  window met the nominal/high numeric classification limits but exceeded the
  2 s practical limit.
- All 12 integrity checks passed, zero safety events occurred, and EXP-0001
  through EXP-0005 fingerprints remained unchanged. The strict Stage gate
  failed by one high-noise UNSAFE-recall case; Stage 1 continues.

### EXP-0008

- Reference run: `EXP-0008_20260830T123909.142216Z_s13101_f41ef89f`.
- Thirteen no-contact and nine contact-mechanics checks passed; the 1 ms RK4
  result agreed with 0.5 ms integration to 0.02% in RMS/peak probe force for
  the convergence case.
- 107 coupled configurations covered seven representative targets, a 27-case
  structured grid, 60 untouched Monte Carlo targets, and 13 one-factor
  vehicle/contact/actuator perturbations. The frozen policy was evaluated at
  three noise levels without refitting.
- The 94-target physical population achieved `82.98%` nominal and `81.91%`
  high-noise binary accuracy, but false-safe rates were `11.76%` and `15.69%`
  (one-sided 95% upper bounds `21.91%` and `26.53%`). Boundary nominal/high
  false-safe rates were `53.33%/60.00%`.
- Median desired-versus-realized force RMS error was `0.399 N` (`152.23%`
  relative); median correlation delay was `224 ms`; 44.86% of trials had no
  frequency bin satisfying the declared ±3 dB/45° delivery criterion.
- Re-contact occurred in every trial. Realized peak force exceeded the
  prospective `1.25 N` probe limit in 101/107 cases; median peak was `2.52 N`
  despite a fixed `0.5 N` reference. No rotor saturation occurred and maximum
  attitude disturbance was `2.89°`.
- Coupling changed the reduced versus coupled binary ground-truth label in
  33/107 configurations (20/94 target-population cases). All 19 nominal false-
  safe cases failed only the coupled settling criterion, and all 19 had a
  coupling-induced ground-truth label change.
- Independent MATLAB equations matched the representative Python force trace
  within `1.13e-11 N`; the subsystem Simulink checks executed successfully.

## Current parameter classification

| Parameter | Category | Evidence |
|---|---|---|
| Stiffness `k` | A — practical | all 6 targets under 0.25–2 Hz alpha-beta-gamma + TLS; worst p95 6.74% |
| Damping `c` | B — restricted | 4/6 targets under 0.5–5 Hz causal low-pass + OLS; worst p95 32.23% |
| Effective mass `m_eff` | B — restricted | 5/6 targets under 0.5–5 Hz causal low-pass + IV; worst p95 34.62% |
| Natural frequency ratio | B — restricted | 5/6 targets; one target invalid |
| Damping ratio | C — not practical | no qualifying causal condition |
| Inverse-mass ratio | C — not practical | only 3/6 targets pass |

Classifications are parameter-specific. Different estimators win each physical
parameter, and no single causal estimator identifies the complete vector. The
stiffness-winning TLS condition produces nonphysical estimates of other
parameters for the high-damping target.

## Strongest findings

- Timing is a serious damping risk. A 1 ms position or acceleration offset can
  create practically significant damping bias for the lightly damped target.
- Force timestamp and filter group-delay errors become practically significant
  for damping and mass at 5 ms in the worst target.
- Timing correction alone is insufficient: synchronized causal-low-pass median
  mass error remains about 85%.
- IV can reduce causal mass EIV bias under a 0.5–5 Hz chirp, but the high-mass
  target has weak instruments and remains outside the limits.
- TLS is not a general improvement; its standardized isotropic-error assumption
  does not match correlated, colored derived-regressor errors.
- Absolute acceleration strength predicts mass accuracy better than normalized
  conditioning. Baseline acceleration RMS spans 0.136–2.663 m/s² while
  normalized conditions remain only about 1.2–2.5.
- Higher frequency increases inertial information but also increases causal
  phase sensitivity. Information alone did not guarantee lower estimator error.
- EXP-0004's adaptive selector labeled damping as weakest in all 139 follow-up
  decisions. Only 41.01% of selected probes reduced damping error, and expected
  variance reduction correlated negatively with realized improvement
  (`rho = -0.478`).
- Confidence stopping was much better calibrated than probe choice: 59/60
  confidence stops met the accuracy thresholds. However, 69/79 trials told to
  continue after probe one were already acceptable, and their final success
  fell to 48.10%.
- A single fixed 0.5–5 Hz chirp is the strongest current empirical method. It
  still misses rigid-target damping and high-mass mass p95 gates, so it does not
  authorize Stage 1 progression.
- EXP-0005 showed that combined frequency/time/stiffness features outperform
  both stiffness-only and estimated full-vector features for three-class risk
  accuracy (`86.25%` versus `45.42%` and `78.33%` nominally).
- Accurate mass was not necessary for many correct decisions: the worst mass-
  error quartile retained `80.00%` nominal accuracy with no false-safe case.
- All nominal false-safe cases were slow-settling targets whose displacement,
  velocity, and oscillation magnitude stayed inside the safe envelope. Their
  true settling times were `2.932–3.000 s`, while predicted upper bounds were
  only `1.538–1.956 s`.
- EXP-0006 verified that the blind spot is observable passively: a 0.5 s
  zero-force prefix changed `9.17%` of nominal decisions and reduced held-out
  nominal/high false-safe rates from `2.44%/3.66%` to zero.
- Threshold dwell fraction and time-to-threshold were the strongest ring-down
  settling diagnostics. Their value is conservative persistence detection;
  ring-down alone is not an adequate safety predictor.
- Separate damping and mass diagnostics did not improve the primary safety
  metric beyond chirp plus ring-down. Complete `[k,c,m_eff]` identification is
  not necessary for the observed SAFE/non-SAFE benefit.

## Pipeline state

Causal low-pass plus derivatives is the best overall causal compromise. At the
baseline band it gives about 16.10 dB acceleration-noise attenuation with
15 ms effective delay. Trailing polynomial and alpha-beta-gamma pipelines
attenuate more noise but add 30–40 ms effective delay and poor damping phase.

Centered Savitzky–Golay remains the offline upper reference. No causal pipeline
approaches it jointly across `k`, `c`, and `m_eff`.

## Estimator and parameterization state

- OLS remains the most stable general baseline.
- TLS is retained as negative evidence, not selected.
- IV is retained as a restricted mass diagnostic; its command-derived
  instruments require adequate strength and correct contact activation.
- Keep `[k, c, m_eff]`. Do not adopt `[omega_n, zeta, 1/m_eff]`; damping ratio
  and inverse mass are category C in the ratio regression.
- Keep `m_eff` in the physical model and as a diagnostic, but do not make an
  accurate separate estimate a prerequisite for every safety decision. EXP-0005
  did not justify removing it from future mechanistic models.

## Current blockers

- The force-domain chirp is not physically delivered within its disturbance
  bound. Separation followed by target rebound produces re-contact impulses,
  and the vehicle/controller bandwidth does not reproduce the positive half-
  chirp above roughly the low-frequency portion of the sweep.
- The hidden sustained-contact maneuver itself is no longer the reduced-order
  2 N input: vehicle/contact control error continuously excites the target and
  increases median settling time from `0.0 s` reduced-order to `2.822 s`
  coupled. The frozen probe policy cannot foresee controller-induced settling.
- The current Stage 3A baseline violates the Stage 1/2 false-safe requirement;
  neither Stage 3B nor higher-fidelity simulation is authorized.

- The 0.5 s passive policy removed false-safe decisions, but high-noise UNSAFE
  recall was `89.58%`, one target below the predeclared 90% three-class gate.
  The miss was conservative UNSAFE-to-CAUTION rather than UNSAFE-to-SAFE.
- A 3 s fixed observation met the nominal/high numeric classification limits
  but exceeded the 2 s practical-duration gate. Causal early stopping had a
  practical 1.5 s median but a 3 s p95 wait.
- The zero false-safe result comes from one 240-target held-out population and
  needs independent replication with the policy fully locked.
- The nominal safe-force lower bound has `93.75%` rather than `95%` coverage,
  so maximum safe interaction force is not yet reliably bounded.
- Damping and mass remain target- and frequency-restricted diagnostics; no
  fixed causal estimator passes the complete target vector.
- The unilateral contact abstraction still lacks gap, impact, probe-body, and
  actuator/contact dynamics.
- The current covariance describes sampling spread better than systematic
  causal/EIV bias; it is not a valid active value-of-information signal.
- Reference runs record a dirty worktree and need a repository commit for
  commit-level provenance.

## Next Stage 3A question

Can a prevalidated hybrid normal-force/contact-state controller enforce bounded
realized probe force, deliberate unloading, and stable sustained-contact
settling before any safety-policy redesign? The next experiment must separate
probe-delivery failure from future-maneuver control failure and first replay
the frozen policy unchanged.

## Scope exclusions

No Gazebo, PX4, ROS 2, MPC, RL, Isaac Sim, perception, hardware, flexible probe,
or nonlinear/multimode target result has been started or claimed.

Stage 1/2 remain frozen at `95fcdbb5d08f2ffa46077aced7f5c882fb2627f8`
(`stage2-complete-20260830`). EXP-0008 is new Stage 3A evidence and does not
rewrite that reduced-order result.

## Last updated

2026-08-30T16:05:00+03:00
EXP-0008 completed the first coupled 6-DoF baseline and identified a failed
policy transfer caused by force-delivery/re-contact and controller-induced
sustained settling. Continue Stage 3A; do not begin Stage 3B.
