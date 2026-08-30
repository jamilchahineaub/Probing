# Decision ledger

## DEC-0023 — Freeze the passed Stage 1/2 reduced-order baseline

- Date: 2026-08-30.
- Options: begin vehicle dynamics in the uncommitted working tree, or first
  establish an immutable version-control checkpoint for all completed Stage 1
  and Stage 2 evidence.
- Evidence: all seven Stage 1 artifact fingerprints matched their recorded
  values, the Python regression suite passed `81/81`, and the passing Stage 2
  run `STAGE2_20260829T221139.099Z_s20101` retained its MATLAB/Simulink
  agreement and zero probe-safety violations.
- Decision: freeze the completed baseline at Git commit
  `95fcdbb5d08f2ffa46077aced7f5c882fb2627f8` and annotated tag
  `stage2-complete-20260830`. EXP-0001 through EXP-0007 and the completed Stage
  2 run are read-only scientific references for Stage 3A.
- Why: coupled-vehicle results must be attributable to the new dynamics rather
  than silent changes in the validated reduced-order implementation.
- Revisitable: no. Corrections, if ever required, must be new, explicitly
  versioned experiments and may not rewrite the frozen artifacts.

## DEC-0001 — Milestones govern progression

- Date: 2026-08-28.
- Options: follow the date-based schedule in `InitialPlan.md`, or use explicit
  evidence gates.
- Evidence: the current project directive explicitly supersedes deadline-driven
  progression and requires quantitative acceptance criteria.
- Decision: use experiment/milestone acceptance criteria and record no delivery
  dates.
- Why: scientific stages should advance only when their claims are supported.
- Revisitable: no, unless the project directive changes; criteria themselves can
  be revised prospectively with documented evidence.

## DEC-0002 — Use a namespaced `src` package

- Date: 2026-08-28.
- Options: top-level generic packages such as `models`, or
  `src/probeing/{models,estimators,controllers,probing,metrics,plotting}`.
- Evidence: the repository started empty, and generic package names are prone to
  import collisions.
- Decision: use the `probeing` namespace while preserving the research-plan
  module boundaries.
- Revisitable: yes, before external APIs depend on it.

## DEC-0003 — First probe is a raised-cosine ramp

- Date: 2026-08-28.
- Options: discontinuous step, linear ramp, half-sine pulse, or smooth bounded
  ramp.
- Evidence: a raised-cosine ramp has analytical position/velocity/acceleration,
  zero endpoint velocity, and no ideal step impulse. It is a bounded version of
  the Stage B step/ramp family.
- Decision: use a 5 mm raised-cosine ramp for analytical model validation.
- Revisitable: yes. Probe families must be compared later; this is not a claim
  that the ramp is information-optimal.

## DEC-0004 — Separate bilateral validation from unilateral contact

- Date: 2026-08-28.
- Options: silently clip tensile force, or expose contact mode explicitly.
- Evidence: `F = k*x + c*x_dot` has a direct mechanical-energy identity in
  bilateral form, while unilateral clipping introduces contact loss and changes
  the work accounting.
- Decision: validate EXP-0001 in explicit `bilateral` mode; implement but do not
  experiment-validate `unilateral` mode yet.
- Revisitable: the separation is retained, but later experiments may make
  unilateral contact the default physical mode.

## DEC-0005 — Immutable run artifacts and seed partitions

- Date: 2026-08-28.
- Options: overwrite a latest-results directory, or create a unique run for each
  execution.
- Evidence: the scientific-integrity requirements forbid overwriting and tuning
  on held-out cases.
- Decision: unique timestamp/seed/random-suffix run IDs, exclusive file creation,
  development seeds 1000–1999, validation 2000–2999, final test 3000–3999.
- Revisitable: seed ranges may expand prospectively; existing assignments do not
  change.

## DEC-0006 — Avoid SciPy in the initial implementation

- Date: 2026-08-28.
- Options: use the installed SciPy or implement the small required quadrature
  directly with NumPy.
- Evidence: SciPy 1.8.0 warns that installed NumPy 1.26.4 is unsupported. EXP-0001
  needs only cumulative trapezoidal integration.
- Decision: use a tested NumPy trapezoidal implementation and record both package
  versions in manifests.
- Revisitable: yes, after creating a compatible isolated environment for later
  ODE and estimator work.

## DEC-0007 — Treat the Milestone A probe as an applied force

- Date: 2026-08-28.
- Options: prescribe target displacement and evaluate reaction force, or apply
  a bounded force and integrate the target response.
- Evidence: estimating `m_eff` requires target acceleration to arise from the
  second-order dynamics; prescribed kinematics would not test that interaction.
- Decision: integrate the bilateral force-driven equation
  `m_eff*x_ddot + c*x_dot + k*x = F(t)` with RK4.
- Revisitable: the force-driven interpretation is fixed for EXP-0001. A later
  Milestone A case may add unilateral contact without rewriting this result.

## DEC-0008 — Diagnose excitation before changing the estimator

- Date: 2026-08-28.
- Options: tune RLS until every probe appears accurate, or freeze transparent
  LS/RLS baselines and report rank, conditioning, and parameter correlation.
- Evidence: EXP-0001 contains full-rank cases with correlation up to `0.985`
  and a rigid/ramp effective-mass error of `-62.68%` under near-ideal noise.
- Decision: preserve those results, report probe-dependent failures, and study
  observability/errors-in-variables effects before adding estimator complexity.
- Revisitable: no for EXP-0001; later estimators must be separate comparisons.

## DEC-0009 — Stage 1 continues after EXP-0002

- Date: 2026-08-29.
- Options: advance to MATLAB/Simulink or UAV dynamics because median cases look
  usable, or enforce the predeclared cross-target tail/bias gate.
- Evidence: EXP-0002 completed 19,800 validation trials. No candidate passed.
  The best chirp/Savitzky–Golay candidate had worst-target p95 errors of
  `19.35%`, `32.98%`, and `62.58%` for `k`, `c`, and `m_eff`, with biases of
  `14.14%`, `28.01%`, and `36.75%`.
- Decision: `CONTINUE_STAGE_1`. Do not advance to MATLAB, EKF, vehicle dynamics,
  or integrated simulation.
- Why: full rank and acceptable median behavior did not produce reliable
  target-wise tails, particularly for damping and effective mass.
- Revisitable: yes, only after a new Stage 1 experiment passes an explicit
  practical sensing gate.

## DEC-0010 — Keep effective mass provisional and retain modal diagnostics

- Date: 2026-08-29.
- Options: remove `m_eff`, keep claiming it is identified, or retain it as a
  provisional parameter while testing whether modal ratios are more robust.
- Evidence: best-candidate mass p95 error was `62.58%`; the alternate natural
  frequency/damping-ratio p95 errors were `36.70%`/`64.90%` and did not rescue
  all target classes.
- Decision: keep `m_eff` in the research model but do not treat it as reliably
  identified. Continue reporting natural frequency and damping ratio alongside
  it.
- Revisitable: yes, after isolating derivative noise and timing misalignment.

## DEC-0011 — Savitzky–Golay is an offline diagnostic baseline, not deployment proof

- Date: 2026-08-29.
- Options: call the best pipeline practical/deployable, or distinguish its
  offline performance from causal operation.
- Evidence: the centered local-polynomial window uses future samples and has an
  approximately half-window realization delay. It was the best candidate but
  still failed GO.
- Decision: use it as an offline upper-bound diagnostic. The next sensing study
  must include common causal filtering and explicit time alignment.
- Revisitable: no for interpretation of EXP-0002.

## DEC-0012 — Stage 1 continues after EXP-0003

- Date: 2026-08-29.
- Options: advance because each physical parameter has at least one useful
  restricted condition, or enforce the requirement that one causal candidate
  identify the task-relevant vector across all targets.
- Evidence: EXP-0003 completed 17,880 held-out evaluations with zero safety
  events. `k` classified A, while `c` and `m_eff` classified B under different
  pipeline/estimator conditions. No fixed causal candidate passed the full
  vector gate.
- Decision: `CONTINUE_STAGE_1`. Independent MATLAB/Simulink validation is not
  authorized by this evidence.
- Why: combining parameter-specific winners would conceal that they rely on
  different bands/estimators and that some full-vector estimates are
  nonphysical.
- Revisitable: yes, after one globally fixed causal estimator returns a
  physical vector under the cross-target gate, or after controller requirements
  prospectively remove a parameter from the task-relevant vector.

## DEC-0013 — Retain direct parameters; do not adopt TLS or ratio regression

- Date: 2026-08-29.
- Options: replace OLS with standardized TLS, switch to
  `[omega_n, zeta, 1/m_eff]`, or retain direct parameters with conditional IV
  diagnostics.
- Evidence: TLS worsened causal damping tails and did not improve median mass
  p95. Ratio-regression damping ratio and inverse mass were category C. IV
  reduced 0.5–5 Hz causal-low-pass mass worst p95 from `108.40%` to `34.62%`
  but failed generally when instruments were weak.
- Decision: retain `[k, c, m_eff]` and OLS as the transparent general baseline.
  Keep IV as a restricted mass diagnostic. Do not adopt TLS or the ratio form.
- Revisitable: yes, if a defensible correlated-error covariance supports
  weighted/structured TLS or real contact data validates IV instrument strength.

## DEC-0014 — Treat relative synchronization below 5 ms as a model requirement

- Date: 2026-08-29.
- Options: treat current sensor timestamps as approximately aligned, or impose
  an explicit synchronization budget before interpreting damping/mass.
- Evidence: the worst lightly damped case crossed the damping-bias threshold at
  the first tested 1 ms displacement/acceleration offset. Force timestamp and
  kinematic filter group delay crossed damping/mass thresholds at 5 ms.
- Decision: future Stage 1 sensing experiments must represent relative channel
  timing explicitly and target sub-millisecond knowledge for lightly damped
  cases; an uncompensated 5 ms relative delay is not acceptable for general
  damping/mass claims.
- Revisitable: thresholds can be refined with denser timing sweeps, but the need
  to model timing is fixed by EXP-0003.

## DEC-0015 — Stop active full-vector identification with the current architecture

- Date: 2026-08-29.
- Options: continue tuning the uncertainty-driven sequence, advance with the
  best fixed probe, or stop active `[k, c, m_eff]` identification and test
  decision-sufficient dynamic outputs.
- Evidence: EXP-0004 used twenty untouched seeds and a 1 N candidate library.
  Adaptive probing achieved `65.00%` full-vector success versus `80.83%` for a
  fixed two-stage comparator within `5.98%` command dose. Its worst-target p95
  RMS error was `33.70%` versus `25.67%`; predicted and realized follow-up
  utility had `rho = -0.478`. The single fixed chirp was stronger at `90.83%`
  success but still failed rigid damping and high-mass tail gates.
- Decision: enforce the EXP-0004 kill criterion. Do not keep tuning active
  probe selection to recover all three physical parameters in this sensing and
  model architecture. Do not advance to MATLAB/Simulink.
- Why: the selector treated systematic causal/EIV bias as reducible variance;
  more nominal information often amplified bias. Continuing to tune after the
  held-out failure would invalidate the scientific test.
- Next evidence: test whether stiffness plus dissipation/resonance bounds are
  sufficient for the later safety decisions using one fixed low–mid chirp.
- Revisitable: only if a new measurement model, bias-aware uncertainty method,
  or independently justified controller requirement materially changes the
  architecture; not by retuning this probe library on EXP-0004 seeds.

## DEC-0016 — Do not advance after the fixed-chirp safety test

- Date: 2026-08-29.
- Options: advance because nominal decision accuracy exceeded 85%, retune the
  classifier on held-out failures, or enforce the EXP-0005 kill criterion and
  isolate the observed slow-settling blind spot.
- Evidence: the combined task-feature predictor achieved `86.25%` nominal
  accuracy and `3.90%` nominal false-safe rate, but high-noise false-safe rate
  was `6.49%` against a predeclared `5%` limit. Unsafe recall was `84.31%`, and
  safe-force lower-bound coverage was `93.75%` nominally. All nominal
  false-safe cases violated only the future settling limit; their predicted
  settling upper bounds were `1.538–1.956 s` versus `2.932–3.000 s` actual.
- Decision: enforce `RECONSIDER_PROBING_ARCHITECTURE` and
  `CONTINUE_STAGE_1`. Do not retune on seeds 6101–6120 and do not start
  MATLAB/Simulink or vehicle work.
- Parameter implication: accurate separate mass was not necessary for many
  decisions—the worst mass-error quartile retained `80.00%` accuracy with no
  nominal false-safe—but the full-parameter feature set had zero nominal
  false-safes. Keep physical parameters as diagnostics; do not make complete-
  vector recovery the objective and do not claim the vector is unnecessary.
- Next evidence: add a fixed zero-force observation window after the same chirp
  and prospectively test settling/ringdown bounds without adding probe energy.
- Revisitable: only with new untouched seeds and an architecture change aimed
  at the settling failure, not with threshold changes on EXP-0005 validation.

## DEC-0017 — Retain passive observation but do not advance Stage 1 yet

- Date: 2026-08-29.
- Options: advance because the false-safe criterion passed, discard passive
  observation because overall accuracy fell, retune the three-class policy on
  the held-out miss, or retain the locked passive architecture and enforce the
  complete predeclared gate.
- Evidence: on 240 untouched EXP-0006 targets, 0.5 s of zero-force observation
  reduced nominal/high false-safe rates from `2.44%/3.66%` to zero without
  changing the `0.206124 N²s` probe dose. Accuracy remained above 80%, but
  high-noise UNSAFE recall was `43/48 = 89.58%`, one case below the 90% gate.
  The miss was UNSAFE-to-CAUTION, not false-safe. A 3 s fixed window met the
  nominal/high classification thresholds but exceeded the 2 s duration limit.
- Decision: retain chirp plus passive ring-down as the strongest current safety
  architecture, but keep `CONTINUE_STAGE_1`. Do not retune on seeds 8101–8120
  and do not start MATLAB/Simulink or vehicle work.
- Parameter implication: adding separate `c` and `m_eff` diagnostics changed
  nominal accuracy by only 0.42 point and did not improve the zero false-safe
  result. Keep physical coefficients as mechanistic diagnostics, but do not
  require complete-vector recovery for the SAFE/non-SAFE decision.
- Next evidence: independently replicate the fully locked 0.5 s policy with a
  prospectively hierarchical endpoint—SAFE versus non-SAFE for the primary
  actuation decision and CAUTION versus UNSAFE for secondary severity—while
  preserving the existing physical class definitions.
- Revisitable: only with untouched seeds or independent implementation; not by
  post-hoc relaxation of the EXP-0006 duration or UNSAFE-recall criterion.
## DEC-0018 — Proceed after EXP-0007 locked replication

The frozen EXP-0006 binary SAFE/NON-SAFE policy replicated on independent broad and boundary-enriched targets with nominal/high-noise false-safe rates below the predeclared limits. Proceed only to independent MATLAB/Simulink validation.

## DEC-0019 — Pass independent MATLAB/Simulink validation

- Date: 2026-08-30.
- Evidence: 1,360 independently generated MATLAB targets, zero false-safe
  cases in low/nominal/high noise, 0.586% one-sided 95% upper bound, nominal
  and high accuracy 88.38%/87.65%, zero probe safety events, and floating-point
  MATLAB/Simulink/Python matched-trajectory agreement.
- Decision: Stage 2 passes. The result is not a Python-specific integration
  artifact; accurate damping and effective-mass recovery remain unnecessary
  for the locked binary decision.
- Authorization: Stage 3 may be planned, but no 6-DoF or integrated vehicle
  work was started in this experiment.
