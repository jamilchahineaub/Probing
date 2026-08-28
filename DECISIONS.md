# Decision ledger

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

