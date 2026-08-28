# Project state

## Current hypothesis

A UAV may be able to use a deliberately small, bounded physical probe to infer
a probabilistic local interaction model, assess whether that model is
sufficiently informative, and use the remaining uncertainty to choose a safe
inspection action: inspect, reduce force, re-probe, or abort. No individual
estimator, controller, or probing method is assumed novel.

## Current milestone

Stage A — reduced-order interaction-model validation.

EXP-0001 completed the Kelvin–Voigt algebraic evaluator sub-milestone. Its
predeclared acceptance criteria were force RMSE and maximum force error no more
than `1e-12 N`, relative energy-balance error no more than `1e-5`, displacement
overshoot no more than `1e-12 m`, and zero configured safety-limit violations.
All criteria passed.

Stage A as a whole is not complete. The mass–spring–damper model
`F = m_eff*x_ddot + c*x_dot + k*x` still needs independent analytical and
numerical validation with quantitative acceptance criteria.

## Active questions

- Does a numerically integrated mass–spring–damper target reproduce analytical
  free and forced responses over the intended parameter/sample-time range?
- Which bounded signals keep force, displacement, velocity, and energy within
  limits while making `k`, `c`, and later `m_eff` observable?
- How should unilateral contact loss be represented without misusing the
  bilateral energy identity?
- Are displacement and velocity measurements realistic enough for the planned
  identification study, or must sensor dynamics/bias be included earlier?

## Current blockers

- `/home/jimmy/Desktop/Drones/Probeing` is not the Git top level; Git resolves
  to `/home/jimmy`, which has no commit. Run manifests therefore cannot record
  a project commit SHA.
- The workstation has SciPy 1.8.0 with NumPy 1.26.4; SciPy reports this NumPy
  version is unsupported. Current code avoids SciPy, but later numerical work
  needs an isolated, compatible environment.
- `InitialPlan.md` contains internal citation tokens rather than resolvable
  bibliography entries. Its literature claims must be source-audited before
  they are used in comparisons or public scientific writing.

## Strongest result so far

Reference run
`EXP-0001_20260828T170017.246126Z_s1101_1bc4b363` passed every configured
acceptance check for a 5 mm raised-cosine ramp on `k = 500 N/m`,
`c = 20 N·s/m`: zero force error against the analytical constitutive equation,
relative energy-balance error `7.48585e-7`, peak force `2.50983 N`, and no
safety-limit violations. This is evidence about the reduced evaluator only.

## Weakest assumption

Prescribed indentation assumes perfect knowledge and enforcement of target
displacement and velocity. It removes the UAV–target coupling that is central
to safe aerial interaction, so success in EXP-0001 does not imply that the
parameters will be observable or that a real probe will be safe.

## Next experiment

Define EXP-0002 for the mass–spring–damper target. Validate underdamped,
critically damped, and overdamped free responses against analytical solutions,
then verify convergence under a bounded forced input. Predeclare error,
convergence-order, and safety acceptance thresholds before running it.

## Last updated

2026-08-28T20:05:44+03:00
