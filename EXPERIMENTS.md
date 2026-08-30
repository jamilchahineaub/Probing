# Experiment index

## EXP-0001 — Milestone A bounded interaction identification

### Scientific question

> Can `k`, `c`, and `m_eff` of an unknown interaction target be identified from
> a small bounded probing interaction under ideal or near-ideal measurements?

### Design

- Model: force-driven bilateral one-dimensional target,
  `F = m_eff*x_ddot + c*x_dot + k*x`, integrated with fixed-step RK4.
- Configuration:
  [exp_0001_interaction_identification.yaml](configs/experiments/exp_0001_interaction_identification.yaml).
- Targets: rigid/high-stiffness, compliant, lightly damped flexible,
  high-damping, low-effective-mass, and high-effective-mass.
- Probes: bounded 1 N ramp, half-sine pulse, sinusoid, linear chirp, and
  multisine. These signals were compared as supplied; none was optimized.
- Measurements: perfect and seeded near-ideal Gaussian-noise displacement,
  velocity, acceleration, and contact force.
- Estimators: unregularized batch least squares, followed by RLS with unit
  forgetting factor and RMS feature preconditioning.
- Matrix: 6 targets × 5 probes × 2 measurement scenarios = 60 cases and 90,060
  samples. Development seed 1101; each case has a deterministic derived seed.
- Diagnostics: normalized regression condition number, numerical rank, and
  inferred parameter correlation are saved for every case.

### Predeclared acceptance

- All perfect-measurement regressions have rank 3.
- Worst normalized condition number is no greater than 1000.
- Maximum perfect-measurement RMS relative error is no greater than `1e-9`
  for batch LS and `1e-6` for RLS.
- Median near-ideal RMS relative error is no greater than 15% for each
  estimator.
- Probe force and simulated displacement, velocity, and acceleration remain
  inside the configured limits.

### Reference run

- Run ID: `EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69`
- Status: success; all seven acceptance checks passed.
- [Manifest](runs/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69/manifest.json)
- [Resolved configuration](runs/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69/config.resolved.yaml)
- [Case metrics](results/tables/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__case_metrics.csv)
- [Summary metrics](results/tables/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__metrics.csv)
- Raw data:
  [compressed CSV](runs/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69/raw_timeseries.csv.gz),
  [compressed NumPy archive](runs/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69/raw_timeseries.npz).

### Results

- All 30 perfect-measurement cases were full rank. Maximum batch and RLS RMS
  relative errors were `2.23e-13` and `6.50e-12`, respectively.
- Near-ideal median RMS relative error was `0.488%` for both batch LS and RLS.
  Their final estimates are nearly equal because RLS uses the same stationary
  linear model, all samples, and a forgetting factor of one.
- The result is strongly probe-dependent:

| Probe | Mean near-ideal RMS parameter error |
|---|---:|
| Ramp | 13.082% |
| Half-sine | 3.849% |
| Sinusoid | 0.698% |
| Chirp | 0.136% |
| Multisine | 0.272% |

- The worst near-ideal case was the rigid/high-stiffness target under the ramp:
  `36.46%` RMS parameter error. Its effective mass was underestimated by
  `62.68%`, while stiffness error was only `-0.0037%` and damping error was
  `-7.74%`.
- Full rank did not imply robust separation. The worst perfect-data parameter
  correlation magnitude was `0.9850` for the compliant target under a single
  sinusoid; its normalized condition number was `11.52`.
- Peak probe force was `1.0 N`, peak target displacement `16.20 mm`, peak
  velocity `0.2967 m/s`, and peak acceleration `12.04 m/s²`. No configured
  safety event occurred.

### Figures

- [Force, displacement, and all probe signals](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__timeseries.png)
- [True versus estimated k, c, and m_eff](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__parameter_comparison.png)
- [RLS parameter error versus time](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__rls_error.png)
- [Identification error by probing signal](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__probe_error.png)
- [Conditioning and parameter correlation](results/figures/EXP-0001_20260828T204945.797835Z_s1101_4eb1fe69__observability.png)

### Conclusion

The three parameters are structurally recoverable in this finite-window,
force-driven linear model under perfect measurements because every configured
response regression has rank 3. Under the configured near-ideal direct sensor
noise, chirp and multisine probes remain accurate across this target set, but a
ramp does not reliably identify effective mass. Therefore EXP-0001 supports a
conditional “yes,” not a claim that every small bounded interaction is
informative.

The direct acceleration measurement, independent Gaussian noise, known applied
force, bilateral signed interaction, exact linear model, and development-only
seed are important limitations. The perfect-data force equation also shares
the same model used to generate acceleration; independent analytical
free/forced-response tests provide the numerical-integrator check.

### Next work within Milestone A

Do not add EKF or UAV dynamics. First repeat the matrix on untouched validation
seeds, sweep measurement noise and sample period, examine errors-in-variables
bias from noisy regressors, and determine whether acceleration can be estimated
without destroying mass identifiability. Preserve the weak ramp result as a
baseline; do not tune RLS to conceal it.

### Historical pilot

The earlier run `EXP-0001_20260828T170017.246126Z_s1101_1bc4b363` validated
only the algebraic Kelvin–Voigt relation under prescribed displacement. It is
retained for provenance but does not answer the EXP-0001 identification
question.

## EXP-0002 — Realistic sensing and practical identifiability stress test

### Scientific question

> Does identification of `k`, `c`, and `m_eff` remain reliable when the
> measurements and contact assumptions become closer to what an actual aerial
> robot could obtain?

### Design

- Authoritative configuration:
  [exp_0002_practical_identifiability.yaml](configs/experiments/exp_0002_practical_identifiability.yaml).
- Frozen EXP-0001 artifact fingerprint before and after the run:
  `fc8d3375b320d6fa81843a5dc8819f17de7ea36c50f1a258bc6e0578f88aba39`.
- Validation seeds: 2101–2110. No EXP-0001 development seed was reused.
- Matrix: six targets, five unchanged probes, bilateral/unilateral contact,
  three coupled imperfection severities, eleven sensing/pipeline combinations,
  and ten Monte Carlo seeds: 19,800 unique trials.
- Sensing regimes: direct reference; numerically derived velocity; numerically
  derived velocity and acceleration; lower-rate position plus IMU-like
  acceleration and complementary velocity; exploratory command-based force
  proxy.
- Common derivative pipelines: finite differences, causal first-order low-pass
  filtering, and NumPy-only Savitzky–Golay local polynomials. Settings were not
  tuned by target or probe.
- Imperfections jointly swept displacement/acceleration/force noise, sample
  rate, position rate, sensor latency, channel timestamp mismatch, and filtering
  delay at mild, nominal, and severe levels.
- Metrics include per-parameter bias/RMSE/95th-percentile/worst error, RLS
  convergence, singular values, rank, normalized condition, correlation,
  errors-in-variables shift, force/impulse/work/energy, target motion, and
  heuristic information/disturbance ratios.

The unilateral Stage 1 model rejects tensile commanded force and records those
intervals as contact loss. It is a no-tension abstraction, not yet probe-body
gap or impact dynamics.

### Predeclared GO criteria

GO required one nominal, unilateral, non-sensorless pipeline with no direct
acceleration to pass every target's 95th-percentile error, bias, rank, force,
and displacement limits. Required 95th-percentile limits were 20% for `k`, 30%
for `c`, and 30% for `m_eff`; bias limits were 10%, 15%, and 15%. At least 99%
of trials had to be full rank.

### Reference run

- Run ID: `EXP-0002_20260828T212540.955871Z_s2101_e8713743`
- Execution status: success; all eight integrity checks passed.
- Stage 1 decision: **CONTINUE_STAGE_1**; 0 of 30 candidates passed GO.
- [Manifest](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/manifest.json)
- [Resolved configuration](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/config.resolved.yaml)
- [Raw Monte Carlo trials](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/monte_carlo_trials.csv.gz)
- [Aggregate distributions](results/tables/EXP-0002_20260828T212540.955871Z_s2101_e8713743__aggregate_metrics.csv)
- [GO/NO-GO candidates](results/tables/EXP-0002_20260828T212540.955871Z_s2101_e8713743__go_no_go_candidates.csv)
- [Decision record](runs/EXP-0002_20260828T212540.955871Z_s2101_e8713743/stage1_decision.json)

### What remained identifiable

The best practical candidate was the unilateral chirp with position-derived
velocity/acceleration using Savitzky–Golay processing. Across every target it
remained full rank. Its worst-target 95th-percentile stiffness error was
`19.35%`, narrowly inside the tail threshold, but stiffness bias reached
`14.14%`, above the 10% limit.

Stiffness was the most robust original parameter. For this candidate its
target-wise 95th-percentile error ranged from `2.21%` for the compliant target
to `19.35%` for the rigid/high-stiffness target.

### What became difficult

- Damping: worst-target 95th-percentile error `32.98%` and bias `28.01%`, led
  by the lightly damped target.
- Effective mass: worst-target 95th-percentile error `62.58%` and bias `36.75%`,
  led by the low-effective-mass target. This is the clearest remaining failure.
- RLS did not remove the batch-regression bias. Only 53.3% of best-candidate
  trials met the strict sustained 20% convergence condition; the rigid and
  low-mass targets had zero qualifying runs.
- The alternate acceleration-form parameterization did not rescue the full
  target set: worst-target 95th-percentile errors were `36.70%` for natural
  frequency and `64.90%` for damping ratio.
- The exploratory sensorless-force proxy failed: median RMS parameter error was
  `82.85%`, and its 95th percentile was `305.65%`. No sensorless success is
  claimed.

All regressions being full rank while parameter tails and biases fail confirms
that normalized rank/conditioning alone are inadequate under noisy regressors,
channel delay, and timestamp mismatch.

### Probe and pipeline comparison

Using the RMS of each parameter's worst-target 95th-percentile error as a
ranking score, the best practical pipeline for each probe gave:

| Probe | Best score | Pipeline |
|---|---:|---|
| Chirp | 42.34% | no direct acceleration + Savitzky–Golay |
| Sinusoid | 50.88% | no direct acceleration + Savitzky–Golay |
| Half-sine | 87.95% | no direct acceleration + Savitzky–Golay |
| Multisine | 98.33% | no direct acceleration + Savitzky–Golay |
| Ramp | 175.23% | no direct acceleration + finite differences |

The same best chirp pipeline had median/95th-percentile RMS errors of
`13.66%`/`27.95%` at nominal severity and `41.51%`/`79.11%` at severe severity.
For that candidate, unilateral and bilateral nominal 95th-percentile errors
were `27.95%` and `31.36%`; the simplified unilateral clamp was not the primary
failure source.

The direct-channel “optimistic” regime was not best after imperfections were
applied. Independent latency and timestamp offsets made directly measured
`x`, `x_dot`, `x_ddot`, and force mutually inconsistent. Deriving all
kinematics from one position stream and filtering the force coherently reduced
that inconsistency, although the centered Savitzky–Golay implementation is an
offline diagnostic with approximately half-window future-data delay, not yet a
deployment-ready causal estimator.

### EXP-0001 ramp failure explanation

The recorded rigid/ramp force contained `95.75%` DC spectral energy; only
`0.148%` was at or above the target natural frequency. Acceleration RMS was
`0.00635 m/s²`, versus `0.9745 m/s²` for chirp. Elastic force accounted for
`99.83%` of force RMS, while damping and inertial contributions were only
`0.784%` and `0.696%`.

Consequently the acceleration column was nonzero and the normalized design
matrix looked benign—condition `1.30`, maximum correlation `0.249`—but its
absolute signal-to-noise ratio was poor. Column normalization concealed this
weak physical scale. In errors-in-variables least squares, noise in a weak
acceleration regressor attenuates its coefficient toward zero, explaining the
`62.68%` effective-mass underestimate while stiffness remained accurate. Chirp
provided `29.89%` of force spectral energy above the natural frequency and an
inertial contribution `1.388` times force RMS, making mass much less fragile in
EXP-0001.

### Figures

- [Error vs noise and sample rate](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__noise_and_sampling.png)
- [Pipeline and separate parameter errors](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__pipeline_and_parameter_errors.png)
- [Singular values, condition, and correlation](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__observability_by_probe.png)
- [Representative clean/noisy/filtered signals](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__representative_signals.png)
- [Information vs disturbance](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__information_disturbance.png)
- [Bilateral vs unilateral performance](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__contact_mode_comparison.png)
- [Failure heatmap](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__failure_heatmap.png)
- [Errors-in-variables shifts](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__errors_in_variables.png)
- [Ramp failure analysis](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__ramp_failure_analysis.png)
- [Exploratory sensorless-force failure](results/figures/EXP-0002_20260828T212540.955871Z_s2101_e8713743__sensorless_force_exploratory.png)

### Conclusion and next experiment

EXP-0002 does not justify progression to MATLAB/Simulink or a UAV stack.
Stiffness remains useful for many cases, but no tested practical pipeline met
the cross-target damping and effective-mass reliability gate. Effective mass
must remain provisional rather than being silently dropped or reported as
identified.

The single most useful next experiment is a Stage 1 synchronized-chirp EIV
isolation study: hold the chirp/targets/contact model fixed; vary noise, sample
rate, latency, and timestamp offset one factor at a time; apply a common causal
filter and explicit time alignment to position and force; and compare ordinary
LS with a transparent total-least-squares or instrumental-variable baseline.
Its purpose is to determine whether mass failure is mainly derivative noise,
timing misalignment, or the parameterization itself. Do not advance the roadmap
stage before that result.

## EXP-0003 — Causal timing and errors-in-variables identifiability

### Scientific question

> Are the poor damping and effective-mass estimates caused primarily by
> timing/filtering/errors-in-variables effects that can be corrected causally,
> or are they fundamentally too weakly identifiable under safe bounded probing?

### Frozen design

- Authoritative configuration:
  [exp_0003_causal_eiv_identifiability.yaml](configs/experiments/exp_0003_causal_eiv_identifiability.yaml).
- EXP-0001 and EXP-0002 were read-only. Their artifact fingerprints remained
  `fc8d3375…ba39` and `f01de136…2efc` before and after this run.
- Twenty untouched held-out seeds, 3101–3120, were used only after development
  runs on seeds 1601–1610.
- The force was a unilateral 1 N linear chirp. Six fixed frequency bands from
  0.25–2 Hz through 0.5–20 Hz were compared without increasing amplitude.
- Six unchanged targets, four causal processing pipelines, the centered
  Savitzky–Golay offline reference, and OLS/TLS/IV were evaluated.
- All estimators used exactly the same held-out measurement realization.
- Weighted TLS was not used because position-derived velocity and acceleration
  errors are colored, correlated, and target-dependent; arbitrary diagonal
  weights would not constitute a defensible covariance model.
- Input-history IV used five delayed copies of the known bounded command as
  instruments. This addresses kinematic regressor noise but assumes the Stage 1
  command/contact activation model remains valid.

The run contains 4,680 one-factor timing trials, 2,400 combined timing-profile
trials, and 10,800 pipeline/frequency/estimator identification trials. All
frequency bands stayed inside the predeclared force, displacement, velocity,
and acceleration limits.

### Reference run

- Run ID: `EXP-0003_20260828T225126.077199Z_s3101_b2da5c53`
- Execution: success; all eleven integrity checks passed.
- Stage 1 decision: **CONTINUE_STAGE_1**; no single causal candidate passed the
  full-vector gate.
- [Manifest](runs/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53/manifest.json)
- [Held-out identification trials](runs/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53/identification_trials.csv.gz)
- [Aggregate distributions](results/tables/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__identification_aggregate.csv)
- [Parameter classifications](results/tables/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__parameter_classifications.csv)
- [Timing thresholds](results/tables/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__timing_thresholds.csv)
- [Decision](runs/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53/stage1_decision.json)

### Timing attribution

The one-factor direct-channel sweep demonstrated that timing can matter at
sub-sample scales. The first tested absolute offsets that exceeded the
predeclared incremental-bias thresholds in at least one target were:

| Timing error | `k` | `c` | `m_eff` |
|---|---:|---:|---:|
| Displacement timestamp | 20 ms | 1 ms | 5 ms |
| Acceleration timestamp | 20 ms | 1 ms | 20 ms |
| Force timestamp | 20 ms | 5 ms | 5 ms |
| Kinematic filter group delay | 20 ms | 5 ms | 5 ms |

The 1 ms damping sensitivity was driven by the lightly damped target: a 1 ms
displacement shift changed its damping bias by 57.57 percentage points, and a
-1 ms acceleration shift changed it by 62.92 points. Five milliseconds of
relative force or kinematic-filter delay also produced practically significant
damping and mass bias.

Timing is therefore a meaningful error source, especially for damping, but it
does not explain most of the causal derivative failure. For the common causal
low-pass pipeline, synchronized versus EXP-0002-like timing changed median
absolute errors only from `39.89%/19.29%/85.23%` to
`38.94%/18.59%/85.69%` for `k/c/m_eff`. Correcting all channels to a common
delay did not remove the mass error. In the centered offline pipeline,
EXP-0002-like mismatch increased damping error from `3.60%` to `8.60%`, while
apparently reducing mass error from `24.06%` to `14.48%`; that reduction is
phase-error cancellation, not evidence that mismatch is beneficial.

### Causal processing

At the baseline 0.5–10 Hz chirp, measured acceleration delay/noise attenuation
were approximately:

| Pipeline | Causal | Effective delay | Noise attenuation | Cost units/sample |
|---|---|---:|---:|---:|
| Backward difference | yes | 5 ms | 0 dB | 8 |
| Causal low-pass + derivative | yes | 15 ms | 16.10 dB | 20 |
| Trailing polynomial | yes | 30 ms | 31.85 dB | 92 |
| Alpha-beta-gamma | yes | 40 ms | 39.31 dB | 24 |
| Centered Savitzky–Golay | no | 0 ms realized phase delay, future samples required | 47.07 dB | 92 |

Causal low-pass was the best overall causal compromise. No causal pipeline
approached the centered reference jointly across all parameters. More
aggressive causal smoothing reduced noise but introduced enough phase distortion
to make damping worse. The centered reference at 0.25–2 Hz reached worst-target
p95 errors of `4.86%`, `24.51%`, and `28.89%` for `k`, `c`, and `m_eff`, but
its mass bias still reached `22.22%` and it remains noncausal.

### EIV estimator findings

TLS did not solve the practical EIV problem. Across all causal target/band
cases, its median target-wise p95 damping error was `217.49%`, compared with
`86.86%` for OLS, and its mass median was essentially unchanged (`82.15%`
versus `82.28%`). Standardized TLS treats perturbations as isotropic after
scaling; the actual derivative errors violate that assumption.

IV produced a real but restricted mass improvement. With the 0.5–5 Hz chirp
and causal low-pass pipeline, worst-target mass p95/bias fell from
`108.40%/107.75%` under OLS to `34.62%/15.12%` under IV. Five of six targets
passed the parameter-specific mass limits. The high-mass target remained just
outside both limits and had the weakest instrument strength (`0.0123`). IV
also produced very poor damping on the lightly damped target (`705.12%` p95)
and was worse than OLS in the cross-case combined score. It is not adopted as
a general solution.

All batch solvers completed numerically. Nevertheless, nonphysical full-vector
estimates occurred in approximately 21.9% of causal OLS groups, 22.4% of TLS
groups, and 32.5% of IV groups on average. Numerical completion is not the same
as physical convergence.

### Regressor geometry and chirp frequency

Normalized singular values and condition numbers stayed deceptively benign:
the baseline causal-low-pass target conditions ranged only from `1.21` to
`2.52`. Effective-mass error still tracked acceleration strength moderately
(`Spearman rho = -0.480`). The high-mass target had only
`0.136 m/s²` acceleration RMS and about 99% OLS mass error, whereas the
low-mass target had `2.663 m/s²` and 13.68% error.

The 0.25–2 Hz band contained the strongest stiffness information and enabled
the best stiffness result. Damping information increased into the 0.5–10 Hz
range, but causal filter phase error made bands above 5 Hz harmful to damping.
Inertial regression information increased with bandwidth, yet mass accuracy did
not improve monotonically because derivative phase/noise and IV instrument
strength dominated the nominal Fisher-information gain. The 0.5–5 Hz band was
the best practical compromise for damping and IV mass, while still respecting
the fixed 1 N bound.

### Parameter classifications

These are parameter-specific classifications. They do not imply that the same
pipeline identified the entire vector.

| Parameter | Category | Best restricted condition | Passing targets | Worst-target p95 / bias |
|---|---|---|---:|---:|
| `k` | **A — practically identifiable** | 0.25–2 Hz, alpha-beta-gamma, TLS | 6/6 | 6.74% / 5.52% |
| `c` | **B — restricted conditions** | 0.5–5 Hz, causal low-pass, OLS | 4/6 | 32.23% / 29.78% |
| `m_eff` | **B — restricted conditions** | 0.5–5 Hz, causal low-pass, IV | 5/6 | 34.62% / 15.12% |
| `omega_n` ratio form | **B — restricted conditions** | 2–10 Hz, causal low-pass | 5/6 | one target invalid |
| `zeta` ratio form | **C — not practical** | no qualifying condition | 1/6 | invalid/large tails |
| `1/m_eff` ratio form | **C — not practical** | no qualifying condition | 3/6 | 59.18% / 57.21% |

The stiffness classification is about `k` alone; the selected TLS condition
produced nonphysical estimates of other parameters for the high-damping target.
There is still no reliable full-vector estimator.

### Parameterization decision

Do not switch to `[omega_n, zeta, 1/m_eff]`. The ratio form made damping ratio
category C and inverse mass category C. Its best causal worst-target p95 errors
were `30.59%`, `857.89%`, and `59.18%`; deriving modal quantities from direct
physical estimates was generally better. Keep `[k, c, m_eff]`, report modal
quantities as diagnostics, and keep `m_eff` explicitly conditional.

### Figures

- [Parameter bias vs timing](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__timing_bias.png)
- [Causal accuracy, delay, attenuation, and cost](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__causal_filter_tradeoff.png)
- [OLS vs TLS vs IV](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__estimator_comparison.png)
- [Mass error vs acceleration strength](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__mass_vs_acceleration_strength.png)
- [Singular values and correlation](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__regressor_geometry.png)
- [Chirp band vs parameter error](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__frequency_band_errors.png)
- [Information vs disturbance](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__information_disturbance_pareto.png)
- [Direct vs modal/scale parameterization](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__parameterization_comparison.png)
- [Representative causal signals](results/figures/EXP-0003_20260828T225126.077199Z_s3101_b2da5c53__representative_causal_signals.png)

### Decision

Stage 1 is **not ready** for independent MATLAB/Simulink validation. Timing
quality explains a meaningful part of damping sensitivity but not the dominant
causal mass error. TLS is not beneficial, IV is conditional, no causal pipeline
matches the offline reference jointly, damping and mass are category B rather
than universally practical, and the alternate parameterization fails. No
MATLAB, EKF, vehicle, Gazebo, PX4, ROS 2, MPC, or Isaac work should begin from
this result.

## EXP-0004 — Sequential uncertainty-driven probing

### Scientific question

> Can sequential, uncertainty-driven probing identify the complete target
> dynamics more reliably than a single fixed probe while respecting the same
> physical disturbance limits?

### Frozen design

- Authoritative configuration:
  [exp_0004_sequential_active_identification.yaml](configs/experiments/exp_0004_sequential_active_identification.yaml).
- EXP-0001 through EXP-0003 were read-only. Their fingerprints remained
  `fc8d3375…ba39`, `f01de136…2efc`, and `067d0a76…84a8` before and after the
  run.
- Development used seeds 1701–1705. The held-out run used new seeds
  4101–4120 only after the probe library, estimator split, selection rule,
  budgets, thresholds, and kill criterion were frozen.
- All probes were unilateral and bounded by 1 N. Candidates were low,
  low–mid, high, and wide chirps plus low, broad, and high multisines.
- The estimator followed the parameter-specific EXP-0003 evidence without
  target tuning: alpha-beta-gamma/TLS for `k`, causal-low-pass/OLS for `c`,
  and causal-low-pass/IV for `m_eff`.
- The adaptive selector received only measured histories, the current estimate,
  jackknife/analytic covariance, residual variance, and predicted candidate
  information/disturbance. Ground truth was used only after selection for
  evaluation.
- Stopping occurred on estimated confidence, insufficient predicted gain,
  budget exhaustion, or three probes. The cumulative command-dose cap was
  `4.55 N² s`; duration was capped at 9 s.

### Reference run

- Run ID: `EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2`
- 600 target/strategy/seed sequences and 1,219 executed probes.
- 973 prospective candidate evaluations.
- Twenty held-out seeds; zero safety events; all eight integrity checks passed.
- Stage 1 decision: **CONTINUE_STAGE_1**.
- Kill criterion: **STOP_ACTIVE_FULL_VECTOR_IDENTIFICATION** for this probing
  architecture.
- [Manifest](runs/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2/manifest.json)
- [Trial data](runs/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2/probe_trials.csv.gz)
- [Strategy summary](results/tables/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__strategy_summary.csv)
- [Decision](runs/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2/stage1_decision.json)

### Baselines and adaptive result

| Strategy | Full-vector success | Median probes / duration | Median dose | Worst-target p95 RMS error |
|---|---:|---:|---:|---:|
| Single 0.5–5 Hz chirp | **90.83%** | 1 / 3 s | 1.482 N² s | **20.92%** |
| Single broad multisine | 34.17% | 1 / 3 s | 0.375 N² s | 192.13% |
| Three repeated chirps | 75.83% | 3 / 9 s | 4.445 N² s | 39.53% |
| Predefined low/low–mid/high | 59.17% | 3 / 9 s | 4.461 N² s | 30.97% |
| Uncertainty-driven | 65.00% | 2.5 / 7.5 s | 3.149 N² s | 33.70% |

The single low–mid chirp was the strongest strategy. Its parameter threshold
success rates were `100.00%`, `94.17%`, and `96.67%` for `k`, `c`, and
`m_eff`. It still did not pass the cross-target tail gate: rigid-target damping
reached `32.13%` p95 and high-mass `m_eff` reached `35.00%` p95.

The fair matched-dose comparator was the first two stages of predefined
probing. Its median dose was `2.961 N² s`, only `5.98%` below the adaptive
median. It achieved `80.83%` full-vector success and `25.67%` worst-target p95
RMS error. Adaptive probing achieved only `65.00%` and `33.70%`, a
`-15.83` percentage-point success change with a worse tail. The predeclared
kill criterion therefore fired.

Adaptive probing modestly improved aggregate mass tails relative to the single
chirp (`24.45%` to `22.04%` overall p95), but degraded stiffness
(`5.49%` to `21.69%`) and damping (`30.92%` to `51.05%`). The lightly damped
and rigid targets ended at zero full-vector success under adaptation, driven by
`56.83%` and `46.44%` damping p95 respectively; rigid stiffness also rose to
`22.42%`.

### Uncertainty and stopping diagnosis

The selector declared damping the weakest parameter in all 139 follow-up
decisions. It never selected a probe specifically for stiffness or mass. This
was a failure of uncertainty calibration, not a lack of candidate variety.

- 41 trials stopped after one probe, 19 after two, and 60 reached three.
- All 60 confidence stops were correct in 59 cases (`98.33%` precision).
- Of the 79 trials told to continue after probe one, 69 (`87.34%`) already met
  the ground-truth accuracy thresholds. Their final success fell to `48.10%`.
- A selected follow-up reduced the named weak-parameter error only `41.01%` of
  the time.
- Predicted variance reduction and realized error reduction were negatively
  correlated (`Spearman rho = -0.478`).

The covariance was useful as a conservative stop certificate for easy targets,
but not as a value-of-information model for choosing another probe. Repeated
causal filtering and parameter-specific EIV bias do not behave like independent
zero-mean variance, so adding nominal Fisher information can increase bias.

### Frequency information and disturbance

Residualized information per command dose favored low multisine for stiffness
and broad multisine for damping and mass. Low/low–mid content was nearly as
informative for stiffness, while damping and inertial information increased
with broader content. That nominal ordering did not predict estimation quality:
the single broad multisine produced `315.80%` overall damping p95 and only
`34.17%` full-vector success. Causal phase/EIV bias dominated its nominal
information advantage. The 0.5–5 Hz chirp remains the best empirical fixed
probe in this model.

Adaptive probing stayed safe: maximum peak force was `1.0 N`, displacement
`15.68 mm`, velocity `0.271 m/s`, and acceleration `5.323 m/s²`. Its median
absolute input energy was `0.0324 J`. The negative result is therefore not a
safety-limit artifact or a win obtained by excess force.

### Figures

- [Uncertainty after each probe](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__uncertainty_after_each_probe.png)
- [Parameter error after each probe](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__parameter_error_after_each_probe.png)
- [Selected sequences by target](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__selected_probe_sequence_by_target.png)
- [Success versus probe count](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__success_rate_vs_probe_count.png)
- [Information versus disturbance](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__information_vs_disturbance.png)
- [Fixed versus adaptive](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__fixed_vs_adaptive.png)
- [Energy versus accuracy](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__cumulative_energy_vs_accuracy.png)
- [Resolved follow-up examples](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__second_probe_resolution_examples.png)
- [Parameter information by probe](results/figures/EXP-0004_20260828T232523.568285Z_s4101_0ceb48c2__parameter_information_by_probe.png)

### Decision

Sequential probing did not improve complete-vector identification. More probes
usually added parameter-distorting causal/EIV bias instead of independent
information. Stop searching for an active sequence that recovers all three
physical coefficients with this architecture. Stage 1 is not ready for
MATLAB/Simulink because no method meets every target-wise p95 gate. The next
Stage 1 experiment should test decision-sufficient outputs—stiffness,
dissipation/resonance bounds, and interaction-risk classification—from one
fixed low–mid chirp, without requiring a separately accurate `m_eff`.

## EXP-0005 — Fixed-chirp decision sufficiency

### Scientific question

> Can one fixed low-mid-frequency chirp reliably estimate task-relevant
> interaction properties and classify whether sustained contact is safe,
> without requiring accurate separate estimates of `k`, `c`, and `m_eff`?

### Frozen design

- Authoritative configuration:
  [exp_0005_decision_sufficiency.yaml](configs/experiments/exp_0005_decision_sufficiency.yaml).
- A single nonadaptive unilateral 0.5 N chirp covered 0.5–5 Hz for 3 s. The
  primary synchronized 200 Hz sensing pipeline was causal low-pass; centered
  or future-sample filtering was prohibited from the primary result.
- The target population was sampled continuously in log space over
  `75–2500 N/m`, `0.3–60 Ns/m`, and `0.10–6 kg`, including targets between the
  original named classes.
- Training used 240 targets on seeds 5101–5120; calibration used 120 targets on
  5201–5210; validation used 240 untouched targets on 6101–6120. Every target
  was evaluated under low, nominal, and high causal sensor noise.
- The hidden future maneuver ramped to and held 2 N, unloaded, and observed
  ringdown. SAFE/CAUTION/UNSAFE labels came only from its simulated
  displacement, velocity, oscillation, settling, and 20 mm contact-travel
  proxy. All validation predictions were created before that response was
  simulated and joined.
- Transparent predictors were a direct threshold rule, ridge response/bound
  models, and multinomial logistic regression. Feature ablations compared
  estimated `[k,c,m_eff]`, stiffness, frequency response, time response, and a
  combined task set. The primary combined set excluded estimated `c` and
  `m_eff`.
- The highest-priority gate required no more than 5% false-safe decisions in
  each noise regime. Accuracy, unsafe recall, peak-displacement upper-bound
  coverage, safe-force lower-bound coverage, and performance in the worst
  mass-error quartile were also frozen before validation.

### Reference run and integrity

- Run ID: `EXP-0005_20260829T191551.247939Z_s6101_8de417c0`.
- 7,920 held-out prediction rows; risk distribution 163 SAFE, 26 CAUTION, and
  51 UNSAFE targets.
- All ten integrity checks passed. Validation probes had zero configured safety
  events. One training target exceeded the `0.5 m/s` probe velocity limit by
  `0.000194 m/s`, retained in the safety audit.
- EXP-0001 through EXP-0004 fingerprints were unchanged before and after the
  run. Seventy-five tests passed before untouched validation.
- [Manifest](runs/EXP-0005_20260829T191551.247939Z_s6101_8de417c0/manifest.json)
- [Decision](runs/EXP-0005_20260829T191551.247939Z_s6101_8de417c0/stage1_decision.json)

### Safety classification

| Noise | Accuracy | False-safe rate | False-safe count | Unsafe recall |
|---|---:|---:|---:|---:|
| Low | 85.83% | 2.60% | 2/77 non-safe | 84.31% |
| Nominal | 86.25% | 3.90% | 3/77 non-safe | 84.31% |
| High | 87.08% | **6.49%** | 5/77 non-safe | 84.31% |

Accuracy passed its nominal/high thresholds. High-noise false-safe rate failed
the 5% limit, and unsafe recall missed the 85% limit by one case in each noise
regime.

The failure was specific. All nominal false-safe cases had safe displacement,
velocity, and late-hold oscillation but unsafe settling of `2.932–3.000 s`.
The predicted settling upper bounds were only `1.538–1.956 s`. The 3 s probe
ends without a deliberate post-chirp zero-force observation, so slow decay can
remain hidden even when the forced response amplitude looks benign.

### Task-quantity prediction

- Peak displacement: `4.84%` median and `17.23%` p95 relative error; nominal
  upper-bound coverage `96.25%`.
- Peak velocity: `3.89%` median and `14.85%` p95 relative error.
- Dominant ringdown frequency: `12.71%` median but `219.28%` p95 relative
  error, showing severe low-amplitude/tail cases.
- Settling: `0.063 s` median and `1.100 s` p95 absolute error. Relative error is
  not interpretable because 139 targets settle immediately under the declared
  rule.
- Disturbance-limited force: `14.70%` median and `48.00%` p95 relative error.
  Its lower-bound coverage was `94.17%`, `93.75%`, and `95.00%` for low,
  nominal, and high noise, so it did not provide the required 95% certificate.
- The contact-travel proxy was detected in 33/34 nominal cases (`97.06%`
  recall) with six false alarms.

### Do explicit parameters matter?

| Nominal feature set | Accuracy | False-safe rate |
|---|---:|---:|
| Estimated `[k,c,m_eff]` | 78.33% | **0.00%** |
| Stiffness only | 45.42% | 2.60% |
| Frequency response | 79.17% | 6.49% |
| Time response | 76.25% | 1.30% |
| Combined task features | **86.25%** | 3.90% |

Stiffness alone did not capture the safety problem. Frequency and time response
were complementary, and the combined task set delivered the highest overall
accuracy without estimated damping or mass. However, the full-parameter set
was more conservative and had zero nominal false-safe cases. The data therefore
show that accurate separate parameters are not necessary for every correct
decision, not that physical parameters can be discarded entirely.

The worst mass-error quartile had `32.94%` median mass error, `80.00%` decision
accuracy, and zero nominal false-safe cases. The 34 targets above 30% mass error
had `85.29%` accuracy and zero false-safe cases. This is direct evidence that
poor mass estimation and correct safety decisions can coexist. It does not
rescue the primary classifier's high-noise failure.

### Figures

- [Representative SAFE/CAUTION/UNSAFE probe responses](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__representative_probe_response.png)
- [Predicted versus actual future quantities](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__quantitative_predictions.png)
- [Confusion matrix and false-safe audit](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__confusion_and_false_safe.png)
- [Risk score versus actual severity](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__risk_score_vs_actual_severity.png)
- [Task-feature space](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__feature_space.png)
- [Noise and feature-set performance](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__sensing_and_feature_set_performance.png)
- [Performance versus target dynamics](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__performance_vs_target_dynamics.png)
- [Physical-parameter error versus decision accuracy](results/figures/EXP-0005_20260829T191551.247939Z_s6101_8de417c0__parameter_accuracy_vs_decision_accuracy.png)

### Decision

The predeclared kill criterion fired: accuracy passed, but high-noise
false-safe rate, unsafe recall, low-noise displacement-bound coverage, and
low/nominal safe-force coverage failed. One fixed chirp is informative but is
not a validated safety certificate under the complete sensing envelope.

Stage 1 remains **CONTINUE_STAGE_1**. Do not retune on seeds 6101–6120 and do
not start MATLAB/Simulink or vehicle work. The next architecture test should
append a fixed zero-force causal observation window after the same chirp. It
adds no probing energy and directly targets the observed slow-settling blind
spot.

## EXP-0006 — Passive ring-down safety observation

### Scientific question

> Does a zero-force causal observation period after the chirp provide enough
> ring-down information to eliminate or substantially reduce false-safe
> decisions without applying additional probing energy?

### Protocol

- Authoritative configuration:
  [exp_0006_passive_ringdown.yaml](configs/experiments/exp_0006_passive_ringdown.yaml).
- The same unilateral 0.5 N, 0.5–5 Hz, 3 s chirp was followed by an exactly
  zero-force observation of 0–3 s. Force-squared dose remained
  `0.206124 N²s` for every duration.
- The EXP-0005 target distribution, hidden 2 N sustained-contact maneuver, and
  SAFE/CAUTION/UNSAFE definitions were unchanged.
- Training used seeds 7101–7120, calibration used 7201–7210, and untouched
  validation used 8101–8120. Each seed supplied 12 log-Latin-hypercube targets.
- The primary synchronized 200 Hz causal-low-pass pipeline was tested at low,
  nominal, and high noise. No centered/noncausal filtering was used.
- Causal prefix features included decay rate/fit, dominant frequency,
  logarithmic decrement and damping diagnostics, residual response, threshold
  dwell/time, zero crossings, and energy decay.
- Comparisons used chirp-only, ring-down-only, chirp plus ring-down, and
  physical-diagnostic plus chirp plus ring-down feature sets. Ground truth was
  joined only after every validation prediction was frozen.
- The predeclared gate required nominal/high false-safe rates no greater than
  1%/2%, at least 80% accuracy and 90% UNSAFE recall in both regimes, no added
  energy, detection of every nominal EXP-0005 failure, and a fixed wait no
  longer than 2 s.

### Reference run and integrity

- Run ID: `EXP-0006_20260829T194621.991365Z_s8101_2966d3d2`.
- 240 validation targets: 158 SAFE, 34 CAUTION, 48 UNSAFE.
- 20,160 held-out prediction rows. All 12 integrity checks passed and no probe
  safety event occurred in any partition.
- EXP-0001 through EXP-0005 artifact fingerprints were unchanged before and
  after execution. Eighty-one tests passed before untouched validation.
- [Manifest](runs/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2/manifest.json)
- [Decision](runs/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2/stage1_decision.json)

### Passive observation result

The zero-second baseline was refitted on EXP-0006's disjoint training and
calibration partitions, so it is a protocol baseline rather than the frozen
EXP-0005 model.

| Wait | Noise | Accuracy | False-safe rate | Count | UNSAFE recall |
|---:|---|---:|---:|---:|---:|
| 0.0 s | Nominal | 89.17% | 2.44% | 2/82 | 89.58% |
| 0.0 s | High | 87.92% | 3.66% | 3/82 | 89.58% |
| 0.5 s | Nominal | 83.33% | **0.00%** | 0/82 | 91.67% |
| 0.5 s | High | 82.92% | **0.00%** | 0/82 | 89.58% |
| 3.0 s | Nominal | 84.58% | **0.00%** | 0/82 | 91.67% |
| 3.0 s | High | 84.58% | **0.00%** | 0/82 | 91.67% |

Half a second of passive observation eliminated all false-safe decisions in
all three noise regimes without extra excitation. The decision changed for
9.17% of nominal and 7.92% of high-noise targets. The change was conservative:
nominal SAFE recall fell from 92.41% to 82.91% as more actual SAFE targets were
labeled CAUTION. No non-SAFE target was labeled SAFE after 0.5 s.

Longer observation did not monotonically improve overall accuracy or settling
prediction. Every 0.5–3 s window retained zero false-safe decisions. A 3 s
fixed window met the numerical nominal/high classification limits but violated
the predeclared 2 s practical-duration limit. At every duration at or below
2 s, one nominal or high-noise UNSAFE recall was `43/48 = 89.58%`, one case
below the 90% gate. Those errors were UNSAFE-to-CAUTION, not false-safe.

### What the ring-down contributed

At 0.5 s nominal noise, chirp-only accuracy/false-safe were `89.17%/2.44%`,
ring-down-only were `57.08%/1.22%`, chirp plus ring-down were
`83.33%/0.00%`, and adding estimated physical diagnostics gave
`83.75%/0.00%`. Ring-down alone is insufficient; its value is a conservative
persistence cue added to the informative chirp response.

Threshold dwell fraction was the strongest ring-down term in the standardized
settling predictor (`|coefficient| = 1.633`), followed by observed time to the
response threshold (`1.590`). These are coefficient diagnostics among
correlated features, not a theoretical feature-optimality claim.

Nominal 0.5 s settling-time error was `0.057 s` median and `0.778 s` p95,
compared with `0.079 s` and `0.939 s` for chirp only. Peak-displacement error
remained `4.07%/16.19%` median/p95, with `97.50%` upper-bound coverage.

### Historical failure audit and early stopping

All five targets false-safe under at least one EXP-0005 noise regime were
reproduced only after model/policy freeze. At 0.5 s all were CAUTION or UNSAFE
at every noise level, and the persistence veto activated on the three nominal
historical failures. However, EXP-0006's new zero-second fitted baseline already
classified the historical cases as non-SAFE. The untouched 8101–8120 cases,
not that legacy audit, provide the direct correction evidence.

The causal truth-independent stopping rule produced zero false-safe decisions,
83.33%/84.17% nominal/high accuracy, and 91.67% UNSAFE recall in both regimes.
Its median wait was 1.5 s, but p95 and maximum wait were 3.0 s. It is promising
but cannot retroactively satisfy the fixed-window criterion.

### Figures

- [Representative chirp and passive responses](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__representative_chirp_ringdown.png)
- [Locked EXP-0005 failure audit](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__exp0005_false_safe_ringdown_audit.png)
- [Settling prediction](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__settling_predicted_vs_actual.png)
- [False-safe versus observation duration](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__false_safe_vs_observation_duration.png)
- [Accuracy versus observation duration](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__accuracy_vs_observation_duration.png)
- [Confidence and causal early stopping](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__confidence_and_early_stopping.png)
- [Decay features by risk class](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__decay_features_by_risk.png)
- [Noise and feature-set comparison](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__noise_and_feature_set_performance.png)
- [Chirp-only versus combined confusion](results/figures/EXP-0006_20260829T194621.991365Z_s8101_2966d3d2__chirp_vs_ringdown_confusion.png)

### Decision

Passive ring-down observation verified a 100% relative reduction in held-out
false-safe decisions at nominal and high noise after 0.5 s, with no probing-
energy increase. Accurate separate damping and effective mass were unnecessary
for that benefit.

The complete predeclared criterion nevertheless did not pass: the best
practical fixed window missed high-noise UNSAFE recall by one target, while the
first fixed window meeting that three-class threshold required 3 s and exceeded
the 2 s duration limit. Stage 1 remains **CONTINUE_STAGE_1**. Do not retune on
seeds 8101–8120 or start independent MATLAB/Simulink validation. The next test
should independently replicate the locked 0.5 s policy and prospectively
separate the primary SAFE/non-SAFE actuation decision from the secondary
CAUTION/UNSAFE severity label.
## EXP-0007

Locked-policy independent replication of EXP-0006. The frozen chirp plus 0.5 s passive observation retained low false-safe performance on a new broad and boundary-enriched population; Stage 1 passes to independent MATLAB/Simulink validation. See `lab/experiments/EXP-0007.md`.

## STAGE 2 — MATLAB/Simulink validation

Independent run `STAGE2_20260829T221139.099Z_s20101` used 1,360 new MATLAB
targets. Nominal/high false-safe rates were zero (0.586% one-sided 95% upper),
binary accuracy was 88.38%/87.65%, boundary false-safe rate was zero, and no
probe violation occurred. MATLAB, Simulink, and matched Python trajectories
agreed to floating-point precision. Stage 2 passes; see
`lab/experiments/STAGE2_MATLAB_VALIDATION.md`.
