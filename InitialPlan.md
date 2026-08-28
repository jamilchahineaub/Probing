# Safe Active Probing and Uncertainty-Aware MPC for Aerial Contact Inspection

## Executive summary

I have turned the research direction into a **repository-ready, 90-day experimental programme** rather than just a list of ideas.

The project is centred on one question:

> **Can a UAV make a deliberately small and bounded physical probe of an unknown structure, determine what it has learned and how uncertain that knowledge still is, and then use that uncertainty to decide whether sustained inspection contact is safe?**

The intended system is:

\[
\boxed{\text{bounded micro-probe}}
\rightarrow
\boxed{\text{interaction identification}}
\rightarrow
\boxed{\text{uncertainty + observability}}
\rightarrow
\boxed{\text{safe contact envelope}}
\rightarrow
\boxed{\text{continue / reduce / re-probe / abort}}
\]

The research review changed one important part of our earlier thinking: **probing by itself is not innovative enough**. Contact-aware Fisher-information maximisation for robot system identification already exists, and aerial robots have already demonstrated interaction with unknown compliant obstacles. citeturn22view0turn16search1

Similarly, “new EKF estimates stiffness” cannot be the entire contribution. Sensorless aerial wrench estimation, momentum-based contact detection, adaptive impedance, hybrid force control, NMPC, and robust interaction with poorly modelled environments are all already represented in the literature. citeturn18search0turn18search1turn18search2turn16search2

The stronger innovation target is therefore:

> **Use safe active probing to estimate a probabilistic local interaction model, explicitly test whether that model is sufficiently informative, and propagate its uncertainty into a predictive decision about how much inspection contact is actually safe.**

That is what the project plan is designed to prove or kill.

The complete repository specification, scripts, Markdown templates, experiment matrices, Docker/CI examples, daily schedule, Mermaid timeline, plotting standard, literature ledger and daily-summary workflow are in the generated file:

**[Download the comprehensive research project Markdown file](sandbox:/mnt/data/SAFE_ACTIVE_PROBING_RESEARCH_PROJECT.md)**

## Research position and innovation target

The literature gives us a much clearer line that we cannot cross with weak novelty claims.

Bodie et al. already demonstrated variable axis-selective impedance plus direct force control for contact-based aerial inspection using an onboard force sensor. Tzoumanikas et al. already incorporated interaction forces into hybrid force/position NMPC for an aerial manipulator. citeturn16search0turn16search3

Brunner et al. went further and demonstrated passivity-based interaction with moving objects without requiring an accurate environment model, using impedance/wrench control, a momentum-based wrench estimator and an energy tank. citeturn16search2 Zhang et al. demonstrated learned variable impedance for uncertain, heterogeneous surfaces and direct simulation-to-real transfer. citeturn18search2

Most importantly for us, Aucone et al. demonstrated aerial traversal of objects over a large and **unknown range of compliance**, including experiments involving compliant obstacles and branches. Their contribution combines interaction morphology, force feedback and predictive control rather than first identifying an exact stiffness model. citeturn16search1

So our paper cannot say:

> “Existing methods assume rigid walls; we handle flexible objects.”

That would be false.

The research gap we should test is narrower and stronger:

\[
\text{Unknown target}
\rightarrow
\text{small bounded experiment}
\rightarrow
p(\theta_{\rm target})
\rightarrow
\text{is this estimate trustworthy?}
\rightarrow
\text{is inspection feasible at a desired risk?}
\]

The key output is therefore not simply

\[
\hat k=500\;{\rm N/m}.
\]

It is closer to:

\[
\theta_e=
[k,c,m_{\rm eff}],
\qquad
\theta_e\sim(\hat\theta_e,P_\theta)
\]

followed by

\[
F_{\rm safe}^{*}
=
\max F_d
\]

subject to probabilistic constraints on target displacement, UAV attitude, force, actuator reserve and contact stability.

That gives us an operational policy:

\[
\boxed{
\text{INSPECT}
\;|\;
\text{REDUCE FORCE}
\;|\;
\text{RE-PROBE}
\;|\;
\text{ABORT}
}
\]

This is materially different from simply asking the controller to survive contact.

There are two particularly interesting later extensions preserved in the plan.

The first is **attachment-instability detection**: distinguish “soft but stable” from “something appears to be becoming detached.” The second is **sensorless probing** in which the UAV infers contact wrench through its own dynamics, IMU and actuation signals. But neither is being artificially forced into the first contribution. Recent 2026 work already demonstrates sensorless external-wrench estimation and proprioceptive contact localisation on UAV systems, so our sensorless contribution would need to concern **uncertain target identification and decision quality**, not merely removing the force sensor. citeturn18search0turn18search1

## Technical stack and simulation strategy

For the required Ubuntu 22.04 host, the plan freezes **ROS 2 Humble Hawksbill**. Humble packages are officially provided for Ubuntu Jammy 22.04. Current PX4 documentation prefers Jazzy on Ubuntu 24.04 for new development but explicitly states that Ubuntu 22.04 users can use Humble. citeturn20search2turn20search0

The primary integrated simulation stack is therefore:

\[
\boxed{
\text{Ubuntu 22.04}
+
\text{ROS 2 Humble}
+
\text{PX4 v1.17.0}
+
\text{Gazebo Harmonic}
}
\]

PX4 v1.17.0 is the current stable release at the project start date; v1.18 is still represented by a pre-release in the current release listing. citeturn19search0 PX4's current Ubuntu tooling supports both Ubuntu 22.04 and 24.04 and installs Gazebo Harmonic. citeturn20search1

There is one subtle but important compatibility choice captured in the repository: on Ubuntu 22.04 with ROS 2 Humble and PX4/Gazebo Harmonic, PX4 explicitly instructs installing:

```bash
sudo apt install ros-humble-ros-gzharmonic
```

rather than accidentally mixing the default Humble Gazebo bridge with the wrong Gazebo generation. citeturn20search0

The simulators are deliberately divided by scientific purpose:

| Platform | What we use it for |
|---|---|
| **JAX / CasADi reduced model** | Massive parameter sweeps, observability, Fisher information, estimator debugging |
| **MuJoCo** | Fast independent contact/flexible-target cross-validation |
| **PX4 + Gazebo Harmonic** | Primary end-to-end UAV, flight-controller, ROS 2, timing and contact simulation |
| **Isaac Sim** | Later high-fidelity physics/perception/deformable-object experiments |
| **AirSim-style environment** | Not on the critical path because it does not solve our central contact-identification problem better than the above stack |

The current research pins in the plan include **PX4 v1.17.0**, **MuJoCo 3.12.0**, **CasADi 3.8.0**, **acados v0.6.0**, and **Isaac Sim 6.0.0**. MuJoCo 3.12.0 was released on August 20, 2026; CasADi 3.8.0 on August 25, 2026; and acados v0.6.0 on August 6, 2026. citeturn19search1turn22view3turn19search3

Isaac Sim 6.0 officially supports Ubuntu 22.04/24.04. NVIDIA's current requirement table lists 32 GB RAM and an RTX 4080 16 GB at the minimum x86_64 tier, 64 GB and an RTX 5080 at the “good” tier, and a 48 GB RTX PRO 6000 Blackwell-class GPU at the ideal tier. citeturn21search0 Because most of our early research is system identification and Monte Carlo rather than photorealistic rendering, the plan deliberately keeps Isaac away from the critical path.

The **exact physical drone remains unspecified**, as requested. We first determine required force, moment arm, thrust reserve, payload, estimator computation and control authority. Only then should we select hardware.

The target modelling progresses through:

\[
\text{rigid wall}
\rightarrow
\text{Kelvin–Voigt}
\rightarrow
\text{mass-spring-damper}
\rightarrow
\text{hinged plate}
\rightarrow
\text{branch mode}
\rightarrow
\text{cable}
\rightarrow
\text{weak attachment}
\rightarrow
\text{nonlinear/OOD target}.
\]

That means we do **not** start by trying to make Gazebo simulate a perfect botanical tree branch. We first determine whether the scientific effect exists in a model we understand analytically.

## Experimental programme

The first month is intentionally aggressive. Its purpose is to discover whether this idea deserves a paper before we spend months making the simulator beautiful.

The first major gate is **identifiability**.

For a Kelvin–Voigt interaction,

\[
F=k\delta+c\dot\delta,
\]

\(k\) and \(c\) can be treated as a simple parameter-estimation problem. Once effective target inertia enters,

\[
F=m_e\ddot x+c\dot x+kx,
\]

the experiment must excite enough dynamics to separate

\[
m_e,\qquad c,\qquad k.
\]

This is where probing becomes a scientific experiment rather than a tap.

We compare:

- ramp/triangular pushes;
- bounded half-sine pulses;
- single-frequency sine;
- chirps;
- multisine;
- PRBS as a system-identification benchmark;
- finally a **bounded Fisher-information-optimised probe**.

Contact-aware Fisher-information behaviour synthesis was demonstrated in general robotics in RSS 2025, so our information-optimal probe must differentiate itself through aerial dynamics, strict interaction bounds and the subsequent inspection-safety decision. citeturn22view0

We evaluate the probes using something like

\[
\mathcal I(\theta)
=
\sum_k
S_k^TR^{-1}S_k,
\qquad
S_k=
\frac{\partial y_k}{\partial\theta},
\]

and measure:

\[
\log\det\mathcal I,
\quad
\lambda_{\min}(\mathcal I),
\quad
\kappa(\mathcal I),
\]

alongside how much the probe actually disturbs the object.

That gives us the plot I think could become one of the central figures:

\[
\boxed{
\text{information gained}
\quad \text{versus} \quad
\text{target/UAV disturbance}
}
\]

The estimator suite is deliberately broader than an EKF:

| Method | Purpose |
|---|---|
| RLS / Bayesian RLS | transparent linear baseline |
| Augmented EKF | initial probabilistic state/parameter estimator |
| UKF | nonlinear filter comparison |
| MHE | constrained optimisation benchmark |
| IMM | rigid/compliant/loose/breakaway mode inference |
| Candidate interaction-aware EKF/IMM | only promoted to a contribution if the baselines expose a real deficiency |

The EKF does **not** get a free pass because we like EKFs. It must be tested for confidence calibration using innovation statistics, NIS/NEES where ground truth is available, posterior coverage and parameter correlation.

A filter that reports

\[
k=500\pm5\;{\rm N/m}
\]

when the true stiffness is 800 N/m is more dangerous for our controller than a noisier but honest estimator.

The controller comparison then proceeds from:

\[
\text{fixed impedance}
\rightarrow
\text{admittance}
\rightarrow
\text{hybrid force-position}
\rightarrow
\text{nominal NMPC}
\rightarrow
\text{robust/passivity baseline}
\rightarrow
\boxed{\text{chance-constrained MPC}}.
\]

A representative chance constraint is:

\[
\Pr(F_c<F_{\max})\ge1-\epsilon_F,
\]

\[
\Pr(|x_e|<x_{e,\max})\ge1-\epsilon_x,
\]

\[
\Pr(|\phi|<\phi_{\max})\ge1-\epsilon_\phi.
\]

For locally Gaussian uncertainty, the first implementation uses deterministic tightening of the form

\[
g(\mu)+
\Phi^{-1}(1-\epsilon)\sigma_g
\le0.
\]

But the plan explicitly does **not** assume Gaussian chance constraints will survive contact-mode changes. If their calibration is poor, sigma-point, scenario-based or distributionally robust alternatives become the next method. Modern contact-rich stochastic-control and nonlinear covariance-steering work provides the methodological foundation for treating these as serious alternatives rather than cosmetic additions. citeturn17search22turn22view2

The full Monte Carlo programme then injects target uncertainty, force/IMU noise, bias, latency, jitter, UAV mass/inertia error, thrust-model error, wind, contact-normal error and sudden target changes.

The crucial metric is:

\[
\boxed{\text{false-safe rate}}
\]

meaning:

> The system declared contact safe, committed to inspection, and subsequently violated the defined safety envelope.

That is much more relevant to our hypothesis than merely showing a 0.2 N improvement in force RMSE.

## Competitive baselines and what we need to beat

The generated plan establishes a Tier-A baseline set.

**Bodie et al.** is the known/rigid inspection baseline: force sensing, impedance and active force control. citeturn16search0turn16search4

**Tzoumanikas et al.** is the hybrid contact-NMPC baseline: interaction forces are already part of aerial predictive control, so “we use MPC during contact” is not novelty. citeturn16search3

**Brunner et al.** is one of the hardest conceptual competitors because their energy-tank/passivity framework explicitly addresses poorly modelled moving environments without accurate environment models. Our explicit identification framework needs to demonstrate a benefit in **inspection feasibility or useful force selection**, not merely stability. citeturn16search2

**Zhang et al.** prevents us claiming variable/adaptive impedance on heterogeneous uncertain surfaces as new. citeturn18search2

**Aucone et al.** is the closest unknown-compliance competitor and is therefore a must-reproduce conceptually. Their work shows that aerial robots can interact with unknown compliant obstacles without first doing our proposed identification step. citeturn16search1

**Guo et al.** gives us a recent quantitative motion/force sanity benchmark: the aerial-calligraphy experiments report about **2.9 cm end-effector position RMSE, 0.7 N force RMSE and 0.59 IoU**. citeturn18search3

**Sathyanarayan and Abraham** is the information-optimal probing competitor: active contact/Fisher-information optimisation exists, so ours must be constrained, aerial and inspection-decision oriented. citeturn22view0

**Naser et al.** is the current sensorless observer competitor: their 2026 AGNO explicitly estimates external interaction wrench without a dedicated force/torque sensor and reports better RMSE than an EKF, particularly for torque. citeturn18search0turn18search8

**Brummelhuis et al.** shows that UAV proprioceptive contact can already be used to infer aspects of an unknown surface, in their case contact localisation and roof inclination. citeturn18search1

**Zhan et al.** is important for the eventual application because their 2026 system closes a contact-aware perception/control loop using onboard sensing and reports a **66.01% velocity-estimation improvement at contact**. citeturn17search2

This leads to a very clean competitive objective:

> We do **not** need to prove that all these controllers are bad. We need to construct conditions where their assumptions differ from ours and then test whether spending a small amount of safe interaction to estimate uncertainty produces a useful decision advantage.

The most informative comparison is likely something like:

```text
Unknown target
     │
     ├── Fixed impedance
     ├── Adaptive impedance
     ├── Nominal NMPC
     ├── Robust/passivity interaction
     ├── Probe + point estimate MPC
     └── Probe + uncertainty-aware MPC  ← ours
```

and plot:

\[
x=\text{inspection completion}
\]

against

\[
y=\text{false-safe probability}.
\]

That is a much stronger paper figure than “our controller has the lowest RMSE.”

## Repository, automation, and reproducibility

The generated Markdown defines a complete repository structure containing:

```text
configs/
src/
sim/
ros2_ws/
scripts/
tests/
literature/
lab/
data/
runs/
results/
docker/
.github/workflows/
```

Every run gets a unique ID and records:

\[
\boxed{
\text{config}
+
\text{seed}
+
\text{Git SHA}
+
\text{container/environment}
+
\text{raw data}
+
\text{metrics}
+
\text{safety events}
}
\]

ROS 2 data are specified for **rosbag2/MCAP**, for which Humble provides an MCAP storage plugin. citeturn15search1 PX4 `.ulg` logs remain alongside the ROS data. Processed scalar time series go to Parquet; YAML stores experiment configuration; JSON holds manifests and machine metrics.

The repository also separates:

```text
development seeds
validation seeds
final test seeds
```

so we cannot continuously tune on the same Monte Carlo cases and then pretend those cases are held out.

The daily workflow is **agentic but deterministic**:

```text
experiment finishes
      ↓
manifest + metrics + events written
      ↓
daily summariser scans runs
      ↓
counts failures/safety events
      ↓
aggregates metrics
      ↓
checks explicit milestone rules
      ↓
writes YYYY-MM-DD.autogen.md
      ↓
human reviews interpretation
```

The included Python script deliberately does **not** call an LLM. It mechanically reports what happened. This satisfies the useful part of an agentic lab—automatic bookkeeping, run accounting, regression detection and milestone tracking—without letting an AI silently invent a scientific story.

The GitHub Actions plan uses `ubuntu-22.04`, which remains an available GitHub-hosted runner label, while expensive nightly Monte Carlo and GPU experiments are assigned to self-hosted research runners. citeturn15search22

Most importantly, the README template contains the requested hard policy:

> **No AI-generated prose is permitted in final scientific deliverables.**

The repository permits machine-generated working summaries and deterministic figures/tables, but paper prose, abstracts, captions, result interpretation, conclusions, novelty claims and public scientific claims require human authorship and evidence verification.

The reproducibility chain is explicitly:

\[
\boxed{
\text{paper figure}
\rightarrow
\text{plotting script}
\rightarrow
\text{frozen analysis config}
\rightarrow
\text{run IDs}
\rightarrow
\text{raw data}
\rightarrow
\text{experiment config}
\rightarrow
\text{Git SHA}
\rightarrow
\text{environment}
\rightarrow
\text{seed}
}
\]

which is the standard we should hold ourselves to from the first experiment rather than trying to reconstruct provenance at paper-writing time.

## Research schedule and immediate objective

The first 30 days are written day-by-day in the file, beginning **August 28, 2026** and ending **September 26, 2026**. The entire 90-day plan runs through **November 25, 2026**.

The first ten days are the most important intellectually:

\[
\text{models}
\rightarrow
\text{probe signals}
\rightarrow
\text{RLS}
\rightarrow
\text{EKF}
\rightarrow
\text{FIM/observability}
\rightarrow
\text{UKF}
\rightarrow
\text{noise Monte Carlo}
\rightarrow
\boxed{\text{GO / NO-GO}}
\]

By Day 10, we should know whether

\[
[k,c,m_{\rm eff}]
\]

are actually identifiable under a probe gentle enough to make sense for this task.

If they are not, we **do not fake success by reporting EKF estimates anyway**. We change the state representation to something we can identify, potentially

\[
[\omega_n,\zeta],
\]

local impedance, or another task-relevant parameterisation.

By Day 22, the plan demands the first genuinely important result:

> **Does uncertainty-aware probing/control reduce false-safe decisions on held-out target/noise combinations?**

By Day 30, we freeze the scope.

That gives us a disciplined choice between:

\[
\boxed{
\text{probing + probabilistic identification + CC-MPC}
}
\]

or, if the data point elsewhere,

\[
\boxed{
\text{hybrid instability detection}
}
\]

or

\[
\boxed{
\text{sensorless interaction inference}
}
\]

or a simpler target-representation paper.

The goal is not to protect our favourite idea. It is to make the first month tell us **which version of the idea deserves to become the paper**.

**[Download `SAFE_ACTIVE_PROBING_RESEARCH_PROJECT.md`](sandbox:/mnt/data/SAFE_ACTIVE_PROBING_RESEARCH_PROJECT.md)**