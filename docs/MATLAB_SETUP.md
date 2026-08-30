# MATLAB/Simulink setup for Ubuntu 22.04

## Environment check (2026-08-30)

- OS: Ubuntu 22.04.5 LTS, x86_64
- MATLAB executable: `/usr/local/bin/matlab`
- MATLAB release: R2026a Update 5 (`26.1.0.3346908`)
- Simulink: available and batch model creation/load/simulation verified
- Available relevant products: Control System Toolbox, Simscape, Simscape
  Multibody, Optimization Toolbox, Model Predictive Control Toolbox, UAV
  Toolbox
- Not detected: Signal Processing Toolbox, System Identification Toolbox,
  Statistics and Machine Learning Toolbox, ROS Toolbox, Robotics System
  Toolbox
- Official installer search before installation: no MathWorks installer found in `~/Downloads`,
  `~/Desktop`, `/tmp`, `/opt`, or `/usr/local`
- GNU Octave was not found and is not an accepted substitute for Stage 2.
- Hardware snapshot: 15 GiB RAM, approximately 201 GiB free on the workspace
  filesystem.

## Official route

MathWorks lists Ubuntu 22.04 LTS as a validated Linux distribution for MATLAB
R2024b. Use the official MathWorks Downloads page and select the Linux 64-bit
installer for R2024b (or a newer release whose system requirements still list
Ubuntu 22.04):

- Downloads: https://www.mathworks.com/downloads/
- Installation guide: https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html
- R2024b Linux requirements: https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/system-requirements-release-2024b-linux.pdf

MATLAB and Simulink must be selected during installation. Select optional
toolboxes only when present in the account license. Do not use unofficial
mirrors, cracked installers, or activation workarounds.

## Manual action required

1. Open the official Downloads page, sign in to the MathWorks account, and
   verify that the account has a MATLAB license including Simulink.
2. Download the Linux installer for R2024b (or the newest Ubuntu-22.04-
   compatible release) and save it in `~/Downloads`.
3. Extract it, then return to Codex with the extracted installer path. If the
   installer requests license selection, credentials, or activation, complete
   those prompts manually; Codex will not bypass them.

## Verification completed

```bash
`matlab -batch "disp(version)"` returned `26.1.0.3346908 (R2026a) Update 5`.
The MATLAB physics test suite passed 3/3 tests (equilibrium, undamped energy,
and damped energy). The generated Simulink model was created, loaded, and
simulated in batch (`SIMULINK_BATCH_OK`).
```

The full Stage 2 population validation has not been started in this setup task.
