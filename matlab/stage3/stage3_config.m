function cfg = stage3_config()
%STAGE3_CONFIG Independently declared EXP-0008 physical constants.
cfg.dt=0.001; cfg.duration=3.5; cfg.g=9.80665;
cfg.mass=1.50; cfg.J=diag([0.029 0.029 0.055]); cfg.arm=0.23;
cfg.kT=1.90e-5; cfg.kQ=2.60e-7; cfg.motor_tau=0.030;
cfg.omega_min=0; cfg.omega_max=900;
cfg.rotor_position=[cfg.arm 0 0;0 cfg.arm 0;-cfg.arm 0 0;0 -cfg.arm 0];
cfg.spin=[1;-1;1;-1];
cfg.kp=[4;4;6]; cfg.kd=[4;4;5]; cfg.kR=[1.8;1.8;1.2]; cfg.kw=[0.35;0.35;0.40];
cfg.max_accel=8; cfg.force_gain=0.10; cfg.admittance=0.04;
cfg.position_limit=0.025; cfg.force_limit=1.25; cfg.passive_retraction=0.006;
cfg.probe_offset=[0.30;0;-0.08]; cfg.surface=[0;0;0.92]; cfg.normal=[1;0;0];
cfg.contact_k=5000; cfg.contact_c=10;
cfg.target_k=2200; cfg.target_c=12; cfg.target_m=1.2;
cfg.probe_amplitude=0.5; cfg.probe_f0=0.5; cfg.probe_f1=5; cfg.probe_duration=3.0;
end
