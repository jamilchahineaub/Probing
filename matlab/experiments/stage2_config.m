function cfg = stage2_config(repository_root)
% Frozen independent Stage 2 validation specification (declared pre-outcome).
if nargin==0, repository_root=fileparts(fileparts(fileparts(mfilename('fullpath')))); end
cfg.id='STAGE2-MATLAB-VALIDATION-0001'; cfg.dt=0.002; cfg.sensor_fs=200;
cfg.probe=struct('amplitude',0.5,'f0',0.5,'f1',5.0,'duration',3.0,'observation',0.5,'contact','unilateral');
cfg.bounds=struct('k',[75 2500],'c',[0.3 60],'m',[0.10 6.0]);
cfg.population=struct('broad_seeds',[20101 20180],'boundary_seeds',[20201 20300], ...
    'cases_per_seed',12,'boundary_safe',200,'boundary_nonsafe',200);
cfg.noise_names={'low','nominal','high'}; cfg.noise_multiplier=[0.5 1 2];
cfg.base_noise=struct('displacement',5e-5,'velocity',0.002,'acceleration',0.1,'force',0.02);
cfg.filter=struct('cutoff_hz',10,'nominal_group_delay_s',1/(2*pi*10)+1/200);
cfg.ring=struct('block_s',0.10,'tail_s',0.10,'x_threshold',5e-4,'v_threshold',0.01,'min_points',3,'peak_noise_multiplier',3);
cfg.maneuver=struct('force',2.0,'ramp_up',0.5,'hold_end',3.5,'ramp_down_end',4.0,'duration',7.0);
cfg.safe=struct('peak_displacement',0.012,'peak_velocity',0.12,'oscillation',0.0015,'settling',2.0);
cfg.unsafe=struct('peak_displacement',0.020,'peak_velocity',0.25,'oscillation',0.004,'settling',2.8);
cfg.contact_tracking_displacement=0.020; cfg.settling_fraction=0.05; cfg.settling_floor=2e-4; cfg.settling_velocity=0.01;
cfg.probe_limits=struct('force',0.500000000001,'displacement',0.025,'velocity',0.5,'acceleration',15.0);
cfg.policy_file=fullfile(repository_root,'configs','policies','exp_0007_locked_exp_0006_policy.json');
cfg.gate=struct('nominal_false_safe',0.01,'high_false_safe',0.02,'nominal_upper',0.01,'high_upper',0.02, ...
    'nominal_accuracy',0.85,'high_accuracy',0.85,'nominal_precision',0.99,'high_precision',0.99, ...
    'nominal_recall',0.99,'high_recall',0.98,'boundary_false_safe',0.02,'boundary_upper',0.025,'boundary_accuracy',0.75);
end
