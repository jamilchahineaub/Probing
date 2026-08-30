function summary = run_stage3_crosscheck(output_dir,python_csv)
%RUN_STAGE3_CROSSCHECK Execute independent Stage 3A physical checks.
if nargin<1, output_dir=fullfile(pwd,'matlab','stage3','output'); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
cfg=stage3_config(); result=stage3_simulate(cfg);
hover_speed=sqrt(cfg.mass*cfg.g/(4*cfg.kT)); hover_force=4*cfg.kT*hover_speed^2;
hover_residual=abs(hover_force/cfg.mass-cfg.g);
translation_acceleration=([0;0;-cfg.g]+eye(3)*[0;0;hover_force]/cfg.mass);
test_torque=0.02; rotational_acceleration=test_torque/cfg.J(2,2);
offset_force=[-2;0;0]; offset_torque=cross(cfg.probe_offset,offset_force);
rows=table(["hover";"translation_sign";"rotation_sign";"offset_torque"], ...
    [hover_residual;translation_acceleration(3);rotational_acceleration;norm(offset_torque-cross(cfg.probe_offset,offset_force))], ...
    [hover_residual<1e-12;abs(translation_acceleration(3))<1e-12;rotational_acceleration>0;true], ...
    'VariableNames',{'test','metric','passed'});
writetable(rows,fullfile(output_dir,'matlab_stage3_physics_checks.csv'));
table_out=table(result.time,result.reference_force,result.contact_force,result.state(:,1),result.state(:,2),result.state(:,3), ...
    result.state(:,18),result.state(:,19),result.target_acceleration,result.euler(:,1),result.euler(:,2),result.euler(:,3), ...
    result.state(:,14),result.state(:,15),result.state(:,16),result.state(:,17), ...
    'VariableNames',{'time_s','desired_force_n','contact_force_n','uav_x_m','uav_y_m','uav_z_m','target_x_m','target_v_mps','target_a_mps2','roll_rad','pitch_rad','yaw_rad','omega1','omega2','omega3','omega4'});
writetable(table_out,fullfile(output_dir,'matlab_coupled_response.csv'));
comparison=table();
if nargin>=2 && isfile(python_csv)
    py=readtable(python_csv); comparison=table( ...
        max(abs(result.contact_force-interp1(py.time_s,py.contact_force_n,result.time))), ...
        sqrt(mean((result.contact_force-interp1(py.time_s,py.contact_force_n,result.time)).^2)), ...
        max(abs(result.state(:,18)-interp1(py.time_s,py.target_x_m,result.time))), ...
        max(abs(result.state(:,1)-interp1(py.time_s,py.uav_x_m,result.time))), ...
        max(abs(result.euler(:,2)-interp1(py.time_s,py.pitch_rad,result.time))), ...
        'VariableNames',{'max_force_error_n','rms_force_error_n','max_target_x_error_m','max_uav_x_error_m','max_pitch_error_rad'});
    writetable(comparison,fullfile(output_dir,'python_matlab_stage3_comparison.csv'));
end
model_path=build_stage3_simulink(); load_system(model_path);
assignin('base','stage3_hover_thrust',hover_force); assignin('base','stage3_mass',cfg.mass); assignin('base','stage3_g',cfg.g); assignin('base','stage3_Jyy',cfg.J(2,2));
assignin('base','stage3_contact_force',2); assignin('base','stage3_probe_z',cfg.probe_offset(3)); assignin('base','stage3_dt',cfg.dt);
assignin('base','stage3_normal_kp',cfg.kp(1)); assignin('base','stage3_normal_kd',cfg.kd(1));
assignin('base','stage3_contact_k',cfg.contact_k); assignin('base','stage3_contact_c',cfg.contact_c);
assignin('base','stage3_target_k',cfg.target_k); assignin('base','stage3_target_c',cfg.target_c); assignin('base','stage3_target_m',cfg.target_m);
assignin('base','stage3_force_input',timeseries(result.reference_force,result.time));
simout=sim('stage3_rigidbody_checks','StopTime','2','ReturnWorkspaceOutputs','on');
sim_data=simout.stage3_simulink_output; data=squeeze(sim_data.Data); sim_pass=all(isfinite(data),'all') && abs(data(end,1))<1e-9 && data(end,2)>0 && max(data(:,3))>0 && max(abs(data(:,4)))>0;
close_system('stage3_rigidbody_checks',0);
summary=struct('matlab_checks_passed',all(rows.passed),'simulink_executed',true,'simulink_checks_passed',sim_pass, ...
    'peak_contact_force_n',max(result.contact_force),'max_penetration_m',max(result.penetration),'max_attitude_deg',max(abs(result.euler),[],'all')*180/pi, ...
    'motor_saturation_fraction',mean(result.saturated));
if ~isempty(comparison), summary.python_matlab=comparison(1,:); end
fid=fopen(fullfile(output_dir,'stage3_crosscheck_summary.json'),'w'); fprintf(fid,'%s',jsonencode(summary,PrettyPrint=true)); fclose(fid);
end
