function summary = run_exp0009_crosscheck(outputDir,pythonStiffCsv,pythonFailureCsv)
%RUN_EXP0009_CROSSCHECK Independent MATLAB/Simulink EXP-0009 reconstruction.
if ~exist(outputDir,'dir'),mkdir(outputDir);end
cfg=exp0009_hybrid_config(); stiff=struct('k',2417.1213919510114,'c',0.5786344665375269,'m',1.1631213761211359); failure=struct('k',213.81074228148387,'c',6.729686778366037,'m',2.1475131326644616);
stiffResult=exp0009_hybrid_simulate(cfg,stiff); failureResult=exp0009_hybrid_simulate(cfg,failure);
write_result(fullfile(outputDir,'matlab_stiff_hybrid.csv'),stiffResult);write_result(fullfile(outputDir,'matlab_acquisition_failure.csv'),failureResult);
transitionRows=[transition_table(stiffResult,"stiff");transition_table(failureResult,"acquisition_failure")];writetable(transitionRows,fullfile(outputDir,'matlab_state_transitions.csv'));
pythonStiff=readtable(pythonStiffCsv);pythonFailure=readtable(pythonFailureCsv);
stiffComparison=compare_case(stiffResult,pythonStiff);failureComparison=compare_case(failureResult,pythonFailure);
writetable([addvars(stiffComparison,"stiff",Before=1,NewVariableNames="case");addvars(failureComparison,"acquisition_failure",Before=1,NewVariableNames="case")],fullfile(outputDir,'python_matlab_exp0009_comparison.csv'));

modelPath=build_exp0009_simulink();assignin('base','exp9_dt',cfg.dt);assignin('base','exp9_mass',cfg.mass);assignin('base','exp9_kp',cfg.kp(1));assignin('base','exp9_kd',cfg.kd(1));assignin('base','exp9_motor_tau',cfg.motorTau);assignin('base','exp9_initial_tip',-cfg.clearance);assignin('base','exp9_contact_k',cfg.contactK);assignin('base','exp9_contact_c',cfg.contactC);assignin('base','exp9_target_k',stiff.k);assignin('base','exp9_target_c',stiff.c);assignin('base','exp9_target_m',stiff.m);
command=timeseries([stiffResult.force_command,stiffResult.tip_command],stiffResult.time);assignin('base','exp9_hybrid_command',command);load_system(modelPath);simout=sim('exp0009_hybrid_crosscheck','StopTime',num2str(stiffResult.time(end)),'ReturnWorkspaceOutputs','on');data=squeeze(simout.exp9_simulink_output.Data);simTime=simout.exp9_simulink_output.Time;close_system('exp0009_hybrid_crosscheck',0);
simTable=table(simTime,data(:,1),data(:,2),data(:,3),data(:,4),'VariableNames',{'time_s','tip_coordinate_m','target_x_m','contact_force_n','force_command_n'});writetable(simTable,fullfile(outputDir,'simulink_stiff_hybrid.csv'));copyfile(modelPath,fullfile(outputDir,'exp0009_hybrid_crosscheck.slx'));
matlabForce=interp1(stiffResult.time,stiffResult.contact_force,simTime);matlabTarget=interp1(stiffResult.time,stiffResult.state(:,18),simTime);
simForceRms=sqrt(mean((data(:,3)-matlabForce).^2));simTargetMax=max(abs(data(:,2)-matlabTarget));
stiffPhase=string(stiffResult.phase(end));failurePhase=string(failureResult.phase(end));
summary=struct('matlab_release',version,'matlab_stiff_terminal_phase',stiffPhase,'python_stiff_terminal_phase',string(pythonStiff.phase(end)), ...
    'matlab_failure_terminal_phase',failurePhase,'python_failure_terminal_phase',string(pythonFailure.phase(end)), ...
    'completion_pattern_agrees',stiffPhase=="DECISION" && string(pythonStiff.phase(end))=="DECISION" && failurePhase=="ABORT" && string(pythonFailure.phase(end))=="ABORT", ...
    'matlab_stiff_peak_force_n',max(stiffResult.contact_force),'python_stiff_peak_force_n',max(pythonStiff.realized_contact_force_n), ...
    'matlab_stiff_zero_probe_separation',all(stiffResult.contact_active(stiffResult.phase=="PROBE")), ...
    'simulink_executed',true,'simulink_all_finite',all(isfinite(data),'all'),'matlab_simulink_force_rms_difference_n',simForceRms, ...
    'matlab_simulink_target_max_difference_m',simTargetMax,'simulink_scope','normal-axis dynamics replay of independently generated MATLAB hybrid commands');
fid=fopen(fullfile(outputDir,'exp0009_matlab_simulink_summary.json'),'w');fprintf(fid,'%s',jsonencode(summary,PrettyPrint=true));fclose(fid);
end

function write_result(path,result)
output=table(result.time,result.phase,result.reference_force,result.force_command,result.contact_force,result.contact_active,result.tip_command,result.state(:,1),result.state(:,18),result.state(:,19),result.euler(:,2),result.actuator_reserve,result.saturated, ...
    'VariableNames',{'time_s','phase','reference_force_n','force_command_n','contact_force_n','contact_active','tip_command_m','uav_x_m','target_x_m','target_v_mps','pitch_rad','actuator_reserve','saturated'});writetable(output,path);
end
function output=transition_table(result,label)
count=numel(result.transition_time);from=strings(count,1);to=strings(count,1);for i=1:count,from(i)=result.phase(result.transition_from(i));to(i)=result.phase(result.transition_to(i));end;output=table(repmat(string(label),count,1),result.transition_time,from,to,'VariableNames',{'case','time_s','from_phase','to_phase'});
end
function output=compare_case(matlabResult,python)
common=matlabResult.time(matlabResult.time>=max(matlabResult.time(1),python.time_s(1)) & matlabResult.time<=min(matlabResult.time(end),python.time_s(end)));matlabForce=interp1(matlabResult.time,matlabResult.contact_force,common);pythonForce=interp1(python.time_s,python.realized_contact_force_n,common);matlabTarget=interp1(matlabResult.time,matlabResult.state(:,18),common);pythonTarget=interp1(python.time_s,python.target_displacement_m,common);output=table(max(abs(matlabForce-pythonForce)),sqrt(mean((matlabForce-pythonForce).^2)),max(abs(matlabTarget-pythonTarget)),string(matlabResult.phase(end)),string(python.phase(end)),'VariableNames',{'maximum_force_difference_n','rms_force_difference_n','maximum_target_difference_m','matlab_terminal_phase','python_terminal_phase'});
end
