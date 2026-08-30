function model_path = build_stage2_simulink()
%BUILD_STAGE2_SIMULINK Construct the independently integrated Stage 2 model.
model='stage2_locked_interaction'; here=fileparts(mfilename('fullpath')); model_path=fullfile(here,[model '.slx']);
if bdIsLoaded(model), close_system(model,0); end
new_system(model); set_param(model,'Solver','ode4','SolverType','Fixed-step','FixedStep','stage2_dt','StopTime','stage2_stop');
add_block('simulink/Sources/From Workspace',[model '/Locked probe input'],'VariableName','probe_input','Position',[25 85 120 115]);
make_passthrough(model,'Probe Command',[155 65 270 135]); make_target(model,[305 45 455 155]);
make_passthrough(model,'Sensors and Noise',[500 45 620 155]); make_passthrough(model,'Causal Signal Processing',[665 45 810 155]);
make_passthrough(model,'Feature Extraction',[855 45 975 155]); make_passthrough(model,'SAFE NON-SAFE Decision',[1020 45 1165 155]);
make_passthrough(model,'Sustained Contact Ground Truth',[1210 45 1370 155]);
add_block('simulink/Sinks/To Workspace',[model '/Plant output'],'VariableName','simulink_output','SaveFormat','Timeseries','Position',[1415 85 1505 115]);
blocks={'Locked probe input','Probe Command','Target Dynamics','Sensors and Noise','Causal Signal Processing','Feature Extraction','SAFE NON-SAFE Decision','Sustained Contact Ground Truth','Plant output'};
for i=1:numel(blocks)-1, add_line(model,[blocks{i} '/1'],[blocks{i+1} '/1'],'autorouting','on'); end
save_system(model,model_path); close_system(model,0);
end
function make_passthrough(model,name,pos)
path=[model '/' name]; add_block('simulink/Ports & Subsystems/Subsystem',path,'Position',pos);
Simulink.SubSystem.deleteContents(path); add_block('simulink/Ports & Subsystems/In1',[path '/In']); add_block('simulink/Ports & Subsystems/Out1',[path '/Out']); add_line(path,'In/1','Out/1');
end
function make_target(model,pos)
path=[model '/Target Dynamics']; add_block('simulink/Ports & Subsystems/Subsystem',path,'Position',pos); Simulink.SubSystem.deleteContents(path);
add_block('simulink/Ports & Subsystems/In1',[path '/Force'],'Position',[20 95 50 115]);
add_block('simulink/Continuous/State-Space',[path '/Mass spring damper'],'A','[0 1;-stage2_k/stage2_m -stage2_c/stage2_m]','B','[0;1/stage2_m]','C','eye(2)','D','[0;0]','X0','[0;0]','Position',[95 40 210 90]);
add_block('simulink/Signal Routing/Demux',[path '/x v'],'Outputs','2','Position',[245 35 250 95]); add_block('simulink/Math Operations/Gain',[path '/minus k'],'Gain','-stage2_k','Position',[285 25 345 55]);
add_block('simulink/Math Operations/Gain',[path '/minus c'],'Gain','-stage2_c','Position',[285 75 345 105]); add_block('simulink/Math Operations/Sum',[path '/force balance'],'Inputs','+++','Position',[390 65 420 115]);
add_block('simulink/Math Operations/Gain',[path '/inverse mass'],'Gain','1/stage2_m','Position',[455 75 525 105]); add_block('simulink/Signal Routing/Mux',[path '/x v a F'],'Inputs','4','Position',[565 40 570 135]);
add_block('simulink/Ports & Subsystems/Out1',[path '/States'],'Position',[620 80 650 100]);
add_line(path,'Force/1','Mass spring damper/1'); add_line(path,'Mass spring damper/1','x v/1'); add_line(path,'x v/1','minus k/1'); add_line(path,'x v/2','minus c/1');
add_line(path,'minus k/1','force balance/1'); add_line(path,'minus c/1','force balance/2'); add_line(path,'Force/1','force balance/3'); add_line(path,'force balance/1','inverse mass/1');
add_line(path,'x v/1','x v a F/1'); add_line(path,'x v/2','x v a F/2'); add_line(path,'inverse mass/1','x v a F/3'); add_line(path,'Force/1','x v a F/4'); add_line(path,'x v a F/1','States/1');
end
