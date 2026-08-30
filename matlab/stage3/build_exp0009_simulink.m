function modelPath = build_exp0009_simulink()
%BUILD_EXP0009_SIMULINK Subsystem model for independent normal-contact replay.
model='exp0009_hybrid_crosscheck'; here=fileparts(mfilename('fullpath')); modelPath=fullfile(here,[model '.slx']);
if bdIsLoaded(model),close_system(model,0);end
new_system(model);set_param(model,'Solver','ode4','SolverType','Fixed-step','FixedStep','exp9_dt');
add_block('simulink/Sources/From Workspace',[model '/Hybrid Command'],'VariableName','exp9_hybrid_command','Position',[25 120 130 155]);
add_block('simulink/Signal Routing/Demux',[model '/Command Split'],'Outputs','2','Position',[155 105 160 170]);
normal_controller(model);uav_dynamics(model);contact_interface(model);target_dynamics(model);
add_block('simulink/Signal Routing/Demux',[model '/UAV Split'],'Outputs','2','Position',[500 85 505 145]);
add_block('simulink/Signal Routing/Demux',[model '/Target Split'],'Outputs','2','Position',[745 260 750 320]);
add_line(model,'Hybrid Command/1','Command Split/1');add_line(model,'Command Split/1','Normal Force Controller/1');add_line(model,'Command Split/2','Normal Force Controller/2');
add_line(model,'Normal Force Controller/1','UAV Normal Dynamics/1');add_line(model,'Contact Interface/1','UAV Normal Dynamics/2');
add_line(model,'UAV Normal Dynamics/1','UAV Split/1');add_line(model,'UAV Split/1','Normal Force Controller/3');add_line(model,'UAV Split/2','Normal Force Controller/4');
add_line(model,'UAV Split/1','Contact Interface/1');add_line(model,'UAV Split/2','Contact Interface/2');
add_line(model,'Target Dynamics/1','Target Split/1');add_line(model,'Target Split/1','Contact Interface/3');add_line(model,'Target Split/2','Contact Interface/4');add_line(model,'Contact Interface/1','Target Dynamics/1');
add_block('simulink/Signal Routing/Mux',[model '/Evidence'],'Inputs','4','Position',[910 105 915 270]);
add_line(model,'UAV Split/1','Evidence/1');add_line(model,'Target Split/1','Evidence/2');add_line(model,'Contact Interface/1','Evidence/3');add_line(model,'Command Split/1','Evidence/4');
add_block('simulink/Sinks/To Workspace',[model '/Output'],'VariableName','exp9_simulink_output','SaveFormat','Timeseries','Position',[960 175 1065 205]);add_line(model,'Evidence/1','Output/1');
save_system(model,modelPath);close_system(model,0);
end

function normal_controller(model)
p=[model '/Normal Force Controller'];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[200 80 365 195]);Simulink.SubSystem.deleteContents(p);
for i=1:4,add_block('simulink/Ports & Subsystems/In1',[p '/In' num2str(i)],'Port',num2str(i),'Position',[10 5+28*i 40 25+28*i]);end
add_block('simulink/Math Operations/Sum',[p '/Position Error'],'Inputs','+-','Position',[70 45 95 75]);add_block('simulink/Math Operations/Gain',[p '/Position Gain'],'Gain','exp9_mass*exp9_kp','Position',[115 40 180 70]);
add_block('simulink/Math Operations/Gain',[p '/Velocity Gain'],'Gain','-exp9_mass*exp9_kd','Position',[115 90 180 120]);add_block('simulink/Math Operations/Sum',[p '/Sum'],'Inputs','+++','Position',[215 55 240 115]);add_block('simulink/Ports & Subsystems/Out1',[p '/Propulsion'],'Position',[280 75 310 95]);
add_line(p,'In2/1','Position Error/1');add_line(p,'In3/1','Position Error/2');add_line(p,'Position Error/1','Position Gain/1');add_line(p,'In4/1','Velocity Gain/1');add_line(p,'In1/1','Sum/1');add_line(p,'Position Gain/1','Sum/2');add_line(p,'Velocity Gain/1','Sum/3');add_line(p,'Sum/1','Propulsion/1');
end
function uav_dynamics(model)
p=[model '/UAV Normal Dynamics'];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[390 65 475 155]);Simulink.SubSystem.deleteContents(p);add_block('simulink/Ports & Subsystems/In1',[p '/Propulsion'],'Port','1','Position',[10 20 40 40]);add_block('simulink/Ports & Subsystems/In1',[p '/Contact'],'Port','2','Position',[10 60 40 80]);add_block('simulink/Math Operations/Sum',[p '/Net'],'Inputs','+-','Position',[65 30 90 70]);add_block('simulink/Continuous/Transfer Fcn',[p '/Motor Lag'],'Numerator','1','Denominator','[exp9_motor_tau 1]','Position',[115 25 180 55]);add_block('simulink/Math Operations/Gain',[p '/Inverse Mass'],'Gain','1/exp9_mass','Position',[205 25 265 55]);add_block('simulink/Continuous/Integrator',[p '/Velocity'],'Position',[290 25 320 55]);add_block('simulink/Continuous/Integrator',[p '/Position'],'InitialCondition','exp9_initial_tip','Position',[345 25 375 55]);add_block('simulink/Signal Routing/Mux',[p '/States'],'Inputs','2','Position',[400 15 405 70]);add_block('simulink/Ports & Subsystems/Out1',[p '/Output'],'Position',[440 35 470 55]);add_line(p,'Propulsion/1','Net/1');add_line(p,'Contact/1','Net/2');add_line(p,'Net/1','Motor Lag/1');add_line(p,'Motor Lag/1','Inverse Mass/1');add_line(p,'Inverse Mass/1','Velocity/1');add_line(p,'Velocity/1','Position/1');add_line(p,'Position/1','States/1');add_line(p,'Velocity/1','States/2');add_line(p,'States/1','Output/1');
end
function contact_interface(model)
p=[model '/Contact Interface'];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[600 115 710 225]);Simulink.SubSystem.deleteContents(p);for i=1:4,add_block('simulink/Ports & Subsystems/In1',[p '/In' num2str(i)],'Port',num2str(i),'Position',[10 5+25*i 40 25+25*i]);end;add_block('simulink/Math Operations/Sum',[p '/Penetration'],'Inputs','+-','Position',[70 30 95 60]);add_block('simulink/Math Operations/Sum',[p '/Closing'],'Inputs','+-','Position',[70 85 95 115]);add_block('simulink/Math Operations/Gain',[p '/K'],'Gain','exp9_contact_k','Position',[120 25 170 55]);add_block('simulink/Math Operations/Gain',[p '/C'],'Gain','exp9_contact_c','Position',[120 85 170 115]);add_block('simulink/Math Operations/Sum',[p '/Raw'],'Inputs','++','Position',[200 50 225 95]);add_block('simulink/Discontinuities/Saturation',[p '/Unilateral'],'LowerLimit','0','UpperLimit','inf','Position',[250 55 305 90]);add_block('simulink/Ports & Subsystems/Out1',[p '/Force'],'Position',[335 65 365 85]);add_line(p,'In1/1','Penetration/1');add_line(p,'In3/1','Penetration/2');add_line(p,'In2/1','Closing/1');add_line(p,'In4/1','Closing/2');add_line(p,'Penetration/1','K/1');add_line(p,'Closing/1','C/1');add_line(p,'K/1','Raw/1');add_line(p,'C/1','Raw/2');add_line(p,'Raw/1','Unilateral/1');add_line(p,'Unilateral/1','Force/1');
end
function target_dynamics(model)
p=[model '/Target Dynamics'];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[600 270 710 335]);Simulink.SubSystem.deleteContents(p);add_block('simulink/Ports & Subsystems/In1',[p '/Force'],'Position',[10 25 40 45]);add_block('simulink/Continuous/State-Space',[p '/MSD'],'A','[0 1;-exp9_target_k/exp9_target_m -exp9_target_c/exp9_target_m]','B','[0;1/exp9_target_m]','C','eye(2)','D','[0;0]','X0','[0;0]','Position',[70 10 190 60]);add_block('simulink/Ports & Subsystems/Out1',[p '/States'],'Position',[220 25 250 45]);add_line(p,'Force/1','MSD/1');add_line(p,'MSD/1','States/1');
end
