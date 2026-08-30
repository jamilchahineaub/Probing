function model_path = build_stage3_simulink()
%BUILD_STAGE3_SIMULINK Independent subsystem-level rigid-body sign model.
model='stage3_rigidbody_checks'; here=fileparts(mfilename('fullpath')); model_path=fullfile(here,[model '.slx']);
if bdIsLoaded(model), close_system(model,0); end
new_system(model); set_param(model,'Solver','ode4','SolverType','Fixed-step','FixedStep','stage3_dt');
make_hover(model); make_offset_rotation(model);
make_coupled_normal(model);
add_block('simulink/Sources/From Workspace',[model '/Probe Command'],'VariableName','stage3_force_input','Position',[40 390 135 420]);
add_block('simulink/Signal Routing/Demux',[model '/UAV x v'],'Outputs','2','Position',[410 380 415 440]);
add_block('simulink/Signal Routing/Demux',[model '/Target x v'],'Outputs','2','Position',[650 430 655 490]);
add_line(model,'Probe Command/1','Normal Force Controller/1');
add_line(model,'Normal Force Controller/1','UAV Normal Dynamics/1'); add_line(model,'Contact Interface/1','UAV Normal Dynamics/2');
add_line(model,'UAV Normal Dynamics/1','UAV x v/1'); add_line(model,'UAV x v/1','Normal Force Controller/2'); add_line(model,'UAV x v/2','Normal Force Controller/3');
add_line(model,'UAV x v/1','Contact Interface/1'); add_line(model,'UAV x v/2','Contact Interface/2');
add_line(model,'Target Dynamics/1','Target x v/1'); add_line(model,'Target x v/1','Contact Interface/3'); add_line(model,'Target x v/2','Contact Interface/4');
add_line(model,'Contact Interface/1','Target Dynamics/1');
add_block('simulink/Signal Routing/Mux',[model '/Validation signals'],'Inputs','4','Position',[900 135 905 390]);
add_block('simulink/Sinks/To Workspace',[model '/Output'],'VariableName','stage3_simulink_output','SaveFormat','Timeseries','Position',[840 180 945 210]);
set_param([model '/Output'],'Position',[970 235 1075 265]);
add_line(model,'Vertical Rigid Body/1','Validation signals/1'); add_line(model,'Offset Contact Rotation/1','Validation signals/2');
add_line(model,'Contact Interface/1','Validation signals/3'); add_line(model,'Target x v/1','Validation signals/4'); add_line(model,'Validation signals/1','Output/1');
save_system(model,model_path); close_system(model,0);
end
function make_coupled_normal(model)
p=[model '/Normal Force Controller']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[170 350 340 455]); Simulink.SubSystem.deleteContents(p);
add_block('simulink/Ports & Subsystems/In1',[p '/Force reference'],'Port','1','Position',[15 20 45 40]);
add_block('simulink/Ports & Subsystems/In1',[p '/Position'],'Port','2','Position',[15 55 45 75]);
add_block('simulink/Ports & Subsystems/In1',[p '/Velocity'],'Port','3','Position',[15 90 45 110]);
add_block('simulink/Math Operations/Gain',[p '/Position feedback'],'Gain','-stage3_mass*stage3_normal_kp','Position',[80 50 145 80]);
add_block('simulink/Math Operations/Gain',[p '/Velocity feedback'],'Gain','-stage3_mass*stage3_normal_kd','Position',[80 90 145 120]);
add_block('simulink/Math Operations/Sum',[p '/Propulsion sum'],'Inputs','+++','Position',[180 45 205 105]);
add_block('simulink/Ports & Subsystems/Out1',[p '/Propulsion'],'Position',[245 65 275 85]);
add_line(p,'Force reference/1','Propulsion sum/1');add_line(p,'Position/1','Position feedback/1');add_line(p,'Velocity/1','Velocity feedback/1');add_line(p,'Position feedback/1','Propulsion sum/2');add_line(p,'Velocity feedback/1','Propulsion sum/3');add_line(p,'Propulsion sum/1','Propulsion/1');

p=[model '/UAV Normal Dynamics']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[460 325 590 405]); Simulink.SubSystem.deleteContents(p);
add_block('simulink/Ports & Subsystems/In1',[p '/Propulsion'],'Port','1','Position',[10 20 40 40]);add_block('simulink/Ports & Subsystems/In1',[p '/Contact'],'Port','2','Position',[10 60 40 80]);
add_block('simulink/Math Operations/Sum',[p '/Balance'],'Inputs','+-','Position',[70 30 95 70]);add_block('simulink/Math Operations/Gain',[p '/Inverse mass'],'Gain','1/stage3_mass','Position',[120 35 180 65]);
add_block('simulink/Continuous/Integrator',[p '/Velocity'],'Position',[205 35 235 65]);add_block('simulink/Continuous/Integrator',[p '/Position'],'Position',[260 35 290 65]);
add_block('simulink/Signal Routing/Mux',[p '/x v'],'Inputs','2','Position',[325 25 330 80]);add_block('simulink/Ports & Subsystems/Out1',[p '/States'],'Position',[370 45 400 65]);
add_line(p,'Propulsion/1','Balance/1');add_line(p,'Contact/1','Balance/2');add_line(p,'Balance/1','Inverse mass/1');add_line(p,'Inverse mass/1','Velocity/1');add_line(p,'Velocity/1','Position/1');add_line(p,'Position/1','x v/1');add_line(p,'Velocity/1','x v/2');add_line(p,'x v/1','States/1');

p=[model '/Contact Interface']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[710 340 835 445]); Simulink.SubSystem.deleteContents(p);
for i=1:4, add_block('simulink/Ports & Subsystems/In1',[p '/In' num2str(i)],'Port',num2str(i),'Position',[10 10+30*i 40 30+30*i]); end
add_block('simulink/Math Operations/Sum',[p '/Penetration'],'Inputs','+-','Position',[70 35 95 65]);add_block('simulink/Math Operations/Sum',[p '/Closing speed'],'Inputs','+-','Position',[70 100 95 130]);
add_block('simulink/Math Operations/Gain',[p '/Contact stiffness'],'Gain','stage3_contact_k','Position',[125 30 195 60]);add_block('simulink/Math Operations/Gain',[p '/Contact damping'],'Gain','stage3_contact_c','Position',[125 100 195 130]);
add_block('simulink/Math Operations/Sum',[p '/Raw force'],'Inputs','++','Position',[225 60 250 105]);add_block('simulink/Discontinuities/Saturation',[p '/No tension'],'LowerLimit','0','UpperLimit','inf','Position',[280 65 330 100]);add_block('simulink/Ports & Subsystems/Out1',[p '/Force'],'Position',[365 75 395 95]);
add_line(p,'In1/1','Penetration/1');add_line(p,'In3/1','Penetration/2');add_line(p,'In2/1','Closing speed/1');add_line(p,'In4/1','Closing speed/2');add_line(p,'Penetration/1','Contact stiffness/1');add_line(p,'Closing speed/1','Contact damping/1');add_line(p,'Contact stiffness/1','Raw force/1');add_line(p,'Contact damping/1','Raw force/2');add_line(p,'Raw force/1','No tension/1');add_line(p,'No tension/1','Force/1');

p=[model '/Target Dynamics']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[710 500 835 580]); Simulink.SubSystem.deleteContents(p);
add_block('simulink/Ports & Subsystems/In1',[p '/Force'],'Position',[10 45 40 65]);add_block('simulink/Continuous/State-Space',[p '/Target MSD'],'A','[0 1;-stage3_target_k/stage3_target_m -stage3_target_c/stage3_target_m]','B','[0;1/stage3_target_m]','C','eye(2)','D','[0;0]','X0','[0;0]','Position',[80 25 205 75]);add_block('simulink/Ports & Subsystems/Out1',[p '/States'],'Position',[245 45 275 65]);add_line(p,'Force/1','Target MSD/1');add_line(p,'Target MSD/1','States/1');
end
function make_hover(model)
p=[model '/Vertical Rigid Body']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[80 70 330 180]); Simulink.SubSystem.deleteContents(p);
add_block('simulink/Sources/Constant',[p '/Hover thrust'],'Value','stage3_hover_thrust','Position',[20 25 80 55]);
add_block('simulink/Sources/Constant',[p '/Weight'],'Value','-stage3_mass*stage3_g','Position',[20 85 80 115]);
add_block('simulink/Math Operations/Sum',[p '/Force sum'],'Inputs','++','Position',[120 50 145 100]);
add_block('simulink/Math Operations/Gain',[p '/Inverse mass'],'Gain','1/stage3_mass','Position',[180 55 245 95]);
add_block('simulink/Continuous/Integrator',[p '/Velocity'],'InitialCondition','0','Position',[280 55 310 85]);
add_block('simulink/Continuous/Integrator',[p '/Position'],'InitialCondition','0','Position',[345 55 375 85]);
add_block('simulink/Ports & Subsystems/Out1',[p '/z'],'Position',[420 60 450 80]);
add_line(p,'Hover thrust/1','Force sum/1');add_line(p,'Weight/1','Force sum/2');add_line(p,'Force sum/1','Inverse mass/1');add_line(p,'Inverse mass/1','Velocity/1');add_line(p,'Velocity/1','Position/1');add_line(p,'Position/1','z/1');
end
function make_offset_rotation(model)
p=[model '/Offset Contact Rotation']; add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',[410 220 680 335]); Simulink.SubSystem.deleteContents(p);
add_block('simulink/Sources/Constant',[p '/Contact pitch moment'],'Value','-stage3_probe_z*stage3_contact_force','Position',[20 45 130 75]);
add_block('simulink/Math Operations/Gain',[p '/Inverse Jyy'],'Gain','1/stage3_Jyy','Position',[175 45 240 75]);
add_block('simulink/Continuous/Integrator',[p '/Pitch rate'],'InitialCondition','0','Position',[280 45 310 75]);
add_block('simulink/Continuous/Integrator',[p '/Pitch angle'],'InitialCondition','0','Position',[350 45 380 75]);
add_block('simulink/Ports & Subsystems/Out1',[p '/pitch'],'Position',[425 50 455 70]);
add_line(p,'Contact pitch moment/1','Inverse Jyy/1');add_line(p,'Inverse Jyy/1','Pitch rate/1');add_line(p,'Pitch rate/1','Pitch angle/1');add_line(p,'Pitch angle/1','pitch/1');
end
