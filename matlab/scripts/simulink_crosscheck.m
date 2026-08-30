function result = simulink_crosscheck(cases,cfg)
% Compare independent MATLAB RK4 against Simulink ode4 on representatives.
model_path=build_stage2_simulink(); load_system(model_path); n=min(6,height(cases)); rows=table(); policy=jsondecode(fileread(cfg.policy_file));
[~,up]=chirp_probe(1/cfg.dt,cfg.probe.duration,cfg.probe.f0,cfg.probe.f1,cfg.probe.amplitude); t=(0:cfg.dt:cfg.probe.duration+cfg.probe.observation)'; command=zeros(size(t)); command(1:numel(up))=up; u=max(command,0);
for i=1:n
    stage2_dt=cfg.dt; stage2_stop=t(end); stage2_k=cases.k(i); stage2_c=cases.c(i); stage2_m=cases.m(i); probe_input=timeseries(u,t);
    assignin('base','stage2_dt',stage2_dt); assignin('base','stage2_stop',stage2_stop); assignin('base','stage2_k',stage2_k); assignin('base','stage2_c',stage2_c); assignin('base','stage2_m',stage2_m); assignin('base','probe_input',probe_input);
    out=sim('stage2_locked_interaction','ReturnWorkspaceOutputs','on'); y=out.simulink_output; sy=squeeze(y.Data); st=y.Time;
    ms=simulate_msd_population(cases(i,:),t,command,true); mx=interp1(t,ms.x(1,:)',st); mv=interp1(t,ms.v(1,:)',st); ma=interp1(t,ms.a(1,:)',st);
    mt=struct('time',t,'command',command,'force',ms.force,'x',ms.x(1,:)','v',ms.v(1,:)','a',ms.a(1,:)'); [mf,~]=extract_stage2_features(cases(i,:),mt,cfg,2); mp=locked_stage1_decision(mf,policy,cfg);
    scmd=interp1(t,command,st); stt=struct('time',st,'command',scmd,'force',sy(:,4),'x',sy(:,1),'v',sy(:,2),'a',sy(:,3)); [sf,~]=extract_stage2_features(cases(i,:),stt,cfg,2); sp=locked_stage1_decision(sf,policy,cfg);
    row=table(cases.target_id(i),max(abs(sy(:,1)-mx)),max(abs(sy(:,2)-mv)),max(abs(sy(:,3)-ma)),mp.binary,sp.binary,mp.binary==sp.binary,'VariableNames',{'target_id','max_x_error','max_v_error','max_a_error','matlab_decision','simulink_decision','decision_match'}); rows=[rows;row]; %#ok<AGROW>
end
close_system('stage2_locked_interaction',0); result=rows;
end
