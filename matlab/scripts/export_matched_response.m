function path = export_matched_response(run_dir,repository_root)
% Export representative MATLAB trajectories for post-hoc Python comparison.
cfg=stage2_config(repository_root); cases=readtable(fullfile(run_dir,'validation_cases.csv'),'TextType','string'); cases=cases([1 961],:);
[~,up]=chirp_probe(1/cfg.dt,cfg.probe.duration,cfg.probe.f0,cfg.probe.f1,cfg.probe.amplitude); t=(0:cfg.dt:cfg.probe.duration+cfg.probe.observation)'; u=zeros(size(t));u(1:numel(up))=up;
S=simulate_msd_population(cases,t,u,true); T=table();
for i=1:height(cases)
    n=numel(t); Q=table(repmat(cases.target_id(i),n,1),repmat(cases.k(i),n,1),repmat(cases.c(i),n,1),repmat(cases.m(i),n,1),t,S.x(i,:)',S.v(i,:)',S.a(i,:)',S.force, ...
      'VariableNames',{'target_id','k','c','m','time_s','matlab_x','matlab_v','matlab_a','force'}); T=[T;Q]; %#ok<AGROW>
end
path=fullfile(run_dir,'matched_matlab_response.csv'); writetable(T,path);
end
