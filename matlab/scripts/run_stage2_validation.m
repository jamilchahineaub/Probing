function result = run_stage2_validation(repository_root)
%RUN_STAGE2_VALIDATION One-shot independent MATLAB Stage 2 reproduction.
if nargin==0, repository_root=fileparts(fileparts(fileparts(mfilename('fullpath')))); end
addpath(genpath(fullfile(repository_root,'matlab'))); cfg=stage2_config(repository_root); policy=jsondecode(fileread(cfg.policy_file));
stamp=char(datetime('now','TimeZone','UTC','Format','yyyyMMdd''T''HHmmss.SSS''Z''')); run_id=['STAGE2_' stamp '_s20101'];
run_dir=fullfile(repository_root,'runs',run_id); assert(~exist(run_dir,'dir'),'Run directory exists'); mkdir(run_dir); mkdir(fullfile(run_dir,'figures'));
environment=check_environment(); [~,git_sha]=system(sprintf('git -C "%s" rev-parse HEAD',repository_root)); git_sha=strtrim(git_sha);
physics=validate_stage2_physics(); writetable(physics,fullfile(run_dir,'physics_validation.csv'));
% Population construction is locked before any policy prediction.
broad=generate_stage2_population(cfg.population.broad_seeds,cfg.population.cases_per_seed,cfg.bounds,'matlab_broad'); broad.stratum=repmat("broad",height(broad),1);
candidates=generate_stage2_population(cfg.population.boundary_seeds,cfg.population.cases_per_seed,cfg.bounds,'matlab_boundary_candidate');
candidate_truth=sustained_contact_truth(candidates,cfg); distance=abs(log(max(candidate_truth.severity,1e-12)));
safe=find(candidate_truth.risk_class=="SAFE"); nonsafe=find(candidate_truth.risk_class~="SAFE"); [~,si]=sort(distance(safe)); [~,ni]=sort(distance(nonsafe));
selected=[safe(si(1:cfg.population.boundary_safe));nonsafe(ni(1:cfg.population.boundary_nonsafe))]; boundary=candidates(selected,:); boundary.target_id="matlab_boundary_"+string((1:height(boundary))'); boundary.stratum=repmat("boundary",height(boundary),1);
cases=[broad;boundary]; writetable(cases,fullfile(run_dir,'validation_cases.csv'));
boundary_selection=table(candidates.target_id(selected),boundary.target_id,candidate_truth.risk_class(selected),candidate_truth.severity(selected),distance(selected), ...
    'VariableNames',{'source_target_id','target_id','risk_class','severity','log_boundary_distance'}); writetable(boundary_selection,fullfile(run_dir,'boundary_selection.csv'));
% Locked bounded probe + passive observation.
[tp,fp]=chirp_probe(1/cfg.dt,cfg.probe.duration,cfg.probe.f0,cfg.probe.f1,cfg.probe.amplitude); t=(0:cfg.dt:cfg.probe.duration+cfg.probe.observation)'; command=zeros(size(t)); command(1:numel(fp))=fp;
probe=simulate_msd_population(cases,t,command,true); probe_peak_x=max(abs(probe.x),[],2); probe_peak_v=max(abs(probe.v),[],2); probe_peak_a=max(abs(probe.a),[],2);
safety=probe_peak_x>cfg.probe_limits.displacement | probe_peak_v>cfg.probe_limits.velocity | probe_peak_a>cfg.probe_limits.acceleration | max(abs(probe.force))>cfg.probe_limits.force;
safety_events=table(cases.target_id(safety),probe_peak_x(safety),probe_peak_v(safety),probe_peak_a(safety),'VariableNames',{'target_id','peak_displacement','peak_velocity','peak_acceleration'}); writetable(safety_events,fullfile(run_dir,'probe_safety_events.csv'));
N=height(cases); P=N*3; target_id=strings(P,1); stratum=strings(P,1); noise_regime=strings(P,1); predicted_risk=strings(P,1); predicted_binary=strings(P,1);
estimated_k=zeros(P,1); estimated_c=zeros(P,1); estimated_m=zeros(P,1); k_error=zeros(P,1); c_error=zeros(P,1); m_error=zeros(P,1); risk_score=zeros(P,1); veto=false(P,1); predicted=zeros(P,4); upper=zeros(P,4); q=0;
% Predictions are fully completed before final sustained-contact outcomes exist.
for i=1:N
    truth=struct('time',t,'command',command,'force',probe.force,'x',probe.x(i,:)','v',probe.v(i,:)','a',probe.a(i,:)');
    for ni=1:3
        q=q+1; [feat,diag]=extract_stage2_features(cases(i,:),truth,cfg,ni); pred=locked_stage1_decision(feat,policy,cfg);
        target_id(q)=cases.target_id(i); stratum(q)=cases.stratum(i); noise_regime(q)=cfg.noise_names{ni}; predicted_risk(q)=pred.risk_class; predicted_binary(q)=pred.binary;
        estimated_k(q)=diag.estimated_k; estimated_c(q)=diag.estimated_c; estimated_m(q)=diag.estimated_m; k_error(q)=diag.k_relative_error; c_error(q)=diag.c_relative_error; m_error(q)=diag.m_relative_error;
        risk_score(q)=pred.risk_score; veto(q)=pred.veto; predicted(q,:)=pred.median; upper(q,:)=pred.upper;
    end
    if mod(i,100)==0, fprintf('Stage 2 features %d/%d\n',i,N); end
end
rows=table(target_id,stratum,noise_regime,predicted_risk,predicted_binary,estimated_k,estimated_c,estimated_m,k_error,c_error,m_error,risk_score,veto, ...
    predicted(:,1),predicted(:,2),predicted(:,3),predicted(:,4),upper(:,1),upper(:,2),upper(:,3),upper(:,4), ...
    'VariableNames',{'target_id','stratum','noise_regime','predicted_risk','predicted_binary','estimated_k','estimated_c','estimated_m','k_relative_error','c_relative_error','m_relative_error','predicted_risk_score','persistence_veto', ...
    'predicted_peak_displacement','predicted_peak_velocity','predicted_oscillation','predicted_settling','upper_peak_displacement','upper_peak_velocity','upper_oscillation','upper_settling'});
% Hidden labels are generated and joined only now.
actual=sustained_contact_truth(cases,cfg); idx=repelem((1:N)',3); rows.actual_risk=actual.risk_class(idx); rows.actual_binary=repmat("NON_SAFE",P,1); rows.actual_binary(rows.actual_risk=="SAFE")="SAFE";
rows.actual_peak_displacement=actual.peak_displacement(idx); rows.actual_peak_velocity=actual.peak_velocity(idx); rows.actual_oscillation=actual.oscillation(idx); rows.actual_settling=actual.settling_time(idx); rows.false_safe=rows.actual_binary=="NON_SAFE" & rows.predicted_binary=="SAFE";
writetable(rows,fullfile(run_dir,'validation_predictions.csv')); writetable(actual,fullfile(run_dir,'sustained_contact_truth.csv'));
summary=table(); for ni=1:3, for s=["overall","broad","boundary"], rr=rows; if s=="overall", rr.stratum(:)="overall"; end; summary=[summary;stage2_metrics(rr,s,cfg.noise_names{ni})]; end; end %#ok<AGROW>
writetable(summary,fullfile(run_dir,'metrics_summary.csv'));
simulink_comparison=simulink_crosscheck(cases([1 161 321 641 961 1161],:),cfg); writetable(simulink_comparison,fullfile(run_dir,'matlab_simulink_crosscheck.csv'));
nom=summary(summary.noise_regime=="nominal" & summary.stratum=="overall",:); high=summary(summary.noise_regime=="high" & summary.stratum=="overall",:); bn=summary(summary.noise_regime=="nominal" & summary.stratum=="boundary",:); bh=summary(summary.noise_regime=="high" & summary.stratum=="boundary",:);
sim_ok=max(simulink_comparison.max_x_error)<1e-9 && max(simulink_comparison.max_v_error)<1e-8 && max(simulink_comparison.max_a_error)<1e-6;
gate=height(cases)>=1300 && isempty(safety_events) && nom.false_safe_rate<=cfg.gate.nominal_false_safe && high.false_safe_rate<=cfg.gate.high_false_safe && nom.false_safe_upper95<=cfg.gate.nominal_upper && high.false_safe_upper95<=cfg.gate.high_upper && nom.binary_accuracy>=cfg.gate.nominal_accuracy && high.binary_accuracy>=cfg.gate.high_accuracy && nom.safe_precision>=cfg.gate.nominal_precision && high.safe_precision>=cfg.gate.high_precision && nom.non_safe_recall>=cfg.gate.nominal_recall && high.non_safe_recall>=cfg.gate.high_recall && bn.false_safe_rate<=cfg.gate.boundary_false_safe && bh.false_safe_rate<=cfg.gate.boundary_false_safe && bn.false_safe_upper95<=cfg.gate.boundary_upper && bh.false_safe_upper95<=cfg.gate.boundary_upper && bn.binary_accuracy>=cfg.gate.boundary_accuracy && bh.binary_accuracy>=cfg.gate.boundary_accuracy && sim_ok;
result=struct('run_id',run_id,'stage2_pass',gate,'stage2_decision',ternary(gate,'PASS','CONTINUE'),'target_count',N,'prediction_count',P,'safety_event_count',height(safety_events), ...
    'predictions_before_outcomes',true,'matlab_simulink_agree',sim_ok,'git_sha',git_sha,'matlab_version',version,'policy_id',policy.policy_id,'policy_sha256',sha256_file(cfg.policy_file));
result.metrics=table2struct(summary); result.environment=environment; result.config=cfg;
fid=fopen(fullfile(run_dir,'summary.json'),'w'); fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true)); fclose(fid); save(fullfile(run_dir,'raw_workspace.mat'),'cases','rows','actual','probe','cfg','result','physics','simulink_comparison','-v7.3');
generate_stage2_figures(run_dir,cfg,physics,cases,rows,actual,summary,t,command,probe,simulink_comparison,repository_root);
copyfile(fullfile(run_dir,'figures','*'),fullfile(repository_root,'results','figures')); copyfile(fullfile(run_dir,'metrics_summary.csv'),fullfile(repository_root,'results','tables',[run_id '__metrics_summary.csv']));
fid=fopen(fullfile(run_dir,'manifest.json'),'w'); fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true)); fclose(fid); disp(jsonencode(result));
end
function h=sha256_file(path), [ok,out]=system(sprintf('sha256sum "%s"',path)); assert(ok==0); p=strsplit(strtrim(out)); h=p{1}; end
function x=ternary(condition,a,b), if condition, x=a; else, x=b; end; end
