function prediction = locked_stage1_decision(feature,policy,cfg)
% Apply the frozen EXP-0006 policy bundle without fitting or calibration.
assert(isequal(string(feature.names),string(policy.feature_names)'),'Feature definitions/order differ from locked policy');
z=(feature.values(:)'-policy.normalization.mean(:)')./policy.normalization.scale(:)';
outcomes={'peak_displacement_m','peak_velocity_m_per_s','late_hold_oscillation_rms_m','hold_settling_time_s'};
medianv=zeros(1,4); upper=zeros(1,4);
for j=1:4
    reg=policy.regressors.(outcomes{j}); coef=reg.coefficients(:); q=coef(1)+z*coef(2:end);
    qu=q+reg.upper_residual; if j==4, medianv(j)=max(expm1(max(min(q,30),-30)),0); upper(j)=max(expm1(max(min(qu,30),-30)),0);
    else, medianv(j)=exp(max(min(q,30),-30)); upper(j)=exp(max(min(qu,30),-30)); end
end
safe_limit=[cfg.safe.peak_displacement cfg.safe.peak_velocity cfg.safe.oscillation cfg.safe.settling];
unsafe_limit=[cfg.unsafe.peak_displacement cfg.unsafe.peak_velocity cfg.unsafe.oscillation cfg.unsafe.settling];
sr=upper./safe_limit; ur=upper./unsafe_limit; contact=upper(1)>cfg.contact_tracking_displacement;
if max(sr)<=1 && ~contact, label="SAFE"; elseif max(ur)>1 || contact, label="UNSAFE"; else, label="CAUTION"; end
decay=expm1(feature.ring.rd_decay_rate_log); residual=exp(feature.ring.rd_residual_rms_log); noise=cfg.base_noise.displacement*feature.noise_multiplier;
veto=feature.observation_duration>=.5 && decay<=.5 && residual>=3*noise && feature.ring.rd_threshold_dwell_fraction<=.10;
if label=="SAFE" && veto, label="CAUTION"; end
prediction=struct('risk_class',label,'binary',ternary(label=="SAFE","SAFE","NON_SAFE"),'median',medianv,'upper',upper, ...
    'risk_score',max(sr),'veto',veto,'contact_loss',contact);
end
function x=ternary(condition,a,b), if condition, x=a; else, x=b; end; end
