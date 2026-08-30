function S = stage2_secondary_analysis(run_dir)
% Post-hoc diagnostics; no policy threshold or feature changes.
R=readtable(fullfile(run_dir,'validation_predictions.csv'),'TextType','string'); S=table();
for noise=["low","nominal","high"]
    Q=R(R.noise_regime==noise,:); dx=abs(Q.predicted_peak_displacement-Q.actual_peak_displacement)./max(abs(Q.actual_peak_displacement),1e-9);
    st=abs(Q.predicted_settling-Q.actual_settling);
    row=table(noise,mean(Q.predicted_risk==Q.actual_risk),median(dx),qbase(dx,.95),median(st),qbase(st,.95), ...
        median(Q.k_relative_error),qbase(Q.k_relative_error,.95),median(Q.c_relative_error),qbase(Q.c_relative_error,.95),median(Q.m_relative_error),qbase(Q.m_relative_error,.95), ...
        'VariableNames',{'noise_regime','three_class_accuracy','peak_displacement_median_relative_error','peak_displacement_p95_relative_error','settling_median_absolute_error_s','settling_p95_absolute_error_s', ...
        'k_median_relative_error','k_p95_relative_error','c_median_relative_error','c_p95_relative_error','m_median_relative_error','m_p95_relative_error'}); S=[S;row]; %#ok<AGROW>
end
writetable(S,fullfile(run_dir,'secondary_summary.csv'));
end
function q=qbase(x,p), x=sort(x(:)); pos=1+(numel(x)-1)*p; lo=floor(pos); hi=ceil(pos); if lo==hi,q=x(lo);else,q=x(lo)+(pos-lo)*(x(hi)-x(lo));end; end
