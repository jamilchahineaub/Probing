function generate_stage2_figures(run_dir,cfg,physics,cases,rows,actual,summary,t,command,probe,simcmp,repository_root)
% Publication-oriented Stage 2 figures (PNG and PDF).
out=fullfile(run_dir,'figures'); [~,run_id]=fileparts(run_dir); colors=lines(3);
% 1 analytical validation.
tt=(0:1e-3:3)'; k=100;c=4;m=1;x0=.01;v0=.02; wn=sqrt(k/m);z=c/(2*sqrt(k*m));wd=wn*sqrt(1-z^2); xa=exp(-z*wn*tt).*(x0*cos(wd*tt)+(v0+z*wn*x0)/wd*sin(wd*tt)); [~,xn]=msd_rk4(k,c,m,tt,zeros(size(tt)),x0,v0);
f=figure('Visible','off'); plot(tt,xa,'k-',tt,xn,'r--','LineWidth',1.2); xlabel('Time (s)'); ylabel('Displacement (m)'); legend('Analytical','MATLAB RK4'); title(sprintf('Independent physics validation; max error %.2g m',max(abs(xa-xn(:))))); grid on; savefig2(f,'analytical_validation');
% 2 chirp time and spectrum/instantaneous frequency.
f=figure('Visible','off'); tiledlayout(2,1); nexttile; plot(t,command,'LineWidth',1); xline(cfg.probe.duration,'--'); ylabel('Command (N)'); title('Locked bounded chirp plus zero-force observation'); grid on;
nexttile; active=t<=cfg.probe.duration; inst=cfg.probe.f0+(cfg.probe.f1-cfg.probe.f0)*t(active)/cfg.probe.duration; plot(t(active),inst,'LineWidth',1.2); xlabel('Time (s)'); ylabel('Instantaneous frequency (Hz)'); ylim([0 5.5]); grid on; savefig2(f,'chirp_time_frequency');
% Representatives under nominal sensing.
for kind=["SAFE","NON_SAFE"]
    if kind=="SAFE", i=find(actual.risk_class=="SAFE",1); else, i=find(actual.risk_class~="SAFE",1); end
    tr=struct('time',t,'command',command,'force',probe.force,'x',probe.x(i,:)','v',probe.v(i,:)','a',probe.a(i,:)'); [~,~,s]=extract_stage2_features(cases(i,:),tr,cfg,2);
    f=figure('Visible','off'); yyaxis left; plot(s.time,1e3*s.displacement,'Color',colors(1,:),'LineWidth',1); ylabel('Measured displacement (mm)'); yyaxis right; plot(t,probe.force,'Color',colors(2,:)); ylabel('Contact force (N)'); xline(3,'--'); xline(3.5,':'); xlabel('Time (s)'); title(sprintf('%s target: chirp and passive ring-down',strrep(kind,'_','-'))); grid on; savefig2(f,['representative_' lower(char(kind))]);
end
% 5 binary confusion nominal/high.
f=figure('Visible','off'); tiledlayout(1,2); for q=1:2, noise=["nominal","high"]; R=rows(rows.noise_regime==noise(q),:); A=R.actual_binary=="NON_SAFE"; P=R.predicted_binary=="NON_SAFE"; M=[sum(~A&~P) sum(~A&P);sum(A&~P) sum(A&P)]; nexttile; imagesc(M); colormap(f,'parula'); text([1 2 1 2],[1 1 2 2],string(M(:)),'HorizontalAlignment','center','Color','w','FontWeight','bold'); xticks(1:2);xticklabels({'SAFE','NON-SAFE'});yticks(1:2);yticklabels({'SAFE','NON-SAFE'});xlabel('Predicted');ylabel('Actual');title(noise(q)); end; savefig2(f,'binary_confusion');
% 6 false-safe confidence.
S=summary(summary.stratum=="overall",:); f=figure('Visible','off'); bar(1:3,100*S.false_safe_rate); hold on; er=errorbar(1:3,100*S.false_safe_rate,zeros(3,1),100*(S.false_safe_upper95-S.false_safe_rate),'k.'); er.CapSize=8; xticks(1:3); xticklabels(S.noise_regime); yline(1,'--','Nominal limit'); yline(2,':','High-noise limit'); ylabel('False-safe rate (%)'); title('MATLAB false-safe rate and one-sided 95% upper bound'); grid on; savefig2(f,'false_safe_confidence');
% 7 high-noise target decision map.
R=rows(rows.noise_regime=="high",:); [~,ix]=ismember(R.target_id,cases.target_id); f=figure('Visible','off'); hold on; sa=R.actual_binary=="SAFE"; scatter(log10(cases.k(ix(sa))),log10(cases.c(ix(sa))),12,[0 .55 0],'filled','MarkerFaceAlpha',.45); scatter(log10(cases.k(ix(~sa))),log10(cases.c(ix(~sa))),12,[.75 0 0],'filled','MarkerFaceAlpha',.45); bad=R.false_safe; scatter(log10(cases.k(ix(bad))),log10(cases.c(ix(bad))),80,'ko','LineWidth',1.5); xlabel('log10 k');ylabel('log10 c');title('High-noise decisions across target dynamics');legend('Actual SAFE','Actual NON-SAFE','False-safe');grid on; savefig2(f,'target_decision_map');
% 8 boundary severity and decisions.
R=rows(rows.noise_regime=="high" & rows.stratum=="boundary",:); [~,ix]=ismember(R.target_id,actual.target_id); f=figure('Visible','off'); scatter(actual.severity(ix),R.predicted_risk_score,28,R.predicted_binary=="SAFE",'filled'); xline(1,'--');yline(1,'--');xlabel('Actual SAFE-envelope severity ratio');ylabel('Predicted bound ratio');title('Boundary-enriched high-noise behavior');colorbar;grid on; savefig2(f,'boundary_behavior');
% 9 EXP-0007 versus MATLAB.
nom=summary(summary.noise_regime=="nominal"&summary.stratum=="overall",:); hi=summary(summary.noise_regime=="high"&summary.stratum=="overall",:); vals=100*[.0019193858 nom.false_safe_rate;.0038387716 hi.false_safe_rate;.8875 nom.binary_accuracy;.8904411765 hi.binary_accuracy];
f=figure('Visible','off'); bar(vals); xticklabels({'Nominal false-safe','High false-safe','Nominal accuracy','High accuracy'}); ylabel('Percent'); legend('Python EXP-0007','MATLAB Stage 2'); title('Cross-population implementation summary');grid on; savefig2(f,'python_exp0007_vs_matlab');
% 10 MATLAB versus Simulink integration errors.
f=figure('Visible','off'); semilogy(1:height(simcmp),simcmp.max_x_error,'o-',1:height(simcmp),simcmp.max_v_error,'s-',1:height(simcmp),simcmp.max_a_error,'^-'); xlabel('Representative target');ylabel('Maximum absolute error');legend('x','v','a');title('MATLAB RK4 versus Simulink ode4');grid on; savefig2(f,'matlab_simulink_agreement');
writetable(physics,fullfile(repository_root,'results','tables',[run_id '__physics_validation.csv']));
    function savefig2(fig,name)
        set(fig,'Color','w','Position',[100 100 900 520]); ax=findall(fig,'Type','axes'); set(ax,'Color','w','XColor','k','YColor','k','GridColor',[.65 .65 .65]); txt=findall(fig,'Type','text'); set(txt,'Color','k');
        base=fullfile(out,[run_id '__' name]); exportgraphics(fig,[base '.png'],'Resolution',180); exportgraphics(fig,[base '.pdf'],'ContentType','vector'); close(fig);
    end
end
