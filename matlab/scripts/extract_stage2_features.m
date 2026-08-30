function [feature,diagnostic,sensed] = extract_stage2_features(case_row,truth,cfg,noise_index)
% Independently reconstruct the locked causal chirp+0.5 s feature vector.
mult=cfg.noise_multiplier(noise_index); ns=cfg.base_noise;
noise=struct('displacement',mult*ns.displacement,'velocity',mult*ns.velocity, ...
    'acceleration',mult*ns.acceleration,'force',mult*ns.force);
ts=(0:1/cfg.sensor_fs:cfg.probe.duration+cfg.probe.observation)';
x=interp1(truth.time,truth.x,ts,'linear'); v=interp1(truth.time,truth.v,ts,'linear');
a=interp1(truth.time,truth.a,ts,'linear'); F=interp1(truth.time,truth.force,ts,'linear');
measurement_seed=case_row.seed*100000+case_row.case_index*100+(noise_index-1);
sensed=causal_sensor(ts,x,v,a,F,noise,measurement_seed,cfg.filter.cutoff_hz);
% Same raw displacement realization, independently processed by the documented
% alpha-beta-gamma stiffness diagnostic.
[abgx,abgv,abga]=abg(sensed.raw_displacement,1/cfg.sensor_fs,0.40,0.08,0.008);
trim=ts>=0.15 & ts<=cfg.probe.duration-0.15; cmd=interp1(truth.time,truth.command,ts,'linear'); cmd=max(cmd(trim),0);
mx=sensed.displacement(trim); mv=sensed.velocity(trim); ma=sensed.acceleration(trim); mf=sensed.force(trim);
A=[mx mv ma]; ols=A\mf;
At=[abgx(trim) abgv(trim) abga(trim)]; aug=[At mf]; scales=sqrt(mean(aug.^2,1)); scales(scales<realmin)=1;
[~,~,V]=svd(aug./scales,0); last=V(:,end); tls=-last(1:3)./last(4).*scales(4)./scales(1:3)';
delays=[0 .025 .050 .075 .100]; Z=zeros(numel(cmd),numel(delays)); tm=ts(trim);
for j=1:numel(delays), Z(:,j)=interp1(tm,cmd,tm-delays(j),'linear',cmd(1)); end
Z=Z-mean(Z,1); zn=sqrt(sum(Z.^2,1)); keep=zn>realmin; Z=Z(:,keep)./zn(keep);
Ap=Z*(Z\A); iv=Ap\mf; if rank(Ap)<3 || any(~isfinite(iv)), iv=ols; end
stiff=tls(1); if ~isfinite(stiff), stiff=ols(1); end
diagnostic=struct('estimated_k',stiff,'estimated_c',ols(2),'estimated_m',iv(3), ...
    'k_relative_error',abs(stiff-case_row.k)/case_row.k,'c_relative_error',abs(ols(2)-case_row.c)/case_row.c, ...
    'm_relative_error',abs(iv(3)-case_row.m)/case_row.m,'measurement_seed',measurement_seed);
names={}; values=[]; add=@append;
    function append(name,value), names{end+1}=name; values(end+1)=value; end %#ok<AGROW>
add('estimated_stiffness_signed_log',signedlog(stiff,100));
% Frequency response features.
xx=mx-mean(mx); ff=mf-mean(mf); N=numel(xx); freq=(0:floor(N/2))'/(N*(tm(2)-tm(1)));
X=fft(xx); U=fft(ff); X=X(1:numel(freq)); U=U(1:numel(freq)); floorU=.05*max(abs(U));
den=U; den(abs(U)<floorU)=floorU; H=X./den; gain=abs(H); band=freq>=.5 & freq<=5;
bi=find(band); [~,p]=max(gain(band)); peak=bi(p); energy=abs(X(band)).^2; centroid=sum(freq(band).*energy)/max(sum(energy),eps);
low=median(gain(freq>=.5 & freq<1.5)); mid=median(gain(freq>=1.5 & freq<3)); high=median(gain(freq>=3 & freq<5.01)); phase=angle(H(peak));
add('fr_peak_gain_log',plog(gain(peak))); add('fr_peak_frequency_hz',freq(peak)); add('fr_centroid_hz',centroid);
add('fr_low_gain_log',plog(low)); add('fr_mid_gain_log',plog(mid)); add('fr_high_gain_log',plog(high));
add('fr_high_low_log_ratio',log(max(high,1e-12)/max(low,1e-12))); add('fr_peak_phase_sin',sin(phase)); add('fr_peak_phase_cos',cos(phase));
add('probe_dominant_frequency_hz',dominant(tm,xx));
% Time-domain chirp features.
free=cmd<=1e-12; fx=mx(free); ft=tm(free); midpoint=median(ft); early=sqrt(mean(fx(ft<=midpoint).^2)); late=sqrt(mean(fx(ft>midpoint).^2)); persistence=late/max(early,1e-12);
peakF=max(max(abs(mf)),1e-12); rmsF=max(sqrt(mean(mf.^2)),1e-12); work=trapz(tm,abs(mf.*mv)); crossings=sum(signbit_local(mv(1:end-1))~=signbit_local(mv(2:end)));
add('probe_peak_displacement_log',plog(max(abs(mx)))); add('probe_rms_displacement_log',plog(sqrt(mean(mx.^2))));
add('probe_peak_velocity_log',plog(max(abs(mv)))); add('probe_rms_velocity_log',plog(sqrt(mean(mv.^2))));
add('probe_peak_acceleration_log',plog(max(abs(ma)))); add('probe_rms_acceleration_log',plog(sqrt(mean(ma.^2))));
add('probe_free_displacement_rms_log',plog(sqrt(mean(fx.^2)))); add('probe_persistence_ratio_log',log(max(persistence,1e-12)));
add('probe_peak_gain_log',plog(max(abs(mx))/peakF)); add('probe_rms_gain_log',plog(sqrt(mean(mx.^2))/rmsF));
add('probe_absolute_work_log',plog(work)); add('probe_final_displacement_signed_log',signedlog(mx(end),1e-3)); add('probe_velocity_zero_crossing_log',log1p(crossings));
% Causal ring-down prefix.
ring=ring_features(sensed,cfg.probe.duration,cfg.probe.observation,cfg,noise.displacement);
rn=fieldnames(ring); for j=1:numel(rn), add(rn{j},ring.(rn{j})); end
feature=struct('names',{names},'values',values,'noise_multiplier',mult,'observation_duration',cfg.probe.observation,'ring',ring);
end

function [x,v,a]=abg(y,dt,alpha,beta,gamma)
n=numel(y); x=zeros(n,1); v=zeros(n,1); a=zeros(n,1); x(1)=y(1);
for i=2:n
    xp=x(i-1)+dt*v(i-1)+.5*dt^2*a(i-1); vp=v(i-1)+dt*a(i-1); r=y(i)-xp;
    x(i)=xp+alpha*r; v(i)=vp+beta*r/dt; a(i)=a(i-1)+2*gamma*r/dt^2;
end
end
function out=ring_features(s,probe_end,duration,cfg,noise_floor)
mask=s.time>=probe_end & s.time<=probe_end+duration+1e-12; t=s.time(mask); x=s.displacement(mask); v=s.velocity(mask); dt=t(2)-t(1);
[centers,env]=blockrms(t,x,cfg.ring.block_s); [decay,r2]=decayfit(centers,env,1.5*noise_floor,cfg.ring.min_points);
centered=x-mean(x); freq=(0:floor(numel(x)/2))'/(numel(x)*dt); Y=fft(centered); Y=abs(Y(1:numel(freq))); dom=0;
if numel(Y)>1 && max(Y(2:end))>realmin, [~,q]=max(Y(2:end)); dom=freq(q+1); end
thr=cfg.ring.peak_noise_multiplier*noise_floor; p=find(centered(2:end-1)>centered(1:end-2) & centered(2:end-1)>=centered(3:end) & centered(2:end-1)>thr)+1; peaks=centered(p);
decs=[]; if numel(peaks)>=2, decs=log(peaks(1:end-1)./peaks(2:end)); decs=decs(isfinite(decs)&decs>0); end
decrement=0; if ~isempty(decs), decrement=median(decs); end; zeta=0; if decrement>0, zeta=decrement/sqrt((2*pi)^2+decrement^2); end
tail=max(round(cfg.ring.tail_s/dt),2); tx=x(end-tail+1:end); tv=v(end-tail+1:end); initial=min(tail,numel(x));
respeak=max(abs(tx)); resrms=sqrt(mean(tx.^2)); initialx=sqrt(mean(x(1:initial).^2)); initialv=sqrt(mean(v(1:initial).^2)); tailv=sqrt(mean(tv.^2));
below=abs(x)<=cfg.ring.x_threshold & abs(v)<=cfg.ring.v_threshold; dwell=0; for i=numel(below):-1:1, if ~below(i), break; end; dwell=dwell+1; end
threshold_dwell=dwell*dt; time_to=max(duration-threshold_dwell,0); crossings=sum(signbit_local(centered(1:end-1))~=signbit_local(centered(2:end)));
energy=x.^2+(v/(2*pi)).^2; [ec,ee]=blockrms(t,sqrt(energy),cfg.ring.block_s); [ed,~]=decayfit(ec,ee,max(noise_floor,1e-12),cfg.ring.min_points);
firstE=mean(energy(1:initial)); lastE=mean(energy(end-initial+1:end));
out=struct('rd_observation_duration_s',duration,'rd_valid',1,'rd_decay_rate_log',log1p(decay),'rd_decay_fit_r2',r2, ...
 'rd_decay_time_constant_log',log1p(1/max(decay,1e-6)),'rd_dominant_frequency_hz',dom,'rd_log_decrement_log',log1p(decrement), ...
 'rd_damping_ratio',zeta,'rd_peak_count_log',log1p(numel(peaks)),'rd_residual_peak_log',plog(respeak),'rd_residual_rms_log',plog(resrms), ...
 'rd_displacement_ratio_log',log(max(resrms,1e-12)/max(initialx,1e-12)),'rd_velocity_ratio_log',log(max(tailv,1e-12)/max(initialv,1e-12)), ...
 'rd_threshold_reached',double(dwell>0),'rd_time_to_threshold_fraction',time_to/max(duration,1e-12), ...
 'rd_threshold_dwell_fraction',threshold_dwell/max(duration,1e-12),'rd_zero_crossing_rate_log',log1p(crossings/max(duration,1e-12)), ...
 'rd_energy_decay_rate_log',log1p(2*ed),'rd_energy_ratio_log',log(max(lastE,1e-18)/max(firstE,1e-18)));
end
function [centers,rmsv]=blockrms(t,y,block)
r=t-t(1); count=max(ceil((r(end)+eps)/block),1); centers=[]; rmsv=[];
for i=0:count-1
    if i<count-1, mask=r>=i*block & r<(i+1)*block; else, mask=r>=i*block & r<=(i+1)*block; end
    if sum(mask)>=2, centers(end+1,1)=mean(r(mask)); rmsv(end+1,1)=sqrt(mean(y(mask).^2)); end %#ok<AGROW>
end
end
function [rate,r2]=decayfit(x,y,floorv,minpts)
ok=y>max(floorv,realmin); if sum(ok)<minpts, rate=0; r2=0; return; end
A=[ones(sum(ok),1) x(ok)]; b=log(y(ok)); q=A\b; pred=A*q; ss=sum((b-pred).^2); total=sum((b-mean(b)).^2);
rate=max(-q(2),0); if total>1e-15, r2=max(0,min(1,1-ss/total)); else, r2=0; end
end
function f=dominant(t,y)
N=numel(y); z=abs(fft(y-mean(y))); z=z(1:floor(N/2)+1); freq=(0:floor(N/2))'/(N*(t(2)-t(1))); f=0;
if numel(z)>1 && max(z(2:end))>realmin, [~,i]=max(z(2:end)); f=freq(i+1); end
end
function y=plog(x), y=log(max(abs(x),1e-12)); end
function y=signedlog(x,s), y=sign(x)*log1p(abs(x)/s); end
function y=signbit_local(x), y=x<0; end
