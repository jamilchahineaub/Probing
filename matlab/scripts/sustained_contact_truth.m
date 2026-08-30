function truth = sustained_contact_truth(cases,cfg)
% Labels derive only from the hidden 2 N sustained-contact maneuver.
t=(0:cfg.dt:cfg.maneuver.duration)'; F=zeros(size(t)); A=cfg.maneuver.force;
r=t<=cfg.maneuver.ramp_up; F(r)=0.5*A*(1-cos(pi*t(r)/cfg.maneuver.ramp_up));
h=t>cfg.maneuver.ramp_up & t<=cfg.maneuver.hold_end; F(h)=A;
d=t>cfg.maneuver.hold_end & t<=cfg.maneuver.ramp_down_end;
F(d)=0.5*A*(1+cos(pi*(t(d)-cfg.maneuver.hold_end)/(cfg.maneuver.ramp_down_end-cfg.maneuver.hold_end)));
sim=simulate_msd_population(cases,t,F,true); nc=height(cases);
peak_x=zeros(nc,1); peak_v=peak_x; osc=peak_x; settling=peak_x; severity=peak_x; risk=strings(nc,1);
forced=t<=cfg.maneuver.hold_end; late=t>=cfg.maneuver.hold_end-1 & t<=cfg.maneuver.hold_end;
eligible=find(t>=cfg.maneuver.ramp_up & t<=cfg.maneuver.hold_end);
for i=1:nc
    x=sim.x(i,:)'; v=sim.v(i,:)'; peak_x(i)=max(abs(x(forced))); peak_v(i)=max(abs(v(forced)));
    center=mean(x(late)); osc(i)=sqrt(mean((x(late)-center).^2)); tol=max(cfg.settling_fraction*abs(center),cfg.settling_floor);
    ok=abs(x-center)<=tol & abs(v)<=cfg.settling_velocity; settling(i)=cfg.maneuver.hold_end-cfg.maneuver.ramp_up;
    for j=eligible'
        if all(ok(j:eligible(end))), settling(i)=t(j)-cfg.maneuver.ramp_up; break; end
    end
    safe=peak_x(i)<=cfg.safe.peak_displacement && peak_v(i)<=cfg.safe.peak_velocity && osc(i)<=cfg.safe.oscillation && settling(i)<=cfg.safe.settling && peak_x(i)<=cfg.contact_tracking_displacement;
    unsafe=peak_x(i)>cfg.unsafe.peak_displacement || peak_v(i)>cfg.unsafe.peak_velocity || osc(i)>cfg.unsafe.oscillation || settling(i)>cfg.unsafe.settling || peak_x(i)>cfg.contact_tracking_displacement;
    if safe, risk(i)="SAFE"; elseif unsafe, risk(i)="UNSAFE"; else, risk(i)="CAUTION"; end
    severity(i)=max([peak_x(i)/cfg.safe.peak_displacement,peak_v(i)/cfg.safe.peak_velocity,osc(i)/cfg.safe.oscillation,settling(i)/cfg.safe.settling]);
end
truth=table(cases.target_id,peak_x,peak_v,osc,settling,risk,severity, ...
    'VariableNames',{'target_id','peak_displacement','peak_velocity','oscillation','settling_time','risk_class','severity'});
end
