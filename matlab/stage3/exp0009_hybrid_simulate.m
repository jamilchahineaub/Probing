function result = exp0009_hybrid_simulate(cfg,target)
%EXP0009_HYBRID_SIMULATE Independent full rigid-body hybrid-contact reconstruction.
t=(0:cfg.dt:cfg.duration)'; n=numel(t); alloc=allocation(cfg);
hover=sqrt(cfg.mass*cfg.g/(4*cfg.kT)); initialPosition=cfg.surface-cfg.probeOffset-cfg.clearance*cfg.normal;
y=[initialPosition;zeros(3,1);1;0;0;0;zeros(3,1);hover*ones(4,1);0;0];
Y=zeros(n,19); force=zeros(n,1); penetration=zeros(n,1); torque=zeros(n,3);
reference=zeros(n,1); variation=zeros(n,1); forceCommand=zeros(n,1); tipCommand=zeros(n,1);
phaseCode=zeros(n,1); contactActive=false(n,1); reserve=zeros(n,1); saturated=false(n,1);
phase=1; phaseStart=0; dwell=0; commandedTip=-cfg.clearance; commandedTipVelocity=0;
previousReference=0; filteredReferenceDerivative=0; lastForceCommand=0;
unloadStartForce=0; unloadStartTip=commandedTip; transitionTime=[]; transitionFrom=[]; transitionTo=[];
for i=1:n
    [contact,vehicle,structure]=contact_eval(y,cfg,target); attitude=rotm2euler(quatrot(vehicle.q)); now=t(i); elapsed=now-phaseStart;
    oldPhase=phase;
    if phase~=7 && phase~=8
        if ~all(isfinite(y)) || contact.force>cfg.abortForce || contact.penetration>cfg.abortPenetration || max(abs(attitude))*180/pi>cfg.abortAttitudeDeg
            phase=8;
        elseif phase==1 && contact.force>=cfg.detectForce
            phase=2;
        elseif phase==2
            stable=elapsed>=cfg.acquireRamp && contact.force>0 && abs(contact.force-cfg.acquireForce)<=cfg.acquireError && abs(contact.closing)<=cfg.acquireVelocity;
            if stable, dwell=dwell+cfg.dt; else, dwell=0; end
            if dwell>=cfg.acquireDwell, phase=3; elseif elapsed>=cfg.acquireTimeout, phase=8; end
        elseif phase==3
            stable=elapsed>=cfg.preloadRamp && contact.force>0 && abs(contact.force-cfg.preload)<=cfg.preloadError && abs(contact.closing)<=cfg.preloadVelocity;
            if stable, dwell=dwell+cfg.dt; else, dwell=0; end
            if dwell>=cfg.preloadDwell, phase=4; elseif elapsed>=cfg.preloadTimeout, phase=8; end
        elseif phase==4 && elapsed>=cfg.probeDuration
            unloadStartForce=lastForceCommand; unloadStartTip=commandedTip; phase=5;
        elseif phase==5 && elapsed>=cfg.unloadDuration
            phase=6; commandedTip=-cfg.passiveRetraction; commandedTipVelocity=0;
        elseif phase==6 && elapsed>=cfg.observeDuration
            phase=7;
        end
    end
    if phase~=oldPhase
        transitionTime(end+1,1)=now; transitionFrom(end+1,1)=oldPhase; transitionTo(end+1,1)=phase; %#ok<AGROW>
        phaseStart=now; elapsed=0; dwell=0;
    end
    [ref,var]=force_reference(phase,elapsed,cfg,unloadStartForce);
    rawDerivative=(ref-previousReference)/cfg.dt; alpha=cfg.dt/(cfg.referenceDerivativeTau+cfg.dt);
    filteredReferenceDerivative=filteredReferenceDerivative+alpha*(rawDerivative-filteredReferenceDerivative); previousReference=ref;
    if phase==1
        desiredTip=commandedTip+cfg.approachVelocity*cfg.dt; normalForce=0;
    elseif any(phase==[2 3 4])
        forceError=ref-contact.force;
        if contact.force<=0
            desiredTip=commandedTip+cfg.approachVelocity*cfg.dt; normalForce=0;
        else
            desiredVelocity=structure.v+filteredReferenceDerivative/cfg.contactK;
            if any(phase==[2 3]), desiredVelocity=desiredVelocity+cfg.compressionVelocity*max(forceError/max(ref,1e-6),0); end
            desiredVelocity=min(max(desiredVelocity,-cfg.maxNormalVelocity),cfg.maxNormalVelocity);
            desiredTip=commandedTip+desiredVelocity*cfg.dt;
            normalForce=min(max(ref,cfg.minimumForceCommand),cfg.maximumForceCommand);
        end
    elseif phase==5
        fraction=min(max(elapsed/cfg.unloadDuration,0),1); smooth=0.5-0.5*cos(pi*fraction);
        desiredTip=(1-smooth)*unloadStartTip-smooth*cfg.passiveRetraction; normalForce=ref;
    else
        desiredTip=commandedTip; normalForce=0;
    end
    maxVelocity=cfg.maxNormalVelocity; if phase==1, maxVelocity=cfg.approachMax; end
    targetVelocity=min(max((desiredTip-commandedTip)/cfg.dt,-maxVelocity),maxVelocity);
    delta=cfg.maxNormalAcceleration*cfg.dt; commandedTipVelocity=min(max(targetVelocity,commandedTipVelocity-delta),commandedTipVelocity+delta);
    if phase==1, commandedTipVelocity=min(max(commandedTipVelocity,0),cfg.approachMax); end
    commandedTip=commandedTip+commandedTipVelocity*cfg.dt; lastForceCommand=normalForce;
    pref=cfg.surface-cfg.probeOffset+commandedTip*cfg.normal;
    [omegaCommand,reserve(i),saturated(i)]=controller(vehicle,pref,normalForce,contact.torque,cfg,alloc);
    Y(i,:)=y'; force(i)=contact.force; penetration(i)=contact.penetration; torque(i,:)=contact.torque';
    reference(i)=ref; variation(i)=var; forceCommand(i)=normalForce; tipCommand(i)=commandedTip;
    phaseCode(i)=phase; contactActive(i)=contact.active;
    if phase==7 || phase==8, Y=Y(1:i,:); t=t(1:i); force=force(1:i); penetration=penetration(1:i); torque=torque(1:i,:); reference=reference(1:i); variation=variation(1:i); forceCommand=forceCommand(1:i); tipCommand=tipCommand(1:i); phaseCode=phaseCode(1:i); contactActive=contactActive(1:i); reserve=reserve(1:i); saturated=saturated(1:i); break; end
    f=@(state) coupled_derivative(state,omegaCommand,cfg,target);
    k1=f(y); k2=f(y+0.5*cfg.dt*k1); k3=f(y+0.5*cfg.dt*k2); k4=f(y+cfg.dt*k3);
    y=y+cfg.dt*(k1+2*k2+2*k3+k4)/6; y(7:10)=y(7:10)/norm(y(7:10)); y(14:17)=min(max(y(14:17),cfg.omegaMin),cfg.omegaMax);
end
euler=zeros(numel(t),3); targetAcceleration=zeros(numel(t),1); rotorThrust=zeros(numel(t),4);
for i=1:numel(t)
    euler(i,:)=rotm2euler(quatrot(Y(i,7:10)'))'; targetAcceleration(i)=(force(i)-target.c*Y(i,19)-target.k*Y(i,18))/target.m; rotorThrust(i,:)=cfg.kT*Y(i,14:17).^2;
end
result=struct('time',t,'state',Y,'phase_code',phaseCode,'phase',cfg.phaseNames(phaseCode)', ...
    'reference_force',reference,'variation_force',variation,'force_command',forceCommand, ...
    'tip_command',tipCommand,'contact_force',force,'contact_active',contactActive, ...
    'penetration',penetration,'contact_torque',torque,'euler',euler, ...
    'target_acceleration',targetAcceleration,'rotor_thrust',rotorThrust, ...
    'actuator_reserve',reserve,'saturated',saturated,'transition_time',transitionTime, ...
    'transition_from',transitionFrom,'transition_to',transitionTo);
end

function [reference,variation]=force_reference(phase,elapsed,cfg,unloadStart)
variation=0;
if phase==2
    fraction=min(max(elapsed/cfg.acquireRamp,0),1); smooth=0.5-0.5*cos(pi*fraction); reference=cfg.detectForce+smooth*(cfg.acquireForce-cfg.detectForce);
elseif phase==3
    fraction=min(max(elapsed/cfg.preloadRamp,0),1); smooth=0.5-0.5*cos(pi*fraction); reference=cfg.acquireForce+smooth*(cfg.preload-cfg.acquireForce);
elseif phase==4
    phaseValue=2*pi*(cfg.probeF0*elapsed+0.5*(cfg.probeF1-cfg.probeF0)/cfg.probeDuration*elapsed^2); variation=max(cfg.probeAmplitude*sin(phaseValue),0); reference=cfg.preload+variation;
elseif phase==5
    fraction=min(max(elapsed/cfg.unloadDuration,0),1); reference=unloadStart*(0.5+0.5*cos(pi*fraction));
else
    reference=0;
end
end

function dy=coupled_derivative(y,omegaCommand,cfg,target)
[contact,v,s]=contact_eval(y,cfg,target); R=quatrot(v.q); thrust=cfg.kT*v.rotor.^2;
rotorForce=[0;0;sum(thrust)]; rotorTorque=zeros(3,1);
for j=1:4, rotorTorque=rotorTorque+cross(cfg.rotorPosition(j,:)',[0;0;thrust(j)])+[0;0;cfg.spin(j)*cfg.kQ*v.rotor(j)^2]; end
acceleration=[0;0;-cfg.g]+(R*rotorForce+contact.vehicle_force)/cfg.mass;
omegaDot=cfg.J\(rotorTorque+contact.torque-cross(v.omega,cfg.J*v.omega));
qdot=0.5*qmult(v.q,[0;v.omega]); motor=(min(max(omegaCommand,cfg.omegaMin),cfg.omegaMax)-v.rotor)/cfg.motorTau;
targetAcceleration=(contact.force-target.c*s.v-target.k*s.x)/target.m;
dy=[v.velocity;acceleration;qdot;omegaDot;motor;s.v;targetAcceleration];
end

function [command,reserve,saturated]=controller(v,pref,normalForce,contactTorque,cfg,alloc)
acceleration=cfg.kp.*(pref-v.position)+cfg.kd.*(-v.velocity); magnitude=norm(acceleration); if magnitude>cfg.maxAccel, acceleration=acceleration*cfg.maxAccel/magnitude; end
force=cfg.mass*([0;0;cfg.g]+acceleration)+normalForce*cfg.normal; desired=desired_rotation(force); R=quatrot(v.q);
skew=0.5*(desired'*R-R'*desired); rotationError=[skew(3,2);skew(1,3);skew(2,1)];
torque=-cfg.kR.*rotationError-cfg.kw.*v.omega+cross(v.omega,cfg.J*v.omega)-contactTorque;
total=max(force'*R(:,3),0); speedSquared=alloc\[total;torque]; clipped=min(max(speedSquared,cfg.omegaMin^2),cfg.omegaMax^2);
saturated=any(abs(clipped-speedSquared)>1e-10); reserve=min((cfg.omegaMax^2-clipped)/(cfg.omegaMax^2-cfg.omegaMin^2)); command=sqrt(clipped);
end

function [contact,v,s]=contact_eval(y,cfg,~)
v=struct('position',y(1:3),'velocity',y(4:6),'q',y(7:10),'omega',y(11:13),'rotor',y(14:17)); s=struct('x',y(18),'v',y(19));
R=quatrot(v.q); tip=v.position+R*cfg.probeOffset; tipVelocity=v.velocity+R*cross(v.omega,cfg.probeOffset);
signedPenetration=(tip-cfg.surface)'*cfg.normal-s.x; closing=tipVelocity'*cfg.normal-s.v; raw=cfg.contactK*signedPenetration+cfg.contactC*closing;
active=signedPenetration>0 && raw>0; contactForce=0; if active, contactForce=raw; end
vehicleForce=-contactForce*cfg.normal; contact=struct('force',contactForce,'vehicle_force',vehicleForce,'torque',cross(cfg.probeOffset,R'*vehicleForce),'penetration',max(signedPenetration,0),'closing',closing,'active',active);
end

function matrix=allocation(cfg)
matrix=zeros(4,4); for j=1:4, moment=cross(cfg.rotorPosition(j,:)',[0;0;cfg.kT]); moment(3)=moment(3)+cfg.spin(j)*cfg.kQ; matrix(:,j)=[cfg.kT;moment]; end
end
function R=desired_rotation(force)
zBody=force/norm(force); heading=[1;0;0]; yBody=cross(zBody,heading); yBody=yBody/norm(yBody); xBody=cross(yBody,zBody); R=[xBody yBody zBody];
end
function product=qmult(a,b)
product=[a(1)*b(1)-dot(a(2:4),b(2:4));a(1)*b(2:4)+b(1)*a(2:4)+cross(a(2:4),b(2:4))];
end
function R=quatrot(q)
q=q/norm(q); w=q(1); x=q(2); y=q(3); z=q(4); R=[1-2*(y^2+z^2),2*(x*y-w*z),2*(x*z+w*y);2*(x*y+w*z),1-2*(x^2+z^2),2*(y*z-w*x);2*(x*z-w*y),2*(y*z+w*x),1-2*(x^2+y^2)];
end
function e=rotm2euler(R)
e=[atan2(R(3,2),R(3,3));asin(min(max(-R(3,1),-1),1));atan2(R(2,1),R(1,1))];
end
