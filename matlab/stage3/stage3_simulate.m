function result = stage3_simulate(cfg)
%STAGE3_SIMULATE Independent equations-based 6-DoF/contact reconstruction.
t=(0:cfg.dt:cfg.duration)'; n=numel(t); u=zeros(n,1);
phase=2*pi*(cfg.probe_f0*t+0.5*(cfg.probe_f1-cfg.probe_f0)/cfg.probe_duration*t.^2);
active=t<=cfg.probe_duration; u(active)=max(cfg.probe_amplitude*sin(phase(active)),0);
A=allocation(cfg); hover=sqrt(cfg.mass*cfg.g/(4*cfg.kT));
y=[cfg.surface-cfg.probe_offset;zeros(3,1);1;0;0;0;zeros(3,1);hover*ones(4,1);0;0];
Y=zeros(n,19); Fc=zeros(n,1); Tau=zeros(n,3); Pen=zeros(n,1); Reserve=zeros(n,1); Saturated=false(n,1);
offset=0;
for i=1:n
    [contact,vehicle,target]=contact_eval(y,cfg); %#ok<ASGLU>
    err=u(i)-contact.force;
    if i>1, offset=offset+cfg.admittance*min(err,0)*cfg.dt; offset=min(max(offset,-cfg.position_limit),cfg.position_limit); end
    passive=t(i)>cfg.probe_duration;
    if passive, offset=min(offset,-cfg.passive_retraction); end
    pref=cfg.surface-cfg.probe_offset+offset*cfg.normal;
    normal_force=u(i)+cfg.force_gain*err; normal_force=min(max(normal_force,-cfg.force_limit),cfg.force_limit);
    if passive, normal_force=0; end
    [omega_cmd,reserve,saturated]=controller(vehicle,pref,normal_force,contact.torque,cfg,A);
    Y(i,:)=y'; Fc(i)=contact.force; Tau(i,:)=contact.torque'; Pen(i)=contact.penetration; Reserve(i)=reserve; Saturated(i)=saturated;
    if i==n, continue; end
    f=@(state) coupled_derivative(state,omega_cmd,cfg);
    k1=f(y); k2=f(y+0.5*cfg.dt*k1); k3=f(y+0.5*cfg.dt*k2); k4=f(y+cfg.dt*k3);
    y=y+cfg.dt*(k1+2*k2+2*k3+k4)/6; y(7:10)=y(7:10)/norm(y(7:10)); y(14:17)=min(max(y(14:17),cfg.omega_min),cfg.omega_max);
end
euler=zeros(n,3); target_a=zeros(n,1); thrust=zeros(n,4);
for i=1:n
    euler(i,:)=rotm2euler(quatrot(Y(i,7:10)'))';
    target_a(i)=(Fc(i)-cfg.target_c*Y(i,19)-cfg.target_k*Y(i,18))/cfg.target_m;
    thrust(i,:)=cfg.kT*Y(i,14:17).^2;
end
result=struct('time',t,'reference_force',u,'contact_force',Fc,'state',Y,'euler',euler, ...
    'contact_torque',Tau,'penetration',Pen,'target_acceleration',target_a,'rotor_thrust',thrust, ...
    'actuator_reserve',Reserve,'saturated',Saturated);
end

function dy=coupled_derivative(y,omega_cmd,cfg)
[contact,v,s]=contact_eval(y,cfg); R=quatrot(v.q);
thrust=cfg.kT*v.rotor.^2; rotor_force=[0;0;sum(thrust)];
rotor_tau=zeros(3,1);
for j=1:4, rotor_tau=rotor_tau+cross(cfg.rotor_position(j,:)',[0;0;thrust(j)])+[0;0;cfg.spin(j)*cfg.kQ*v.rotor(j)^2]; end
acc=[0;0;-cfg.g]+(R*rotor_force+contact.vehicle_force)/cfg.mass;
omega_dot=cfg.J\(rotor_tau+contact.torque-cross(v.omega,cfg.J*v.omega));
qdot=0.5*qmult(v.q,[0;v.omega]); motor=(min(max(omega_cmd,cfg.omega_min),cfg.omega_max)-v.rotor)/cfg.motor_tau;
target_acc=(contact.force-cfg.target_c*s.v-cfg.target_k*s.x)/cfg.target_m;
dy=[v.velocity;acc;qdot;omega_dot;motor;s.v;target_acc];
end

function [cmd,reserve,saturated]=controller(v,pref,normal_force,contact_torque,cfg,A)
acc=cfg.kp.*(pref-v.position)+cfg.kd.*(-v.velocity); an=norm(acc); if an>cfg.max_accel, acc=acc*cfg.max_accel/an; end
force=cfg.mass*([0;0;cfg.g]+acc)+normal_force*cfg.normal; Rd=desired_rotation(force); R=quatrot(v.q);
skew=0.5*(Rd'*R-R'*Rd); eR=[skew(3,2);skew(1,3);skew(2,1)];
tau=-cfg.kR.*eR-cfg.kw.*v.omega+cross(v.omega,cfg.J*v.omega)-contact_torque;
total=max(force'*R(:,3),0); speed2=A\[total;tau]; clipped=min(max(speed2,cfg.omega_min^2),cfg.omega_max^2);
saturated=any(abs(clipped-speed2)>1e-10); reserve=min((cfg.omega_max^2-clipped)/(cfg.omega_max^2-cfg.omega_min^2)); cmd=sqrt(clipped);
end

function [contact,v,s]=contact_eval(y,cfg)
v=struct('position',y(1:3),'velocity',y(4:6),'q',y(7:10),'omega',y(11:13),'rotor',y(14:17)); s=struct('x',y(18),'v',y(19));
R=quatrot(v.q); tip=v.position+R*cfg.probe_offset; tipv=v.velocity+R*cross(v.omega,cfg.probe_offset);
penetration=(tip-cfg.surface)'*cfg.normal-s.x; closing=tipv'*cfg.normal-s.v;
raw=cfg.contact_k*penetration+cfg.contact_c*closing; force=0; if penetration>0, force=max(raw,0); end
vehicle_force=-force*cfg.normal; torque=cross(cfg.probe_offset,R'*vehicle_force);
contact=struct('force',force,'vehicle_force',vehicle_force,'torque',torque,'penetration',max(penetration,0));
end

function A=allocation(cfg)
A=zeros(4,4);
for j=1:4
    moment=cross(cfg.rotor_position(j,:)',[0;0;cfg.kT]); moment(3)=moment(3)+cfg.spin(j)*cfg.kQ;
    A(:,j)=[cfg.kT;moment];
end
end

function R=desired_rotation(force)
zb=force/norm(force); heading=[1;0;0]; yb=cross(zb,heading); yb=yb/norm(yb); xb=cross(yb,zb); R=[xb yb zb];
end

function product=qmult(a,b)
product=[a(1)*b(1)-dot(a(2:4),b(2:4)); a(1)*b(2:4)+b(1)*a(2:4)+cross(a(2:4),b(2:4))];
end

function R=quatrot(q)
q=q/norm(q); w=q(1); x=q(2); y=q(3); z=q(4);
R=[1-2*(y^2+z^2),2*(x*y-w*z),2*(x*z+w*y);2*(x*y+w*z),1-2*(x^2+z^2),2*(y*z-w*x);2*(x*z-w*y),2*(y*z+w*x),1-2*(x^2+y^2)];
end

function e=rotm2euler(R)
e=[atan2(R(3,2),R(3,3));asin(min(max(-R(3,1),-1),1));atan2(R(2,1),R(1,1))];
end
