function metrics = validate_stage2_physics()
% Quantitative analytical verification independent of Python trajectories.
dt=1e-4; t=(0:dt:5)'; rows=cell(0,3);
% Underdamped free response.
k=100;c=4;m=1;x0=.01;v0=.02;wn=sqrt(k/m);z=c/(2*sqrt(k*m));wd=wn*sqrt(1-z^2);
xa=exp(-z*wn*t).*(x0*cos(wd*t)+(v0+z*wn*x0)/wd*sin(wd*t)); [~,x,v,~]=msd_rk4(k,c,m,t,zeros(size(t)),x0,v0);
rows(end+1,:)={'free_underdamped',max(abs(x(:)-xa)),max(abs(v(:)-gradient(xa,dt)))};
% Constant-force response is equilibrium plus a homogeneous transient.
F0=10;xeq=F0/k; ya=xeq+exp(-z*wn*t).*((x0-xeq)*cos(wd*t)+(v0+z*wn*(x0-xeq))/wd*sin(wd*t));
[~,x,v,~]=msd_rk4(k,c,m,t,F0*ones(size(t)),x0,v0); rows(end+1,:)={'forced_step',max(abs(x(:)-ya)),max(abs(v(:)-gradient(ya,dt)))};
% Undamped exact response and energy conservation.
c=0; xa=x0*cos(wn*t)+v0/wn*sin(wn*t); [~,x,v,~]=msd_rk4(k,c,m,t,zeros(size(t)),x0,v0); E=.5*m*v.^2+.5*k*x.^2;
rows(end+1,:)={'undamped',max(abs(x(:)-xa)),max(E)-min(E)};
% Overdamped roots.
c=30;r=roots([m c k]); A=[1 1;r(1) r(2)]\[x0;v0]; xa=A(1)*exp(r(1)*t)+A(2)*exp(r(2)*t);
[~,x,v,~]=msd_rk4(k,c,m,t,zeros(size(t)),x0,v0); rows(end+1,:)={'highly_damped',max(abs(x(:)-xa)),max(abs(v(:)-gradient(xa,dt)))};
% Equilibrium and dissipation identities.
[~,x,v,~]=msd_rk4(100,10,1,t,F0*ones(size(t)),0,0); rows(end+1,:)={'equilibrium',abs(x(end)-F0/100),abs(v(end))};
[~,x,v,~]=msd_rk4(100,2,1,t,zeros(size(t)),.01,0); E=.5*v.^2+50*x.^2; rows(end+1,:)={'dissipation',max(diff(E)),E(end)/E(1)};
metrics=cell2table(rows,'VariableNames',{'test','primary_error','secondary_error'});
end
