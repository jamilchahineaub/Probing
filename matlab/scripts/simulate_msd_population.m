function sim = simulate_msd_population(cases,t,command,unilateral)
% Vectorized independent RK4 model for F=m*xdd+c*xd+k*x.
t=t(:); command=command(:); if unilateral, force=max(command,0); else, force=command; end
k=cases.k(:); c=cases.c(:); m=cases.m(:); nc=numel(k); nt=numel(t);
x=zeros(nc,nt); v=zeros(nc,nt);
for n=1:nt-1
    h=t(n+1)-t(n); u0=force(n); u1=force(n+1); um=(u0+u1)/2;
    xn=x(:,n); vn=v(:,n);
    k1x=vn; k1v=(u0-c.*vn-k.*xn)./m;
    k2x=vn+h*k1v/2; k2v=(um-c.*(vn+h*k1v/2)-k.*(xn+h*k1x/2))./m;
    k3x=vn+h*k2v/2; k3v=(um-c.*(vn+h*k2v/2)-k.*(xn+h*k2x/2))./m;
    k4x=vn+h*k3v; k4v=(u1-c.*(vn+h*k3v)-k.*(xn+h*k3x))./m;
    x(:,n+1)=xn+h*(k1x+2*k2x+2*k3x+k4x)/6;
    v(:,n+1)=vn+h*(k1v+2*k2v+2*k3v+k4v)/6;
end
a=(force'-c.*v-k.*x)./m;
sim=struct('time',t,'command',command,'force',force,'x',x,'v',v,'a',a);
end
