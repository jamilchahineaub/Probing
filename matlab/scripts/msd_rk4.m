function [t,x,v,a] = msd_rk4(k,c,m,t,F,x0,v0)
% Independent fixed-step RK4 integration of m*x''+c*x'+k*x=F.
t=t(:)'; F=F(:)'; N = numel(t); x=zeros(1,N); v=zeros(1,N); a=zeros(1,N); x(1)=x0; v(1)=v0;
for n=1:N-1
    h=t(n+1)-t(n); u0=F(n); u1=F(n+1); um=0.5*(u0+u1);
    f=@(z,u)[z(2); (u-c*z(2)-k*z(1))/m]; z=[x(n);v(n)];
    q1=f(z,u0); q2=f(z+h*q1/2,um); q3=f(z+h*q2/2,um); q4=f(z+h*q3,u1);
    z=z+h*(q1+2*q2+2*q3+q4)/6; x(n+1)=z(1); v(n+1)=z(2);
end
a=(F-c*v-k*x)/m;
end
