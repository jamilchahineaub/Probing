function sensed = causal_sensor(t,x,v,a,F,noise_std,seed,cutoff_hz)
% Synchronized causal sensing with an explicit first-order low-pass filter.
rng(seed,'twister'); t=t(:); n=numel(t); dt=t(2)-t(1);
raw_x=x(:)+noise_std.displacement*randn(n,1);
raw_v=v(:)+noise_std.velocity*randn(n,1); %#ok<NASGU> retained sensor channel
raw_a=a(:)+noise_std.acceleration*randn(n,1); %#ok<NASGU>
raw_F=F(:)+noise_std.force*randn(n,1);
tau=1/(2*pi*cutoff_hz); alpha=dt/(tau+dt);
xf=zeros(n,1); Ff=zeros(n,1); xf(1)=raw_x(1); Ff(1)=raw_F(1);
for i=2:n
    xf(i)=xf(i-1)+alpha*(raw_x(i)-xf(i-1));
    Ff(i)=Ff(i-1)+alpha*(raw_F(i)-Ff(i-1));
end
vf=[(xf(2)-xf(1))/dt; diff(xf)/dt];
af=[(xf(3)-2*xf(2)+xf(1))/dt^2; (xf(3)-2*xf(2)+xf(1))/dt^2; diff(xf,2)/dt^2];
sensed=struct('time',t,'displacement',xf,'velocity',vf,'acceleration',af, ...
    'force',Ff,'raw_displacement',raw_x,'raw_acceleration',raw_a, ...
    'raw_force',raw_F,'alpha',alpha,'group_delay_s',tau+dt);
end
