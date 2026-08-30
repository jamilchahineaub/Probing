function tests = test_stage2_physics
tests=functiontests(localfunctions);
end
function testEquilibrium(testCase)
t=(0:0.001:10)'; [~,x,v,~]=msd_rk4(100,2,1,t,ones(size(t))*10,0,0);
verifyLessThan(testCase,abs(x(end)-0.1),5e-6); verifyLessThan(testCase,abs(v(end)),1e-4);
end
function testUndampedEnergy(testCase)
t=(0:0.001:10)'; [~,x,v,~]=msd_rk4(100,0,1,t,zeros(size(t)),0.01,0.02);
E=0.5*1*v.^2+0.5*100*x.^2; verifyLessThan(testCase,max(E)-min(E),1e-7);
end
function testDampedEnergy(testCase)
t=(0:0.001:5)'; [~,x,v,~]=msd_rk4(100,1,1,t,zeros(size(t)),0.01,0);
E=0.5*v.^2+50*x.^2; verifyLessThan(testCase,E(end),E(1));
end
function testAnalyticalSuite(testCase)
M=validate_stage2_physics(); verifyLessThan(testCase,max(M.primary_error),5e-6); verifyLessThan(testCase,M.secondary_error(strcmp(M.test,'undamped')),1e-8);
end
function testForcedResponse(testCase)
t=(0:1e-4:3)'; k=80;c=5;m=1.2;F0=4;wn=sqrt(k/m);z=c/(2*sqrt(k*m));wd=wn*sqrt(1-z^2);xeq=F0/k;
xa=xeq+exp(-z*wn*t).*(-xeq*cos(wd*t)-z*wn*xeq/wd*sin(wd*t)); [~,x]=msd_rk4(k,c,m,t,F0*ones(size(t)),0,0);
verifyLessThan(testCase,max(abs(x(:)-xa)),1e-8);
end
function testLockedChirp(testCase)
[t,F]=chirp_probe(500,3,.5,5,.5); verifyEqual(testCase,t(end),3,'AbsTol',1e-12); verifyLessThanOrEqual(testCase,max(abs(F)),.5); verifyEqual(testCase,numel(t),1501);
end
function testCausalSensorDelay(testCase)
t=(0:.005:1)'; x=sin(2*pi*t); z=zeros(size(t)); ns=struct('displacement',0,'velocity',0,'acceleration',0,'force',0); s=causal_sensor(t,x,2*pi*cos(2*pi*t),z,z,ns,1,10);
verifyEqual(testCase,s.alpha,.005/(1/(2*pi*10)+.005),'AbsTol',1e-14); verifyEqual(testCase,s.displacement(2),s.alpha*x(2),'AbsTol',1e-14);
end
