function [t,F] = chirp_probe(fs,duration,f0,f1,amplitude)
% Bounded linear chirp, independently implemented from the specification.
t=(0:1/fs:duration)'; phase=2*pi*(f0*t+(f1-f0)/(2*duration)*t.^2);
F=amplitude*sin(phase); F=max(min(F,amplitude),-amplitude);
end
