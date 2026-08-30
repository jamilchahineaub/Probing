function cfg = exp0009_hybrid_config()
%EXP0009_HYBRID_CONFIG Independently declared frozen EXP-0009 constants.
cfg.dt=0.001; cfg.duration=14.0; cfg.g=9.80665;
cfg.mass=1.50; cfg.J=diag([0.029 0.029 0.055]);
cfg.kT=1.90e-5; cfg.kQ=2.60e-7; cfg.motorTau=0.030;
cfg.omegaMin=0; cfg.omegaMax=900; cfg.arm=0.23;
cfg.rotorPosition=[cfg.arm 0 0;0 cfg.arm 0;-cfg.arm 0 0;0 -cfg.arm 0];
cfg.spin=[1;-1;1;-1]; cfg.probeOffset=[0.30;0;-0.08];
cfg.surface=[0;0;0.92]; cfg.normal=[1;0;0];
cfg.contactK=5000; cfg.contactC=10;
cfg.kp=[4;4;6]; cfg.kd=[8;4;5]; cfg.kR=[8;8;1.2]; cfg.kw=[0.70;0.70;0.40]; cfg.maxAccel=8;
cfg.clearance=0.002; cfg.approachVelocity=0.001; cfg.approachMax=0.0013;
cfg.detectForce=0.03; cfg.acquireForce=0.05; cfg.acquireRamp=0.20;
cfg.acquireDwell=0.20; cfg.acquireError=0.04; cfg.acquireVelocity=0.015; cfg.acquireTimeout=4.0;
cfg.preload=0.50; cfg.preloadRamp=2.0; cfg.preloadDwell=0.30;
cfg.preloadError=0.20; cfg.preloadVelocity=0.020; cfg.preloadTimeout=4.0;
cfg.probeDuration=3.0; cfg.observeDuration=0.5; cfg.unloadDuration=0.40; cfg.passiveRetraction=0.010;
cfg.probeAmplitude=0.5; cfg.probeF0=0.5; cfg.probeF1=5.0;
cfg.maxNormalVelocity=0.08; cfg.maxNormalAcceleration=2.0; cfg.compressionVelocity=0.001;
cfg.referenceDerivativeTau=0.020; cfg.minimumForceCommand=-0.20; cfg.maximumForceCommand=0.50;
cfg.abortForce=1.50; cfg.abortAttitudeDeg=15; cfg.abortPenetration=0.002;
cfg.phaseNames=["APPROACH","CONTACT_ACQUIRE","PRELOAD","PROBE", ...
    "CONTROLLED_UNLOAD","PASSIVE_OBSERVE","DECISION","ABORT"];
end
