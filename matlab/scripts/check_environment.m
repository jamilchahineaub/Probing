function info = check_environment()
%CHECK_ENVIRONMENT Record MATLAB, Simulink, and optional toolbox availability.
info.matlab_version = version;
info.simulink = exist('simulink','file') == 2;
products = {'System Identification Toolbox','Control System Toolbox', ...
    'Signal Processing Toolbox','Statistics and Machine Learning Toolbox', ...
    'Simscape','Simscape Multibody','Optimization Toolbox', ...
    'Model Predictive Control Toolbox','UAV Toolbox','ROS Toolbox', ...
    'Robotics System Toolbox'};
info.products = struct();
for i = 1:numel(products)
    key = matlab.lang.makeValidName(products{i});
    % `ver` is portable across network/named-user licenses; license feature
    % names are release-dependent and may exceed license()'s length limit.
    installed = ver;
    info.products.(key) = any(strcmp({installed.Name}, products{i}));
end
disp(info)
if ~exist('matlab/experiments','dir'), mkdir('matlab/experiments'); end
fid = fopen('matlab/experiments/environment.txt','w');
fprintf(fid,'MATLAB version: %s\nSimulink license: %d\n',info.matlab_version,info.simulink);
names = fieldnames(info.products);
for i=1:numel(names), fprintf(fid,'%s: %d\n',names{i},info.products.(names{i})); end
fclose(fid);
end
