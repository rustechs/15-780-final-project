%% Parameters
% Save Name
sysName = 'nl-ti-sync-machine';

% Sampling Rate/Length
tPeriod = 1e-3; % [sec]
tSamples = 0:tPeriod:30; % [sec]

% Gaussian Sampling Noise
mu = 0; 
Sigma = 0.1; 

%% System Structure and Dynamics

% % Linear Example -- 2nd Order Spring/Mass/Damper System
% k = 0.1;
% b = 0.1;
% m = 1;
% 
% A = [0 1; -k/m -b/m];
% C = [1 0];
% 
% f = @(t,x) A*reshape(x,length(x),1);
% c = @(x) C*reshape(x,length(x),1);

% Nonlinear Example -- 2nd Order Model of a Sync Power Machine
% See slide 39 of http://www.dt.fee.unicamp.br/~geromel/non_second.pdf
f = @(t,x) [x(2); 0.5-sin(x(1))];
c = @(x) x;

%% Initial condition
x0 = [1 -1]; % TODO 
y0 = c(x0);

%% State and output space dimensions
n = length(x0);
m = length(y0);

%% Save system specification into mat file
save(['configs/' sysName])
