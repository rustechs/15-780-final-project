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
% B = [];

% f = @(t,x) A*reshape(x,length(x),1);
% c = @(x) C*reshape(x,length(x),1);

% Nonlinear Example -- 2nd Order Model of a Sync Power Machine
% See slide 39 of http://www.dt.fee.unicamp.br/~geromel/non_second.pdf
%
% xDot = f(t,x) + g(t,x)*u(t)
% y = c(x)

f1 = @(t,x) [x(2); 0.5-sin(x(1))];
c = @(x) x;
g = @(t,x) zeros(2,3);

f = @(t,x,u) f1(t,x) + g(t,x)*reshape(u,length(u),1);

%% Initial condition
x0 = [1 -1]; % TODO 
y0 = c(x0);

%% Inputs
u = @(t) ones(1,3);

%% State and output space dimensions
n = length(x0); % state-space dim
m = length(y0); % output-space dim
r = length(u(0)); % input-space dim

%% Save system specification into mat file
save(['configs/' sysName])
