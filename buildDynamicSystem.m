function buildDynamicSystem(sysName)

    % Parameters
    tPeriod = 1e3; % [Hz]
    tSamples = 0:tPeriod:10; % [sec]
    
    mu = []; % TODO
    Sigma = []; % TODO

    % System Structure and Dynamics
    A = []; % TODO
    C = []; % TODO

    f = @(t,x) A*reshape(x,length(x),1);
    c = @(x) C*reshape(x,length(x),1);
    
    % Initial condition
    x0 = []; % TODO 
    y0 = c(x0);
    
    % State and output space dimensions
    n = length(x0);
    m = length(y0);

    % Save system specification into mat file
    save(['configs/' sysName])
    
end