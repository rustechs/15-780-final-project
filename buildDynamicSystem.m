function buildDynamicSystem(sysName)

    % Parameters
    T_sample = 1e3; % [Hz]
    mu = []; % TODO
    Sigma = []; % TODO

    % System Structure and Dynamics
    A = []; % TODO
    C = []; % TODO

    f = @(t,x) A*x;
    c = @(x) C*x;

    % Save system specification into mat file
    save(['configs/' sysName])
    
end