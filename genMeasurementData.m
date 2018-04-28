function genMeasurementData(sysName)

    % Load system specifications into a struct
    sys = load(sysName);
    
    % Save all generated data to a struct
    data = struct([]);
    
    % Generate true system data
    [data.t,data.x] = ode45(odeFcn,sys.tSamples,sys.x0);
    
    data.yTrue = zeros(length(data.t),sys.m);
    for i = 1:length(data.t)
        data.yTrue(i,:) = sys.c(data.x(i,:));
    end
    
    % Generate noisy measurements
    rng('shuffle');
    R = chol(sys.Sigma);
    data.yNoisy = data.yTrue + (repmat(sys.mu,length(data.t),1) + randn(size(data.yTrue))*R);
    
    % Save generated measurements as raw text files in data folder. Text
    % file names match the name of the config file from which they were
    % generated.
    
    save(['data/' sysName '.data'],'-struct','data','-ascii')
    
    % Plot some stuff
    figure
    hold on;
    plot(data.t,data.yTrue,'-');
    plot(data.t,data.yNoisy,'--');
    box on;
    grid minor;
    xlabel('time');
    ylabel('outputs');
end