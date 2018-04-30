function genMeasurementData(sysName)

    % Load system specifications into a struct
    sys = load(['configs/' sysName]);

    % Generate true system data
    [t,x] = ode45(sys.f,sys.tSamples,sys.x0);
    
    yTrue = zeros(length(t),sys.m);
    for i = 1:length(t)
        yTrue(i,:) = sys.c(x(i,:));
    end
    
    % Generate noisy measurements
    rng('shuffle');
    R = chol(sys.Sigma);
    yNoisy = yTrue + (repmat(sys.mu,length(t),1) + randn(size(yTrue))*R);
    
    % Save generated measurements as raw text files in data folder. Text
    % file names match the name of the config file from which they were
    % generated.
    
    saveMat = [t x yTrue yNoisy];
    
    dlmwrite(['data/' sysName '.data'],saveMat,'\t');
    
    % Plot some stuff
    figure
    hold on;
    plot(t,yNoisy,'.');
    plot(t,yTrue,'-','LineWidth',2);
    box on;
    grid minor;
    xlabel('time');
    ylabel('outputs');
end