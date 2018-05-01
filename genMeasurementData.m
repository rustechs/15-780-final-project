function genMeasurementData(sysName,approxDegree)

    if nargin < 2
        approxDegree = 5;
    end

    % Load system specifications into a struct
    sys = load(['configs/' sysName]);

    % Generate true system data
    [t,x] = ode45(@(t,x) sys.f(t,x,sys.u(t)),sys.tSamples,sys.x0);
    
    yTrue = zeros(length(t),sys.m);
    u = zeros(length(t),sys.r);
    xDot = zeros(length(t),sys.n);
    for i = 1:length(t)
        u(i,:) = sys.u(t(i));
        yTrue(i,:) = sys.c(x(i,:));
        xDot(i,:) = sys.f(t(i),x(i,:),u(i,:));
    end
    
    % Generate noisy measurements
    rng('shuffle');
    R = chol(sys.Sigma);
    yNoisy = yTrue + (repmat(sys.mu,length(t),1) + randn(size(yTrue))*R);
    
    % Save generated measurements as raw text files in data folder. Text
    % file names match the name of the config file from which they were
    % generated.
    
    infoMat = [sys.r sys.n sys.n sys.m approxDegree];
    saveMat = [u x xDot yTrue yNoisy];
    
    dlmwrite(['data/' sysName '.data'],infoMat,'delimiter','\t');
    dlmwrite(['data/' sysName '.data'],saveMat,'-append','delimiter','\t');
    
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