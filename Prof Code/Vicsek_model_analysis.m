% Vicsek model analysis

% clc
clear
% close all

%%
N = 100; % Number of agents

% Time of simulation and time step
Time = 10^4;
dt = 1;

% Noise added
sig1 = pi/12;
clc
% Speed
v0 = 0.1;

% Domain size
L = 10;

RI = 1;

% heterogeneity (for later)
sig2 = sig1;
Nr = 0.0; N1 = round(Nr*N); N2 = N - N1;


% plotting
plotflag = 1;
showgrpdir = 0;

plotnetwork = 0;
showgraph = 1 - plotnetwork;

% AGENT BASED MODEL: SIMULATION
Vicsek_code

show2Dhistogram = 0;
showheterogeneity = 0;
showorderpartimeseries = 1;
timeseriesmodelling = 0;
%% Order parameter

if showorderpartimeseries == 1

    px = mean(cos(th),1);
    py = mean(sin(th),1);
    p = sqrt((px.^2 + py.^2)');

    subplot(1,3,1)
    plot(px')
    hold all
    plot(py')

    subplot(1,3,2)
    plot(p)

    p = p(2500:end); % cut out the initial effects

    subplot(1,3,3)
    autocorr(p,100)

end
%% 2D histogram

if show2Dhistogram == 1

    px = mean(cos(th),1);
    py = mean(sin(th),1);

    ctrs{1}=linspace(-1,1,50);
    ctrs{2}=linspace(-1,1,50);

    hist3([px(1000:end)' py(1000:end)'],ctrs,'CdataMode','auto')
    xlim([-1,1])
    ylim([-1,1])
    colorbar
    % view(2)
    shading interp
    % axis equal
    xlim([-1 1])
    ylim([-1 1])

end
%% Capturing the heterogeneous behaviour

if showheterogeneity == 1
    % figure

    px = mean(cos(th),1);
    py = mean(sin(th),1);
    p = sqrt(px.^2 + py.^2);

    px = px./p;
    py = py./p;

    th_g = atan2(py,px);

    del_th = zeros(N,Time);
    for k = 1:N
        % del_th(k,:) = (dot([cos(th_g(:,:)); sin(th_g(:,:))], [cos(th(1,:)); sin(th(1,:))]));
        del_th(k,:) = atan2((sin(th_g).*cos(th(k,:)) - cos(th_g).*sin(th(k,:))),...
            (cos(th_g).*cos(th(k,:)) + sin(th(k,:)).*sin(th_g)));
    end

    histogram(del_th(:,1000:end),linspace(-pi,pi,250),'Normalization','probability')
    xlim([-pi pi])

end

%%

if timeseriesmodelling == 1

    px = mean(cos(th),1);
    py = mean(sin(th),1);
    p = sqrt(px.^2 + py.^2);

    y = p(3000:end)';

    Mdl = arima(1,0,0);
    EstMdl = estimate(Mdl,(y));
    residuals = infer(EstMdl,(y));
    prediction = (y)+residuals;
    figure()
    histogram((y))
    hold all
    histogram(prediction)

end