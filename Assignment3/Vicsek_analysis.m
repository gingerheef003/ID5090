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
% clc
% Speed
v0 = 0.1;

% Domain size
L = 10;

RI = 1;

% heterogeneity (for later)
sig2 = sig1;
Nr = 0.0; N1 = round(Nr*N); N2 = N - N1;

% plotting
plotflag = 0;
showgrpdir = 0;
plotnetwork = 1;
showgraph = 1 - plotnetwork;

% AGENT BASED MODEL: SIMULATION
Vicsek_code

show2Dhistogram = 0;
showheterogeneity = 0;
showorderpartimeseries = 1;
timeseriesmodelling = 0;
compute_network_params = 1;
compute_polarization = 1;
compute_autocorrelation = 1;

%% Order parameter

if compute_polarization == 1
    px = mean(cos(th),1);
    py = mean(sin(th),1);
    p = sqrt((px.^2 + py.^2)');
    
    % Plot the order parameter
    subplot(1,3,1)
    plot(px')
    hold on
    plot(py')

    subplot(1,3,2)
    plot(p)
    title('Polarization Order Parameter')

    p = p(2500:end); % cut out the initial effects

    subplot(1,3,3)
    autocorr(p, 'NumLags', 100)
    title('Autocorrelation Function')
end

%% Network parameters
% if compute_network_params == 1
%     closeness_centrality = zeros(N, Time);
%     eigenvector_centrality = zeros(N, Time);
%     degree_centrality = zeros(N, Time);
% 
%     for t = 1:Time
%         G = graph(Adj);
%         closeness_centrality(:, t) = centrality(G, 'closeness');
%         eigenvector_centrality(:, t) = centrality(G, 'eigenvector');
%         degree_centrality(:, t) = centrality(G, 'degree');
%     end
% 
%     % Averaging over all agents
%     avg_closeness_centrality = mean(closeness_centrality, 1);
%     avg_eigenvector_centrality = mean(eigenvector_centrality, 1);
%     avg_degree_centrality = mean(degree_centrality, 1);
% 
%     % Plotting the network parameters
%     figure
%     subplot(3,1,1)
%     plot(avg_closeness_centrality)
%     title('Average Closeness Centrality')
% 
%     subplot(3,1,2)
%     plot(avg_eigenvector_centrality)
%     title('Average Eigenvector Centrality')
% 
%     subplot(3,1,3)
%     plot(avg_degree_centrality)
%     title('Average Degree Centrality')
% end


if compute_network_params == 1
    closeness_centrality = zeros(N, Time);
    eigenvector_centrality = zeros(N, Time);  % Note: Eigenvector centrality may not work for directed graphs
    degree_centrality = zeros(N, Time);
    
    for t = 1:Time
        G = digraph(Adj_all(:, :, t));  % Use directed graph
        
        % Closeness centrality (incoming direction)
        closeness_centrality(:, t) = centrality(G, 'incloseness');
        
        % Eigenvector centrality is not standard for directed graphs in MATLAB.
        % Use PageRank or Hubs/Authorities instead.
        % eigenvector_centrality(:, t) = centrality(G, 'pagerank');  % Alternative
        
        % Degree centrality (incoming edges)
        degree_centrality(:, t) = centrality(G, 'indegree');
    end

    % Average over agents
    avg_closeness_centrality = mean(closeness_centrality, 1);
    disp(mean(avg_closeness_centrality))
    avg_degree_centrality = mean(degree_centrality, 1);
    disp(mean(avg_degree_centrality))
    
    % Plot results
    figure
    subplot(2,1,1)
    plot(avg_closeness_centrality)
    title('Average In-Closeness Centrality')
    
    subplot(2,1,2)
    plot(avg_degree_centrality)
    title('Average In-Degree Centrality')
end

%% Relationship between polarization and network parameters
if compute_polarization == 1 && compute_network_params == 1
    figure
    subplot(3,1,1)
    plot(p)
    hold on
    plot(avg_closeness_centrality)
    title('Polarization Order Parameter and Closeness Centrality')
    % 
    % subplot(3,1,2)
    % plot(p)
    % hold on
    % plot(avg_eigenvector_centrality)
    % title('Polarization Order Parameter and Eigenvector Centrality')

    subplot(3,1,3)
    plot(p)
    hold on
    plot(avg_degree_centrality)
    title('Polarization Order Parameter and Degree Centrality')
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
    hold on
    histogram(prediction)

end