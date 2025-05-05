clc
clear
close all

N = 100; % Number of agents

% Time of simulation and time step
Time = 10^4;
dt = 1;

% Noise added
sig1 = pi/12;
clc
% Speed
v0 = 0.05;

% Domain size
L = 5;

% heterogeneity (for later)
sig2 = sig1;
Nr = 0.0; N1 = round(Nr*N); N2 = N-N1;

% plotting
plotflag = 1;

% AGENT BASED MODEL: SIMULATION
Vicsek_code

px = mean(cos(th), 1);
py = mean(sin(th), 1);
p = sqrt((px.^2 + py.^2));

subplot()


px = mean(cos(th),1);
py = mean(sin(th),1);
p = sqrt(px.^2 + py.^2);

px = px./px;
py = 0*py./p;

th_g = atan2(py,px);

del_th = zeros(N,Time);

for k = 1:N
    del_th(k,:) = atan2((sin(th_g).*cost(th(k,:)) - cost(th_g).*sin(th(k,:))), (cost(th_g).*cost(th(k,:)) + sin(th(k,:)).*sing(th_g)));
end

histogram(del_th(:,1000:end), linspace(-pi,pi,250), 'Normalization','probability');
xlim([-pi pi]);