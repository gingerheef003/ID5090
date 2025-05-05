clear
clc

N = 100;

Time = 10^4;
dt = 1;

sig1 = pi/12;
clc

v0 = 0.05;

L = 10;

RI = 1;

sig2 = sig1;
Nr = 0.0; N1 = round(Nr*N); N2 = N-N1;

show_plot = 1;
show_polarization_graph = 0;

vicsek

% show2Dhistogram = 1;
% showheterogeneity = 0;
% showorderparttimeseries = 0;
% 
% if showorderparttimeseries == 1
%     px = mean(cos(th));
%     py = mean(sin(th));
% 
% 
% end
% 
% if show2Dhistogram == 1
% end
% 
% if showheterogeneity == 1
% end
% 

