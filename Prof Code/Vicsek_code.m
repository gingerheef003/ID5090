% clc
% clear
% close
%
% N = 100;
% Nr = 0.0;
%
% N1 = round(Nr*N);
% N2 = N-N1;
%
% Time = 10^4;
% dt = 1;
%
% sig1 = 0.3;
% sig2 = sig1;
%
% v0 = 0.1;
%
% L = 25;

% rd = [0.05*ones(N1,1); 0.1*ones(N2,1)];

rd = RI*ones(N,1);
sig = [sig1*ones(N1,1); sig2*ones(N2,1)];

x = zeros(N,Time);
y = zeros(N,Time);
th = zeros(N,Time);

x(:, 1) = L*rand(N, 1);
y(:, 1) = L*rand(N, 1);
th(:,1) = pi*(2*rand(N,1) - 1);

for t = 2:Time
    if rem(t, 1000) == 0
        disp(t)
    end

    %     dij = pdist([x(:,1) y(:,1)]);

    Adj = zeros(N);

    for i = 1:N
        dist_x = x(:,t-1) - repmat(x(i,t-1), N,1); % distance between x coordinate of i and x coordinates of other agents
        dist_x = dist_x - (round(dist_x/L))*L; % taking the shortest distance because of periodic boundary conditions
        dist_y = y(:,t-1) - repmat(y(i, t-1), N,1); % similarly for y
        dist_y = dist_y - (round(dist_y/L))*L;

        dist_mag = sqrt(dist_x.^2 + dist_y.^2); %magnitude of rij
        dist_mag(i) = rd(i);

        js = find(dist_mag <= rd(i));

        Adj(i,js) = 1;
        Adj(i,i) = 0;

        th(i, t) = atan2(mean(sin(th(js, t-1))), mean(cos(th(js, t-1)))) ;
        th(i, t) = th(i, t) + (sig(i)*(rand(1)*2-1));

        th(i, t) = wrapToPi(th(i,t));
    end

    x(:,t) = x(:,t-1) + cos(th(:, t))*v0*dt;
    y(:,t) = y(:,t-1) + sin(th(:, t))*v0*dt;

    x(x(:,t) > L, t) = x(x(:,t) > L, t) - L;
    x(x(:,t) < 0, t) = x(x(:,t) < 0, t) + L;
    y(y(:,t) > L, t) = y(y(:,t) > L, t) - L;
    y(y(:,t) < 0, t) = y(y(:,t) < 0, t) + L;

    if plotflag == 1
        quiver(x(1:N1, t), y(1:N1, t), cos(th(1:N1,t)), sin(th(1:N1,t)), 0.3, LineWidth=2, ShowArrowHead="on")
        hold on
        quiver(x(N1+1:N, t), y(N1+1:N, t), cos(th(N1+1:N,t)), sin(th(N1+1:N,t)), 0.3, LineWidth=2, ShowArrowHead="on")
        axis equal
        axis([0 L 0 L])

        if showgrpdir == 1
            px = mean(cos(th(:,t)),1);
            py = mean(sin(th(:,t)),1);
            quiver(L/2,L/2,px,py,1,'LineWidth',3)
        end

        if plotnetwork == 1
            plot(graph(Adj),'XData',x(:,t), 'YData', y(:,t), 'LineWidth', 2)
        end
        drawnow limitrate
        hold off

    elseif showgraph == 1
        plot(graph(Adj),'LineWidth', 2)

        drawnow limitrate
        hold off
    end

end
disp('sim complete')
