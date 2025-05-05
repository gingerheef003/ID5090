clc
clear
close all

% Parameters
N = 300;
Time = 10^4;
dt = 1;
v0 = 0.05;
L = 10;

% Parameter sets for different cases
params = struct('name', {'GlobalFlocking', 'LocalFlocking', 'NoFlocking'}, ...
                'RI', {3, 1.5, 0.5}, ...
                'sig', {pi/24, pi/12, pi/6});

% Diagnostic results
diagnostics = struct('name', {'GlobalFlocking', 'LocalFlocking', 'NoFlocking'}, ...
                     'OrderParameter', cell(1,3), ...
                     'VelocityCorrelation', cell(1,3), ...
                     'PositionCorrelation', cell(1,3));

% Run simulations for each parameter set
for p = 1:length(params)
    % Extract parameters
    RI = params(p).RI;
    sig = params(p).sig;
    
    % Initial conditions
    pos = rand(N, 2) * L;
    theta = rand(N, 1) * 2 * pi;
    
    % Store data
    pos_data = zeros(N, 2, Time);
    theta_data = zeros(N, Time);
    
    % Visualization setup
    figure;
    axis equal;
    axis([0 L 0 L]);
    vel = v0 * [cos(theta) sin(theta)]; % Initial velocities for visualization
    quiver(pos(:,1), pos(:,2), vel(:,1), vel(:,2), 'off', 'Marker', '.', 'ShowArrowHead', 'on', 'MarkerSize', 10);
    hold on;
    
    for t = 1:Time
        % Update positions
        pos = pos + v0 * [cos(theta) sin(theta)] * dt;

        % Apply periodic boundary conditions
        pos = mod(pos, L);

        % Compute the new orientations
        new_theta = theta;
        for i = 1:N
            % Find neighbors within the interaction radius
            dist = sqrt((pos(:,1) - pos(i,1)).^2 + (pos(:,2) - pos(i,2)).^2);
            neighbors = dist <= RI;
            % Compute the average direction of motion of the neighbors
            avg_theta = atan2(mean(sin(theta(neighbors))), mean(cos(theta(neighbors))));
            % Add noise
            new_theta(i) = avg_theta + sig * (rand - 0.5);
        end

        % Update theta
        theta = new_theta;

        % Store data
        pos_data(:,:,t) = pos;
        theta_data(:,t) = theta;

        % Update visualization
        vel = v0 * [cos(theta) sin(theta)];
        quiver(pos(:,1), pos(:,2), vel(:,1), vel(:,2), 'off', 'Marker', '.', 'ShowArrowHead', 'on', 'MarkerSize', 10);
        axis equal;
        axis([0 L 0 L]);
        axis off
        hold off

        drawnow limitrate;
    end
    
    % Save results
    save([params(p).name '.mat'], 'pos_data', 'theta_data');
    
    % Calculate diagnostics
    % Order Parameter
    order_param = zeros(1, Time);
    for t = 1:Time
        vel = v0 * [cos(theta_data(:,t)) sin(theta_data(:,t))];
        order_param(t) = norm(sum(vel, 1)) / (N * v0);
    end
    diagnostics(p).OrderParameter = order_param;
    
    % Velocity Correlation
    vel_corr = zeros(1, Time-1);
    for t = 1:Time-1
        vel_t = v0 * [cos(theta_data(:,t)) sin(theta_data(:,t))];
        vel_t1 = v0 * [cos(theta_data(:,t+1)) sin(theta_data(:,t+1))];
        vel_corr(t) = mean(dot(vel_t, vel_t1, 2));
    end
    diagnostics(p).VelocityCorrelation = vel_corr;
    
    % Position Correlation
    pos_corr = zeros(1, Time-1);
    for t = 1:Time-1
        pos_corr(t) = mean(sqrt(sum((pos_data(:,:,t) - pos_data(:,:,t+1)).^2, 2)));
    end
    diagnostics(p).PositionCorrelation = pos_corr;
end

% Plot diagnostics
figure;
for p = 1:length(params)
    subplot(3,1,1);
    plot(diagnostics(p).OrderParameter);
    hold on;
    title('Order Parameter');
    legend({params.name});
    
    subplot(3,1,2);
    plot(diagnostics(p).VelocityCorrelation);
    hold on;
    title('Velocity Correlation');
    legend({params.name});
    
    subplot(3,1,3);
    plot(diagnostics(p).PositionCorrelation);
    hold on;
    title('Position Correlation');
    legend({params.name});
end