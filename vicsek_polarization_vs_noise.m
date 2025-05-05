clc
clear
close all

% Parameters
N = 500; % Number of agents
Time = 10^4; % Time steps of simulation
dt = 1; % Time step size
v0 = 0.05; % Constant velocity
L = 10; % Domain size
RI = 1; % Radius of interaction
show_plot = false; % Flag to show quiver plot

% Noise strength values to test
sig_values = [0 pi/24 pi/12 pi/6 pi/3 pi/2];
final_polarization = zeros(1, length(sig_values));

for s = 1:length(sig_values)
    sig = sig_values(s);

    % Initial conditions
    pos = rand(N, 2) * L; % Random initial positions within the domain
    theta = rand(N, 1) * 2 * pi; % Random initial orientations
    vel = v0 * [cos(theta) sin(theta)]; % Initial velocities

    % Store data
    pos_data = zeros(N, 2, Time);
    vel_data = zeros(N, 2, Time);
    theta_data = zeros(N, Time);
    polarization_data = zeros(1, Time);

    % Visualization setup
    if show_plot
        figure;
        axis equal;
        axis([0 L 0 L]);
        quiver(pos(:,1), pos(:,2), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale", "on", "AutoScaleFactor", 0.5);
        hold on;
    end

    for t = 1:Time
        % Update positions
        pos = pos + vel * dt;

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

        % Update velocities
        vel = v0 * [cos(new_theta) sin(new_theta)];

        % Update theta
        theta = new_theta;

        % Store data
        pos_data(:,:,t) = pos;
        vel_data(:,:,t) = vel;
        theta_data(:,t) = theta;

        % Calculate and store polarization parameter
        polarization_data(t) = abs(mean(exp(1i * theta)));

        % Update visualization
        if show_plot
            quiver(pos(:,1), pos(:,2), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale", "on", "AutoScaleFactor", 0.5);
            axis equal;
            axis([0 L 0 L]);
            axis off
            hold off
            drawnow limitrate;
        end

        if mod(t, 1000) == 0
            fprintf('Noise Level: %.2f, Iteration: %d/%d\n', sig, t, Time);
        end
    end

    % Store final polarization value for the current noise level
    final_polarization(s) = polarization_data(end);
end

% Plot final polarization values over different noise strengths
figure;
plot(sig_values, final_polarization, '-o');
xlabel('Noise Strength (\sigma)');
ylabel('Final Polarization');
title('Final Polarization vs. Noise Strength');
grid on;