clc
clear
close all

% Parameters
N = 500;

Time = 10^4;
dt = 1;

sig = pi/1;

v0 = 0.05;

L = 10;

RI = 1;

show_live_plot = false;
show_polarization_graph = true;

% Initial Conditions
pos = zeros(N,2,Time);
pos(:,:,1) = rand(N*2)*L;

th = zeros(N,Time);
th(:,1) = rand(N,1) * 2*pi;
vel = v0 * [cos(th(:,1)) sin(th(:,1))];

pol = zeros(2,Time);
pol_vec = mean(exp(1i * th(:,1)));
pol(:,1) = [abs(pol_vec) angle(pol_vec)];

% Visualization setup
if show_live_plot
    figure;
    axis equal;
    axis([0 L 0 L]);
    quiver(pos(:,1,1), pos(:,2,1), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale", "on", "AutoScaleFactor", 0.5);
    hold on;
end

for t = 2:Time
    vel = v0*[cos(th(:,t-1)) sin(th(:,t-1))];
    pos(:,:,t) = pos(:,:,t-1) + vel * dt;

    % Apply periodic boundary conditions
    pos(:,:,t) = mod(pos(:,:,t), L);

    distances = pdist2(pos(:,:,t), pos(:,:,t));
    neighbors = distances <= RI;

    sin_th = sin(th(:,t-1));
    cos_th = cos(th(:,t-1));

    for i = 1:N
        neighbor_idx = neighbors(i, :);
        th_avg = atan2(mean(sin_th(neighbor_idx)), mean(cos_th(neighbor_idx)));
        th(i,t) = th_avg + sig * (rand - 0.5); % Noise
    end


    % Update velocities
    vel = v0 * [cos(new_theta) sin(new_theta)];

    pol_vec = mean(exp(1i * th(:,t)));
    pol(:,t) = [abs(pol_vec) angle(pol_vec)];

    % Update visualization
    if show_live_plot
        quiver(pos(:,1,t), pos(:,2,t), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale", "on", "AutoScaleFactor", 0.5);
        hold on
        if show_pol_plot
            quiver(L/2, L/2, pol(1,t) * cos(pol(2,t)), pol(1,t) * sin(pol(2,t)), 'r', 'LineWidth', 2, 'MaxHeadSize', 2, 'AutoScale', 'on', 'AutoScaleFactor', 1);
        end
        axis equal;
        axis([0 L 0 L]);
        axis off
        hold off
        drawnow limitrate;

    end

    if mod(t, 1000) == 0
        fprintf('Iteration: %d/%d\n', t, Time);
    end
end

if save_mat
    save('polarization_data.mat', 'pol');
    save('position_data.mat', 'pos');
    save('theta_data.mat', 'th');
end

if show_pol_mag_graph
    figure;
    plot(1:Time, pol(1,:));
    xlabel('Time');
    ylabel('Polarization Magnitude');
    title('Polarization Magnitude Over Time');
end

if show_pol_2dgraph
    px = mean(cos(th), 1);
    py = mean(sin(th), 1);
    ctrs = {linspace(-1, 1, 50), linspace(-1, 1, 50)};
    figure;
    hist3([px(1000:end)' py(1000:end)'], 'Ctrs', ctrs, 'CdataMode', 'auto');
    xlim([-1, 1]);
    ylim([-1, 1]);
    colorbar;
    view(2);
    shading interp;
    axis equal;
end