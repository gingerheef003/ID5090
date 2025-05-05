%% Setting up the vicsek code
% Parameters
N = 500; % Number of agents
Time = 10000; % Time steps of simulation
dt = 1; % Time step size
sig = pi/12; % Noise strength
v0 = 0.05; % Constant velocity
L = 10; % Domain size
RI = 1; % Radius of interaction
show_plot = false; % show quiver plot
show_pol_plot = true; % show polarization vector
show_pol_graph = false; % show polarization graph
show_pol_mag_graph = false; % show polarization magnitude graph
save_mat = false; % save variables in mat file

function [pos, th, pol] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat)
    % Initial conditions
    pos = zeros(N, 2, Time);
    th = zeros(N, Time);
    pol = zeros(2, Time);

    pos(:,:,1) = rand(N, 2) * L;
    th(:,1) = rand(N, 1) * 2 * pi;
    polarization_vector = mean(exp(1i * th(:,1)));
    pol(1, 1) = abs(polarization_vector);
    pol(2, 1) = angle(polarization_vector);
    

    if show_plot
        vel = v0 * [cos(th(:,1)) sin(th(:,1))];
        figure;
        axis equal;
        axis([0 L 0 L]);
        quiver(pos(:,1,1), pos(:,2,1), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale", "on", "AutoScaleFactor", 0.5);
        hold on;
    end

    for t = 2:Time
        % Update positions
        vel = v0 * [cos(th(:,t-1)) sin(th(:,t-1))];
        pos(:,:,t) = pos(:,:,t-1) + vel * dt;

        % Apply periodic boundary conditions
        pos(:,:,t) = mod(pos(:,:,t), L);

        % Compute the new orientations
        distances = pdist2(pos(:,:,t), pos(:,:,t));
        neighbors = distances <= RI;

        sin_th = sin(th(:,t-1));
        cos_th = cos(th(:,t-1));

        for i = 1:N
            neighbor_idx = neighbors(i, :);
            th_avg = atan2(mean(sin_th(neighbor_idx)), mean(cos_th(neighbor_idx)));
            th(i,t) = th_avg + sig * (rand - 0.5); % Noise
        end

        % Calculate and store polarization parameter
        polarization_vector = mean(exp(1i * th(:,t)));
        pol(1, t) = abs(polarization_vector); % Magnitude
        pol(2, t) = angle(polarization_vector); % Angle

        % Update visualization
        if show_plot
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
    

    if show_pol_graph
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

    if show_pol_mag_graph
        figure;
        plot(1:Time, pol(1,:));
        xlabel('Time');
        ylabel('Polarization Magnitude');
        title('Polarization Magnitude Over Time');
    end

end

%% Run the simulation
[pos, th] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat);


%% Problem 1
params = struct('name', {'GlobalFlocking', 'LocalFlocking', 'NoFlocking'}, ...
                'RI', {3, 1, 0}, ...
                'sig', {pi/24, pi/12, pi/6});

for p = 1:length(params)
    fprintf('%s\n', params(p).name);
    RI = params(p).RI;
    sig = params(p).sig;

    [pos, th] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat);

    save([params(p).name '.mat'], 'pos', 'th');
end

%% Visualize Flocking from .mat file
function visualize_simulation(file_name)
    % Load the data
    data = load(file_name);
    pos_data = data.pos;
    th_data = data.th;

    % Parameters
    [~, Time] = size(th_data);
    v0 = 0.05; % Constant velocity
    L = 10; % Domain size

    % Create figure for animation
    figure;
    axis equal;
    axis([0 L 0 L]);

    for t = 1:Time
        pos = pos_data(:, :, t);
        theta = th_data(:, t);
    
        vel = v0 * [cos(theta) sin(theta)];
    
        avg_vel = mean(vel);
        polarization_magnitude = norm(avg_vel);
        polarization_angle = atan2(avg_vel(2), avg_vel(1));
    
        quiver(pos(:,1), pos(:,2), vel(:,1), vel(:,2), 'off', 'Marker', 'none', 'ShowArrowHead', 'on', 'MarkerSize', 10, "AutoScale","on", "AutoScaleFactor",0.5);
        hold on;
        quiver(L/2, L/2, polarization_magnitude * cos(polarization_angle), polarization_magnitude * sin(polarization_angle), 'r', 'LineWidth', 2, 'MaxHeadSize', 2, AutoScale='on', AutoScaleFactor='1');
        axis equal;
        axis([0 L 0 L]);
        axis off
        hold off
        drawnow limitrate;
    end
end

visualize_simulation('GlobalFlocking.mat')

%% Problem 2
function analyze_theta_differences(file_name)
    data = load(file_name);
    th_data = data.th;

    [N, Time] = size(th_data);
    theta_diff = zeros(N, Time);

    for t = 1:Time
        pol = angle(mean(exp(1i * th_data(:, t))));
        theta_diff(:, t) = th_data(:, t) - pol;
    end

    avg_theta_diff = mean(theta_diff, 2);

    % Plot the histogram of the average differences
    figure;
    histogram(avg_theta_diff, 100, 'Normalization', 'pdf');
    xlabel('Difference of \theta to Polarization Parameter');
    ylabel('Probability Density');
    title('Histogram of \theta Differences');
end

analyze_theta_differences('NoFlocking.mat')

%% Problem 3
conditions = struct('name', {'0', 'pi/24', 'pi/12', 'pi/6', 'pi'}, ...
                    'sig', {0, pi/24, pi/12, pi/6, pi});

figure;
hold on;
xlabel('Time');
ylabel('Polarization Parameter');
title('Polarization Parameter Over Time for Different Conditions');
legend;
for c = 1:length(conditions)
    fprintf('Condition: %s\n', conditions(c).name);
    sig = conditions(c).sig;

    [pos, th, pol] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat);
    plot(1:Time, pol(1,:), 'DisplayName', conditions(c).name);
end
hold off

%% Problem 4
sig_values = [0 pi/24 pi/12 pi/6 pi/3 pi/2];
num_realizations = 10; % Number of Monte Carlo realizations
avg_final_polarization = zeros(1, length(sig_values));


for s = 1:length(sig_values)
    sig = sig_values(s);
    final_polarization = zeros(1, num_realizations);
    for r = 1:num_realizations
        fprintf('Noise Level: %.2f, Realization: %d/%d\n', sig, r, num_realizations);

        [pos, th, pol] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat);
        final_polarization(r) = pol(1,end);
    end
    avg_final_polarization(s) = mean(final_polarization);
end

figure;
plot(sig_values, avg_final_polarization, '-o');
xlabel('Noise Strength (\sigma)');
ylabel('Final Polarization');
title('Final Polarization vs. Noise Strength');
grid on;