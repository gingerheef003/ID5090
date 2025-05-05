%% Setting up the vicsek code
% Parameters
N = 100; % Number of agents
Time = 350; % Time steps of simulation
dt = 0.001; % Time step size
v0 = 100; % Constant velocity
L = 2; % Domain size
% sig = pi/12; % Noise strength
% RI = 1; % Radius of interaction
show_plot = false; % show quiver plot
show_pol_plot = true; % show polarization vector
show_pol_graph = false; % show polarization graph
show_pol_mag_graph = false; % show polarization magnitude graph
save_mat = false; % save variables in mat file
export_data = true;

function [pos, th, pol] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat, export_data)
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

        if t == 50 || t == 200 || t == 350
            edge_list = [];
            for i = 1:N
                for j = i+1:N
                    if neighbors(i,j)
                        edge_list = [edge_list; i, j];
                    end
                end
            end
            writematrix(edge_list, sprintf('edges_%d.csv', t))
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
params = struct('sig', {10*pi/180, 30*pi/180}, 'RI', {0.20, 0.05});

for p=1:length(params)
    RI = params(p).RI;
    sig = params(p).sig;

    [pos, th] = run_vicsek(N, L, v0, RI, sig, Time, dt, show_plot, show_pol_plot, show_pol_graph, show_pol_mag_graph, save_mat, export_data);
    disp('Done with 1');
    pause;
end

