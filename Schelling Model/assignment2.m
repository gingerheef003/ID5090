clc
clear
close all

% Parameters
N = 100; % number of rows and columns
p = 0.3; % tolerance of agents
e = 0.1; % fraction of empty spaces
f = 0.5; % fraction of group 1
max_iter = 5000;

show_frac_satis = 0;
show_modularity = 0;
show_segregation_coeff = 0;

% [A, n_e] = initialize_grid(N, e, f);
% [A, segregation_index, avg_similarity, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis, show_modularity, show_segregation_coeff);
% visualize_results(A, segregation_index, avg_similarity, show_frac_satis, frac_satis, show_modularity, modularity);


function [A, n_e] = initialize_grid(N, e, f)
    n_e = floor(N^2 * e);
    g = N^2 - n_e;
    g1 = floor(f*g);
    g2 = g - g1;

    A = zeros(N,N);
    randomInd = randperm(N*N, g1+g2);
    A(randomInd(1:g1)) = 1;
    A(randomInd(g1+1:end)) = 2;
end

function [A, segregation_index, avg_similarity, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis, show_modularity, show_segregation_coeff)
    N = size(A, 1);
    g = nnz(A ~= 0);
    kernel = [1 1 1; 1 0 1; 1 1 1];
    segregation_index = zeros(max_iter, 1);
    avg_similarity = zeros(max_iter, 1);
    frac_satis = zeros(max_iter, 1);
    modularity = zeros(max_iter, 1);

    % CHANGE: Created a figure for real-time visualization
    figure('Position', [100, 100, 800, 400]);

    for t = 1:max_iter
        A_padded = padarray(A, [1 1], 'circular', 'both');
        neigh1 = conv2(A_padded == 1, kernel, 'valid');
        neigh2 = conv2(A_padded == 2, kernel, 'valid');
        neighs = conv2(A_padded ~= 0, kernel, 'valid');
        
        % CHANGE: Vectorized operation for identifying dissatisfied agents
        diss1 = (A == 1) & (neigh1 < p*neighs);
        diss2 = (A == 2) & (neigh2 < p*neighs);
        A(diss1 | diss2) = 0;
        
        n_diss1 = nnz(diss1);
        n_diss2 = nnz(diss2);
        n_vac = n_diss1 + n_diss2 + n_e;
        
        filling = zeros(n_vac, 1);
        filling(1:n_diss1) = 1;
        filling(n_diss1+1:n_diss1+n_diss2) = 2;
        idx = randperm(n_vac);
        filling = filling(idx);
        A(A == 0) = filling;

        % CHANGE: Calculate and store metrics
        segregation_index(t) = calculate_segregation_index(A);
        avg_similarity(t) = mean([(neigh1(A==1) ./ neighs(A==1)); (neigh2(A==2) ./ neighs(A==2))]);
        
        if show_frac_satis
            satis = g - n_diss1 - n_diss2; 
            frac_satis(t) = satis / g;
        end
        if show_modularity
            modularity(t) = compute_modularity(A);
        end

        % CHANGE: Visualize in real-time
        subplot(1,2,1);
        imagesc(A);
        colormap([1 1 1; 1 0 0; 0 0 1]);
        axis equal off;
        title(sprintf('Iteration %d', t));

        subplot(1,2,2);
        plot(1:t, segregation_index(1:t), 'b-', 1:t, avg_similarity(1:t), 'r-', 1:t, frac_satis(1:t), 'g-', 'LineWidth', 2);
        legend('Segregation Index', 'Avg Similarity', 'Fraction Satisfied');
        xlabel('Iteration');
        ylabel('Metric Value');
        title('Segregation Metrics');

        drawnow;

        if n_diss1 == 0 && n_diss2 == 0
            break;
        end

    end
    segregation_index = segregation_index(1:t);
    avg_similarity = avg_similarity(1:t);
    frac_satis = frac_satis(1:t);
    modularity = modularity(1:t);

    if show_segregation_coeff
        disp('Segregation Coefficient')
        disp(compute_segregation_coefficient(A))
    end
end

% CHANGE: Added new function to calculate segregation index
function index = calculate_segregation_index(A)
    horizontal_edges = sum(sum(diff(A,1,2)~=0));
    horizontal_same = sum(sum(diff(A,1,2)==0 & A(:,1:end-1)~=0));
    
    vertical_edges = sum(sum(diff(A,1,1)~=0));
    vertical_same = sum(sum(diff(A,1,1)==0 & A(1:end-1,:)~=0));
    
    total_edges = horizontal_edges + vertical_edges;
    same_group_edges = horizontal_same + vertical_same;
    
    index = same_group_edges / total_edges;
end

function Q = compute_modularity(A)
    [N, ~] = size(A);

    [row, col] = find(A ~= 0);
    num_nodes = length(row);

    Adj = zeros(num_nodes, num_nodes);

    offsets = [-1, -1; -1, 0; -1, 1;
                0, -1;        0, 1;
                1, -1;  1, 0;  1, 1];

    k = zeros(num_nodes, 1);
    m = 0;

    for i = 1:num_nodes
        for j = i+1:num_nodes
            is_neighbor = false;
            for idx = 1:size(offsets, 1)
                neighbor_row = mod(row(i) + offsets(idx, 1) - 1, N) + 1;
                neighbor_col = mod(col(i) + offsets(idx, 2) - 1, N) + 1;

                if neighbor_row == row(j) && neighbor_col == col(j)
                    is_neighbor = true;
                    break;
                end
            end
            if is_neighbor
                Adj(i, j) = 1;
                Adj(j, i) = 1;
                k(i) = k(i) + 1;
                k(j) = k(j) + 1;
                m = m + 1;
            end
        end
    end

    Q = 0;
    for i = 1:num_nodes
        for j = 1:num_nodes
            if A(row(i), col(i)) == A(row(j), col(j))
                Q = Q + (Adj(i, j) - (k(i) * k(j)) / (2 * m));
            end
        end
    end

    Q = Q / (2 * m);
end

function visualize_results(A, segregation_index, avg_similarity, show_frac_satis, frac_satis, show_modularity, modularity)
    figure('Position', [100, 100, 800, 600]);

    subplot(2,2,1);
    imagesc(A);
    colormap([1 1 1; 1 0 0; 0 0 1]);
    axis equal off;
    title('Final Population Map');

    subplot(2,2,2);
    histogram(A(A~=0));
    title('Population Distribution');
    xlabel('Group');
    ylabel('Count');

    subplot(2,2,3);
    plot(segregation_index, 'b-');
    title('Segregation Index Over Time');
    xlabel('Iteration');
    ylabel('Segregation Index');

    subplot(2,2,4);
    plot(avg_similarity, 'r-');
    title('Average Similarity Over Time');
    xlabel('Iteration');
    ylabel('Average Similarity');

    if show_frac_satis
        figure;
        plot(frac_satis, 'g-');
        title('Fraction Satisfied vs Time');
        xlabel('Iteration');
        ylabel('Fraction Satisfied');
    end

    if show_modularity
        figure;
        plot(modularity, 'b-');
        title('Modularity vs Time');
        xlabel('Iteration');
        ylabel('Modularity');
    end
    
end

function clusters = identify_clusters(A)
    % Identify clusters using DBSCAN
    % Convert matrix A to a list of points
    [row, col] = find(A ~= 0);
    data = [row, col];

    % Parameters for DBSCAN
    epsilon = 1; % Maximum distance to be considered as a neighbor
    minPts = 2; % Minimum number of points to form a dense region

    % Run DBSCAN
    clusters = dbscan(data, epsilon, minPts);
end

function s = compute_segregation_coefficient(A)
    total_agents = nnz(A);
    clusters = bwconncomp(A ~= 0);

    cluster_sizes = cellfun(@numel, clusters.PixelIdxList);
    sum_of_squares = sum(cluster_sizes.^2);

    s = (2 * sum_of_squares) / (total_agents^2);
end



%% Problem 1
N = 50;
p = 0.65;
e = 0.75;
f = 0.5;

show_frac_satis = 1;
show_modularity = 1;

[A, n_e] = initialize_grid(N, e, f);
[A, segregation_index, avg_similarity, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis, show_modularity, show_segregation_coeff);
visualize_results(A, segregation_index, avg_similarity, show_frac_satis, frac_satis, show_modularity, modularity);

%% Problem 2
N = 100;
p = 0.3;
e = 0.1;
f = 0.5;

show_frac_satis = 0;
show_modularity = 0;

[A, n_e] = initialize_grid(N, e, f);
[A, segregation_index, avg_similarity, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis, show_modularity, show_segregation_coeff);
visualize_results(A, segregation_index, avg_similarity, show_frac_satis, frac_satis, show_modularity, modularity);

%% Problem 3
show_segregation_coeff = 1;

[A, n_e] = initialize_grid(N, e, f);
[A, segregation_index, avg_similarity, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis, show_modularity, show_segregation_coeff);