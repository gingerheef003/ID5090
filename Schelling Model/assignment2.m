

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

% [A, n_e] = initialize_grid(N, e, f);
% [A, frac_satis, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis);

%% Function definitions

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

function [A, modularity] = run_simulation(A, p, max_iter, n_e, show_frac_satis)
    N = size(A, 1);
    g = nnz(A ~= 0);
    kernel = [1 1 1; 1 0 1; 1 1 1];

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
        
        if show_frac_satis
            satis = g - n_diss1 - n_diss2; 
            frac_satis(t) = satis / g;
        end

        imagesc(A);
        colormap([1 1 1; 1 0 0; 0 0 1]);
        axis equal off;
        title(sprintf('Iteration %d', t));
        drawnow;

        if n_diss1 == 0 && n_diss2 == 0
            break;
        end

    end

    frac_satis = frac_satis(1:t);
    modularity = modularity(1:t);

    if show_frac_satis
        figure;
        plot(frac_satis, 'b-');
        title('Fraction Satisfied vs Time');
        xlabel('Iteration');
        ylabel('Fraction Satisfied');
    end
end

function Adj = create_adjacency_matrix(A)
    [N, ~] = size(A);
    [row, col] = find(A ~= 0);
    num_nodes = length(row);

    Adj = zeros(num_nodes, num_nodes);

    offsets = [-1, -1; -1, 0; -1, 1;
                0, -1;        0, 1;
                1, -1;  1, 0;  1, 1];

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
            end
        end
    end
end

function Q = compute_modularity(A)

    [row, col] = find(A ~= 0);
    num_nodes = length(row);

    Adj = create_adjacency_matrix(A);

    k = sum(Adj, 2);
    m = sum(k) / 2;

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

function [cluster_sizes1, cluster_sizes2, segregation_coeff] = compute_segregation_coefficient(A)
    g = nnz(A);

    clusters1 = bwconncomp(A == 1);
    cluster_sizes1 = cellfun(@numel, clusters1.PixelIdxList);

    clusters2 = bwconncomp(A == 2);
    cluster_sizes2 = cellfun(@numel, clusters2.PixelIdxList);

    sum_sq = sum(cluster_sizes1.^2) + sum(cluster_sizes2.^2);
    segregation_coeff = (2 * sum_sq) / (g^2);
end


%% Problem 1
N = 50;
p = 0.65;
e = 0.0075;
f = 0.5;

show_frac_satis = 1;

[A, n_e] = initialize_grid(N, e, f);
A =  run_simulation(A, p, max_iter, n_e, show_frac_satis);

disp('Modularity')
disp(compute_modularity(A));

%% Problem 2
N = 100;
p = 0.3;
e = 0.1;
f = 0.5;

show_frac_satis = 0;
[A, n_e] = initialize_grid(N, e, f);
run_simulation(A, p, max_iter, n_e, show_frac_satis);

%% Problem 3
N = 100;
p = 0.3;
e = 0.1;
f = 0.5;

show_frac_satis = 0;

[A, n_e] = initialize_grid(N, e, f);
A = run_simulation(A, p, max_iter, n_e, show_frac_satis);
[cluster_sizes1, cluster_sizes2, segregation_coeff] = compute_segregation_coefficient(A);

disp('Cluster sizes of group 1');
disp(cluster_sizes1);
disp('Cluster sizes of group 2');
disp(cluster_sizes2);

disp('Segregation Coeff');
disp(segregation_coeff)

disp('Modularity')
disp(compute_modularity(A));

%% Problem 4
N = 100;

show_frac_satis = 0;

p = 0.75;
e = 0.8;
f = 0.5;
[A, n_e] = initialize_grid(N, e, f);
A = run_simulation(A, p, max_iter, n_e, show_frac_satis,);
[~, ~, segregation_coeff] = compute_segregation_coefficient(A);

disp('Segregation Coeff (case 1)');
disp(segregation_coeff);

p = 0.75;
e = 0.1;
f = 0.1;
[A, n_e] = initialize_grid(N, e, f);
A = run_simulation(A, p, max_iter, n_e, show_frac_satis);
[~, ~, segregation_coeff] = compute_segregation_coefficient(A);

disp('Segregation Coeff (case 2)');
disp(segregation_coeff);

p = 0.2;
e = 0.1;
f = 0.5;
[A, n_e] = initialize_grid(N, e, f);
A = run_simulation(A, p, max_iter, n_e, show_frac_satis);
[~, ~, segregation_coeff] = compute_segregation_coefficient(A);

disp('Segregation Coeff (case 3)');
disp(segregation_coeff);

%% Problem 5
p = 0.3;
e = 0.1;
f = 0.5;
Ns = [10, 25, 50, 75, 100, 125, 150, 175, 200];

segregation_coeffs = zeros(size(Ns));
average_cluster_sizes = zeros(size(Ns));

for idx = 1:length(Ns)
    N = Ns(idx); 
    
    [A, n_e] = initialize_grid(N, e, f);
    A = run_simulation(A, p, max_iter, n_e, false, false);
    
    [cluster_sizes1, cluster_sizes2, segregation_coeff] = compute_segregation_coefficient(A);
    segregation_coeffs(idx) = segregation_coeff;
    
    average_cluster_sizes(idx) = mean([cluster_sizes1 cluster_sizes2]); % Average size
    
    fprintf('Domain Size: %d x %d\n', N, N);
    fprintf('Segregation Coefficient: %.4f\n', segregation_coeff);
    fprintf('Average Cluster Size: %.2f\n\n', average_cluster_sizes(idx));
end

% Plot results
figure;
subplot(2, 1, 1);
plot(Ns, segregation_coeffs, 'o-', 'LineWidth', 2);
xlabel('Domain Size (N)');
ylabel('Segregation Coefficient');
title('Scaling of Segregation Coefficient with Domain Size');
grid on;

subplot(2, 1, 2);
plot(Ns, average_cluster_sizes, 'o-', 'LineWidth', 2);
xlabel('Domain Size (N)');
ylabel('Average Cluster Size');
title('Scaling of Average Cluster Size with Domain Size');
grid on;