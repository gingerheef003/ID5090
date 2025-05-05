% clc
% 
% 
% function Adj = create_adjacency_matrix(A)
%     [N, ~] = size(A);
%     [row, col] = find(A ~= 0);
%     num_nodes = length(row);
% 
%     Adj = zeros(num_nodes, num_nodes);
% 
%     offsets = [-1, -1; -1, 0; -1, 1;
%                 0, -1;        0, 1;
%                 1, -1;  1, 0;  1, 1];
% 
%     for i = 1:num_nodes
%         for j = i+1:num_nodes
%             is_neighbor = false;
%             for idx = 1:size(offsets, 1)
%                 neighbor_row = mod(row(i) + offsets(idx, 1) - 1, N) + 1;
%                 neighbor_col = mod(col(i) + offsets(idx, 2) - 1, N) + 1;
% 
%                 if neighbor_row == row(j) && neighbor_col == col(j)
%                     is_neighbor = true;
%                     break;
%                 end
%             end
%             if is_neighbor
%                 Adj(i, j) = 1;
%                 Adj(j, i) = 1;
%             end
%         end
%     end
% end
% 
% A = [
%     1 1 1 0;
%     1 0 0 0;
%     0 0 1 0;
%     0 0 0 0;
% ];
% 
% function clusters = identify_clusters(A, epsilon, minPts)
%     % Identify clusters using DBSCAN based on an adjacency matrix.
%     %
%     % Inputs:
%     %   adjacency_matrix - NxN matrix (1 indicates connection, 0 otherwise)
%     %   epsilon          - Maximum distance to be considered as a neighbor
%     %   minPts           - Minimum number of points to form a dense region
%     %
%     % Output:
%     %   clusters         - Cluster labels for each node (-1 for noise)
% 
%     % Convert adjacency matrix to coordinates (edge list)
%     Adj = create_adjacency_matrix(A)
%     [row, col] = find(Adj);
%     edges = [row, col]; % Each row represents an edge between two nodes
% 
%     clus = bwconncomp(A ~= 0);
%     clus_sizes = cellfun(@numel, clus.PixelIdxList)
% 
%     % Use DBSCAN to cluster based on edge distances
%     clusters = dbscan(edges, epsilon, minPts);
%     [edges clusters]
% end
% 
% clusters = identify_clusters(A,1,1);
% 
% 
% 
% 
% % A = [
% %     0 1 0 2;
% %     1 0 2 0;
% %     0 2 1 0;
% % 2 0 0 1];










% 
% 
% function [adjacency_matrix, clusters] = matrix_clustering(input_matrix)
%     N = size(input_matrix, 1);
%     non_empty_indices = find(input_matrix ~= 0);
%     num_non_empty = length(non_empty_indices);
% 
%     % Create adjacency matrix
%     adjacency_matrix = zeros(num_non_empty);
% 
%     for i = 1:num_non_empty
%         for j = i+1:num_non_empty
%             [row_i, col_i] = ind2sub([N, N], non_empty_indices(i));
%             [row_j, col_j] = ind2sub([N, N], non_empty_indices(j));
% 
%             % Check periodic boundaries
%             dx = min(abs(row_i - row_j), N - abs(row_i - row_j));
%             dy = min(abs(col_i - col_j), N - abs(col_i - col_j));
% 
%             % Consider adjacent if within 1 step in any direction
%             if dx <= 1 && dy <= 1
%                 adjacency_matrix(i, j) = 1;
%                 adjacency_matrix(j, i) = 1;
%             end
%         end
%     end
% 
%     % Perform DBSCAN clustering
%     epsilon = 1.5; % Maximum distance between two samples for one to be considered as in the neighborhood of the other
%     min_samples = 2; % Minimum number of samples in a neighborhood for a point to be considered as a core point
% 
%     clusters = dbscan(adjacency_matrix, epsilon, min_samples);
% end
% 
% 
% % Example usage
% N = 10;
% input_matrix = randi([0, 2], N, N); % Random matrix with 0, 1, and 2
% [adj_matrix, clusters] = matrix_clustering(input_matrix);
% 
% % Visualize the results
% figure;
% subplot(1, 2, 1);
% colormap([1 1 1; 1 0 0; 0 0 1]);
% imagesc(input_matrix);
% title('Input Matrix');
% colorbar;
% 
% subplot(1, 2, 2);
% scatter(1:length(clusters), clusters, 50, clusters, 'filled');
% colormap([1 1 1; 1 0 0; 0 0 1]);
% title('Cluster Assignments');
% colorbar;

function [idx, corepts] = periodic_dbscan(X, epsilon, minpts)
    N = size(X, 1);
    
    % Create extended matrix with periodic boundaries
    X_extended = [X X(:,1:epsilon); 
                  X(1:epsilon,:) X(1:epsilon,1:epsilon)];
    
    % Convert matrix to point cloud
    [row, col, val] = find(X_extended);
    points = [row col];
    
    % Apply DBSCAN
    [idx_extended, corepts] = dbscan(points, epsilon, minpts, 'Distance', 'cityblock')
    
    % Extract labels for original matrix
    idx = idx_extended(1:N^2);
    idx = reshape(idx, N, N);
    
    % Resolve periodic clusters
    unique_labels = unique(idx);
    unique_labels(unique_labels == -1) = [];
    for i = 1:length(unique_labels)
        label = unique_labels(i);
        [r, c] = find(idx == label);
        if any(r <= epsilon) && any(r > N-epsilon)
            idx(idx == label) = i;
        elseif any(c <= epsilon) && any(c > N-epsilon)
            idx(idx == label) = i;
        end
    end
end

% Example usage
N = 10;
X = randi([0 2], N, N);
[idx, corepts] = periodic_dbscan(X, 1, 4);

% Visualize results
figure;
subplot(1,2,1);
imagesc(X);
title('Original Matrix');
colorbar;

subplot(1,2,2);
imagesc(idx);
title('Clustered Matrix');
colorbar;
