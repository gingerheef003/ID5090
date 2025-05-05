N = 100; 

A = zeros(N, N);
A(1,2) = 1;
A(2,1) = 1;
edges = [2, 1];

for t=3:N
    degrees = sum(A, 2);
    probabilities = degrees / sum(degrees);
    
    target = randsample(1:t-1, 1, true, probabilities(1:t-1));
    
    A(t, target) = 1;
    A(target, t) = 1;
    
    edges = [edges; t, target];
end

writematrix(edges, 'edges3.csv');