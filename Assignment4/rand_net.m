N = 100;
p = 0.0202;

nodes = 1:N;
edges = [];
for i=1:N
    for j=i+1:N
        r = rand;
        if r < p
            edges = [edges; i, j];
        end
    end
end

writematrix(edges, 'edges2.csv');
writematrix(nodes, 'nodes2.csv');