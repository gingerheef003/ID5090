N = 20;
A = ones(N);
for i = 1:N
    A(i,i) = 0;
end

plot(graph(A));