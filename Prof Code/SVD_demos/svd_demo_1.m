clc
clear
close

%% SVD for a random matrix

A = rand(10, 5);

[U, S, V] = svd(A);

disp('U'); disp(U)
disp('S'); disp(S) 
disp('V'); disp(V)

%% SVD reduced

[U, S, V] = svd(A, 'econ');

disp('U'); disp(U)
disp('S'); disp(S) 
disp('V'); disp(V)

%% SVD approximation

figure

plot(diag(S),'-s')
hold on

FN = zeros(size(A,2),1);

for p = 1:5
    A_rank_p = U(:,1:p) * S(1:p,1:p) * V(:,1:p)';

    disp('difference')
    disp((A - A_rank_p))
    
    FN(p) = sum((A - A_rank_p).^2, "all");

end

plot(FN, '--ks')

ylabel('$\sigma$, Frobenius norm', 'Interpreter','latex');