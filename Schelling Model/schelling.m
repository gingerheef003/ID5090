clc
clear
close all

N = 100; % number of rows and columns
p = 0.75; % tolerance of agents
e = 0.1; % fraction of empty spaces
f = 0.5; % fraction of group 1
colormap([1 1 1; 1 0 0; 0 0 1]); % white(0: empty), red(1: group1), blue(2: group2)
max_iter = 5000;


n_e = floor(N^2 * e);
g = N^2 - n_e;
g1 = floor(f*g);
g2 = g - g1;

A = zeros(N,N);
randomInd = randperm(N*N, g1+g2);
A(randomInd(1:g1)) = 1;
A(randomInd(g1+1:end)) = 2;

figure(1);
imagesc(A);
axis equal;
axis off;
title('Population Map');
pause;

kernel = [
    1 1 1;
    1 0 1;
    1 1 1
];
n_diss1 = 1;
n_diss2 = 1;

for t=1:max_iter
    neigh1 = conv2(A == 1, kernel, 'same');
    neigh2 = conv2(A == 2, kernel, 'same');
    neighs = conv2(A ~= 0, kernel, 'same');
    
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
    size(filling)
    size(A(A==0))
    A(A == 0) = filling;

    figure(1);
    imagesc(A);
    axis equal;
    axis off;
    title(num2str(t));

    if n_diss1 == 0 && n_diss2 == 0
        break;
    end
    
    pause;
end