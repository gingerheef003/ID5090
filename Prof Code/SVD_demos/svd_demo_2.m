clc
clear
close

%% SVD for a picture
addpath('DATA')

A = imread('dog.jpg');

X=double(rgb2gray(A));  % Convert RBG to gray, 256 bit to double.
nx = size(X,1); ny = size(X,2);

imshow(uint8(X))
axis equal
%% SVD

[U,S,V] = svd(X);

figure, subplot(2,2,1)
imagesc(X), axis off, colormap gray 
axis equal
title('Original')

plotind = 2;
for r=[2 10 50]  % Truncation value
    
    Xapprox = U(:,1:r)*S(1:r,1:r)*V(:,1:r)'; % Approx. image

    subplot(2,2,plotind), plotind = plotind + 1;
    imagesc(Xapprox), axis off
    axis equal
    title(['r=',num2str(r,'%d'),', ',num2str(100*r*(nx+ny)/(nx*ny),'%2.2f'),'% storage']);
end
set(gcf,'Position',[100 100 550 400])

%%
semilogy(diag(S))
xlim([1 100])