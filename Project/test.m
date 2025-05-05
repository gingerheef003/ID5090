clc;
close;
clear all;

N = 100;
L = 50;
T = 1000;

a = zeros(N);

posX = randi(L,1,N);
posY = randi(L,1,N);

right = randi(3,1,N) - 2;
up = randi(3,1,N) - 2;

figure(1);
scatter(posX, posY, 'filled');
axis equal;
axis off;
axis([0 L 0 L])

for t=2:T
    posX = mod(posX + right, L);
    posY = mod(posY + up, L);

    figure(1);
    scatter(posX, posY, 'filled');
    axis equal;
    axis off;
    axis([0 L 0 L])
end

