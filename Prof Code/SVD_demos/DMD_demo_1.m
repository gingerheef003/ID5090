clc
clear

%% Data loading and reshaping

load('spiralwaves.mat')

u = reshape(u,n^2,length(tspan));
v = reshape(v,n^2,length(tspan));
F = [u; v]; % data matrix

X = F(:,1:end-1);
Y = F(:,2:end);

%% DMD matrix
A = Y * pinv(X);

%% Eigen values and vectors
[v, e]=eig(A);

%%
plot(real(diag(e)), imag(diag(e)),'or')
hold on
axis equal
plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi), 'k')

%%

Xt = X(:,1);
for t = 2:10000
    Xt = A * Xt;

    if rem(t,10) == 0
        u_fore = reshape(Xt(1:end/2), n,n);
        pcolor(x, y, u_fore)

        shading interp
        legend off
        colorbar
        axis equal

        xlim([-L/2 L/2])
        ylim([-L/2 L/2])

        drawnow limitrate
    end
end


