%% Q1
clc
clear
close all

load('data_Q1.mat')

plot(y)

figure
autocorr(y)


%% Q2
clc
clear
close all

load('data_Q2.mat')

mini = min([walker3; walker1; walker2]);
maxi = max([walker3; walker1; walker2]);

A = zeros(maxi, maxi);


for t=2:length(walker1)
    A(walker1(t-1), walker1(t)) = 1;
    A(walker1(t), walker1(t-1)) = 1;
end

for t=2:length(walker2)
    A(walker2(t-1), walker2(t)) = 1;
    A(walker2(t), walker2(t-1)) = 1;
end

for t=2:length(walker3)
    A(walker3(t-1), walker3(t)) = 1;
    A(walker3(t), walker3(t-1)) = 1;
end





%% Q3
clc
clear
close all

load('data_Q3.mat')

k = sum(A,2);
Diag = zeros(76,76);
for i=1:76
    Diag(i,i) = k(i);
end

L = Diag - A;
[V,D] = eig(L);


%% Q4
clc
clear
close all

load('data_Q4.mat')

% Frequency
figure;

subplot(1,2,1)
histogram(x1, 'Normalization', 'pdf'); % Normalized histogram
xlabel('Value');
ylabel('Frequency');
title('Frequency Distribution of x1');
grid on;

subplot(1,2,2)
histogram(x2, 'Normalization', 'pdf'); % Normalized histogram
xlabel('Value');
ylabel('Frequency');
title('Frequency Distribution of x2');
grid on;

% Drift
drift_x1 = diff(x1);
drift_x2 = diff(x2);

figure;

subplot(1,2,1)
plot(x1(1:end-1), drift_x1, '.');
xlabel('Value');
ylabel('Drift');
title('Instantaneous Drift x1');
grid on;

subplot(1,2,2);
plot(x2(1:end-1), drift_x2, '.');
xlabel('Value');
ylabel('Drift');
title('Instantaneous Drift x2');
grid on;

% Diffusion
diff_x1 = sqrt(abs(drift_x1));
diff_x2 = sqrt(abs(drift_x2));

figure

subplot(1,2,1);
plot(x1(1:end-1), diff_x1, '.');
xlabel('Value');
ylabel('Diffusion');
title('Instantaneous Diffusion x1');
grid on;

subplot(1,2,2);
plot(x2(1:end-1), diff_x2, '.');
xlabel('Value');
ylabel('Diffusion');
title('Instantaneous Diffusion x2');
grid on;


%% Q3
clc
clear
close all

load('data_Q5.mat')

figure
scatter(height, weight)



%% Q6
clc
clear
close all

load('data_Q6.mat')

[U, S, V] = svd(V_xyt, 'econ');

figure
title('Three dominant modes')
subplot(2,2,1)
imagesc(reshape(U(:,1),199,449))

subplot(2,2,2)
imagesc(reshape(U(:,2),199,449))

subplot(2,2,3)
imagesc(reshape(U(:,3),199,449))

V_approx = U(:,1:3) * S(1:3,1:3) * V(:,1:3)';

figure
for t=1:151
    imagesc(reshape(V(:,t), 199,449))
    drawnow limitrate
end