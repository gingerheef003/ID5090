% Time series models -- intro
clc
clear
close all

Time = 1e4;
w = zeros(Time,1);
x = zeros(Time,1);

%% Model 1 (white noise only)
for i=2:Time
    w(i-1) = randn(1,1);
    % w(i-1) = unfrnd(-1,1);
    x(i) = w(i-1);
end

figure
subplot(1,2,1)
plot(x)
subplot(1,2,2)
autocorr(x)

%% Model 2 (moving average)
for i=2:Time
    w(i) = randn(1,1);
    % w(i) = unfrnd(-1,1);
    x(i) = w(i) + 20*w(i-1);
end

figure
subplot(1,2,1)
plot(x)
subplot(1,2,2)
autocorr(x)


%% Model 3 (auto regression)
for i=2:Time
    w(i) = randn(1,1);
    % w(i) = unfrnd(-1,1);
    x(i) = w(i) + 0.3*x(i-1);
end

figure
subplot(1,2,1)
plot(x)
subplot(1,2,2)
autocorr(x)