% Example time series data
time_series = randn(1, 1000); % Replace with your time series data
dt = 1; % Time step (adjust as needed)

% Calculate instantaneous drift and diffusion
drift = diff(time_series) / dt; % First derivative (drift)
diffusion = sqrt(abs(drift)); % Approximation of diffusion

% Plot instantaneous drift
figure;
plot(time_series(1:end-1), drift, '.');
xlabel('Value');
ylabel('Drift');
title('Instantaneous Drift');
grid on;

% Plot instantaneous diffusion
figure;
plot(time_series(1:end-1), diffusion, '.');
xlabel('Value');
ylabel('Diffusion');
title('Instantaneous Diffusion');
grid on;