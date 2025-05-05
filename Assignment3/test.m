% Set the number of time steps
tsteps = 500;

% Generate white noise sequence (mean 0, variance 1)
wn = randn(tsteps, 1);

% Compute the sample ACF with normalization
[acf, lags] = xcorr(wn, 'coeff');

% Extract non-negative lags only
non_neg_lags = lags >= 0;
acf = acf(non_neg_lags);
lags = lags(non_neg_lags);

% Calculate 95% confidence interval bounds
conf = 1.96 / sqrt(tsteps);

% Plot the ACF
figure;
stem(lags, acf, 'Marker', 'none', 'LineWidth', 1.5);
hold on;
plot(lags, conf * ones(size(lags)), '--r', 'LineWidth', 1.5);
plot(lags, -conf * ones(size(lags)), '--r', 'LineWidth', 1.5);
xlabel('Lag');
ylabel('ACF');
title('Sample Autocorrelation Function of White Noise');
xlim([0, 50]); % Display up to lag 50 for clarity
grid on;
hold off;

% Check how many lags exceed the confidence interval
num_violations = sum(abs(acf(2:end)) > conf); % Exclude lag 0
expected_violations = 0.05 * (length(acf) - 1);

fprintf('Number of lags outside 95%% confidence interval: %d\n', num_violations);
fprintf('Expected approximate number of violations: %.1f\n', expected_violations);