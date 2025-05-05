

%%
t = (0:0.01:5*pi)'
f = 100 + sin(t) + 0.5 * randn(size(t));

TH = [];
k1 = 1;
%%
for k=1:5
    TH(:,k1) = t.^(k-1);
    func(k1,1) = string(strcat('t^', num2str(k-1)));

    k1 = k1+1;
end

for j=0.25:0.25:2
    TH(:,k1) = sin(j*t);
    func(k1,1) = string(strcat('sin(', num2str(j), 't)'));

    k1 = k1+1;
end

for j=0.25:0.25:2
    TH(:,k1) = cos(j*t);
    func(k1,1) = string(strcat('cos(', num2str(j), 't)'));

    k1 = k1+1;
end


%% Sparse regression
[coeff, stats] = lasso(TH, f, 'Lambda', 0.000001, RelTol=1e-6, MaxIter=1e6);
disp([func coeff])

fpred = TH*coeff;

scatter(t,f,'filled');
alpha(0.5)
hold on
plot(t, fpred)

LSQ = pinv(TH) * f;
fpred_lsq = TH*LSQ;
plot(t, fpred_lsq, '--k')
hold off