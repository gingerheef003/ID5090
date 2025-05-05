clc
clear

U = unifrnd(-pi/8, pi/8, 1000, 1);
acf_val = autocorr(U, 'NumLags',20)