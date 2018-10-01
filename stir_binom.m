function [c] = stir_binom(n,k);
% STIR_BINOM(N,K) calculates the binomial coeffcient (n-choose-k) using
% Stirling's approximation n! = n*log(n)- n. Actually uses extra terms in
% the approximation so its n! = (n + 1/2)*log(n) - n + (1/2)*log(2*pi)

logc = (n+ 1/2)*log(n) - (k + 1/2)*log(k) - (n - k + 1/2)*log(n-k) - (1/2)*log(2*pi);
c = round(exp(logc));