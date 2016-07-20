function [S] = samplePopTrack(p_k,p_xi_givenk,T)
% S = SAMPLEPOPTRACK(p_k, p_xi_givenk, T)
% Generate samples from Population Tracking model given the parameters.
%
% Input:    p_k: Population synchrony distribution, p(k)
%           p_xi_givenk, the probability that each individual neuron is
%               active given the population rate, p(xi|k)
%           T: Number of time samples
% Output:   S: T-by-N matrix of binary samples


N = length(p_k)-1; % Number of neurons
cumppop = cumsum(p_k); % Get cumulative of p(k)

S = zeros(T,N); % Initialize output sample matrix

% Draw samples
for t = 1:T
    k = find(cumppop>rand,1); % Number of neurons active (plus one)
    
    ktest = nan;
    while ktest ~= k-1;
        testsamp = floor(rand(N,1)+p_xi_givenk(:,k));
        ktest = sum(testsamp);
    end
    S(t,:) = testsamp';
end