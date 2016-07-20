function [p_k,p_xi_givenk,p_xi] = fitPopTrack(X);
% [p_k, p_xi_givenk, p_xi] = FITPOPTRACK(X);
% Fits population tracking model to binary spike data.
%
% Input:    X, binary data matrix (T-by_N)
% Outputs:  p_k, the population synchrony distribution, p(k)
%           p_xi_givenk, the probability that each individual neuron is
%               active given the population rate, p(xi|k)
%           p_xi, the mean firing rate of each neuron, p(xi), useful for
%               building an independent neuron model
%
% The data X should be in the form of a T-by-N matrix where each row
% is a different time sample (T total) and each column is a different
% neuron (N total).
% The model fitting is two-step, one step for the population synchrony
% distribution p(k) and one for the conditional probability of each neuron
% being active given the population synchrony level, p(xi|k).
% Both steps include a Bayesian regularization that requires specifying at
% least two hyperparameters: 1) the concentration parameter for the
% dirichlet prior for p(k), alpha. 2) the scaling factor on the variance of
% the beta prior for each p(xi|k), prior_var_scale.

[T,N] = size(X); % Number of timepoints and neurons, respectively

%%%%%%%%
% HYPER PARAMETERS
%%%%%%%

alpha = 1e-2; % Hyperparameter on diriclet prior for pop synch distribution

prior_var_scale = 0.5; % scale variance of beta prior on p_xi_givenk estimates

%%%%%%%%
% INITIAL PROCESSING
%%%%%%%%

% Vector for population synchrony count
popvec = [0:N];

%%%%%%%%%
% MEAN FIRING RATE OF EACH NEURON
%%%%%%%%%
p_xi = mean(X,1);

%%%%%%%%
% POPULATION RATE
%%%%%%%%
n_on = sum(X,2); % count number of units at each bin
nhist = hist(n_on,popvec); % Histogram synchrony distribution
nhist = nhist + alpha; % Add regularising pseudocount from dirichlet prior
p_k = nhist./sum(nhist); % Normalise to get probability distribution

%%%%%%%%
% GET PRATE GIVEN POPRATE
%%%%%%%%
p_xi_givenk = zeros(N,N+1); % Initialize
p_xi_givenk(:,1) = 0; % Silent state, all OFF
p_xi_givenk(:,end)=1; % All ON

for i = 1:N-1; % loop over popuation levels
    
    tn = find(n_on==i); % Timesteps where j neurons active
    npop = length(tn); % Number of such timesteps
    nactive_vec = sum(X(tn,:),1)'; % number of times each neuron active
    
    mu = i/N; % Mean of prior
    sigma2 = prior_var_scale^2*mu*(1-mu); % Variance of prior
    beta1 = mu*((mu - mu^2 - sigma2)/sigma2); % parameter for beta prior
    beta2 = beta1*( 1/mu - 1); % parameter for beta prior
    
    % Posterior mean estimate
    p_xi_givenk(:,i+1) = (nactive_vec+beta1)./(npop + beta1 + beta2);
    
end