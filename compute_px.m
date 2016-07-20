function px = compute_px(x, p_k, p_xi_givenk, ak)
% px = COMPUTE_PX(x, p_k, p_xi_givenk, ak);
%
% Compute probability of a binary pattern, or set of binary patterns, given
% the Population Tracking model parameters.
%
% Inputs:   x, 1-by-N binary neural activity pattern. Can also be in form
%               of T-by-N matrix of many patterns.
%           p_k, Population synchrony distribution, p(k)
%           p_xi_givenk, the probability that each individual neuron is
%               active given the population rate, p(xi|k)
%           ak (OPTIONAL), (N+1)-by-1 vector of precomputed normalizing constants
% Output:   px, pattern probability, or list of probabilities if input was
%               a set of patterns.

[T,N] = size(x); % Number of patterns, neurons

if nargin == 3; % If no ak supplied, compute it
    
    brute_thresh = 1e5;
    nsamples = 1e5;
    
    if T > 1 % If more than one pattern is presented, compute ak for all k
        ak = compute_ak(p_xi_givenk,brute_thresh,nsamples);
        
    else % if only one pattern presented, compute ak only for that k
        ak = zeros(N+1,1); % Initialize vector
        kactive = length(find(x)); % Number of ON neurons
        pvec = p_xi_givenk(:,kactive+1); % Select p_xi_givenk vector for correct k
        nwords_wkactive = nchoosek(N,kactive); % Number of words with k active
        if nwords_wkactive<brute_thresh
            patternmat = nchoosek([1:N],kactive); % all possible patterns
            cumsumpword = 0;
            for i = 1:nwords_wkactive
                onindsk = patternmat(i,:); % ON neurons
                offindsk = setdiff([1:N],onindsk); % OFF neurons
                cumsumpword = cumsumpword + prod(pvec(onindsk))*prod(1-pvec(offindsk));
            end
            ak(kactive+1) = cumsumpword;
        else
            if std(pvec)==0 % If homogeneous (due to lack of data at given k)
                ak(kactive+1) = nwords_wkactive*(pvec(1)^kactive) * (1-pvec(1))^(N-kactive);
            else % If heterogeneous
                cumsumpword = 0;
                for i = 1:nsamples
                    onindsk = randperm(N,kactive); % Choose k random ON neurons
                    offindsk = setdiff([1:N],onindsk); % OFF neurons
                    cumsumpword = cumsumpword + prod(pvec(onindsk))*prod(1-pvec(offindsk));
                end
                ak(kactive+1) = cumsumpword*(nwords_wkactive/nsample_kactive);
            end
        end
    end
end


% Compute pattern probabilities
px = zeros(T,1);
for i = 1:T
    oninds = find(x(i,:)); % ON neurons
    offinds = setdiff([1:N],oninds); % OFF neurons
    kactive = length(oninds); % Number of ON neurons
    
    % Pattern probability
    px(i) = p_k(kactive+1)*prod(p_xi_givenk(oninds,kactive+1))*prod(1-p_xi_givenk(offinds,kactive+1))./ak(kactive+1);
end