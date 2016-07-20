function ak = compute_ak(p_xi_givenk,brute_thresh,nsamples)
% ak = COMPUTE_AK(p_xi_givenk, brute_thresh, nsamples)
% Compute the vector of a_k values needed to normalize the Population
% Tracking model probability distribution. It is equal to the probability
% of having k active neurons as a function of k assuming neurons are
% conditionally indepdendent.
%
% The algorithm attempts the brute force enumeration for the exact value,
% then if too large uses an approximate method (importance sampling).
% See paper for details.
% 
% Input:    p_xi_givenk, the matrix of the probabilities that each 
%               neuron is active given the population rate, p(xi|k)
%           brute_thresh, the threshold number of patterns above which
%               brute force enumeration is abandoned (suggest ~1e4 to 1e5)
%           nsamples, the number of samples per k for importance sampling
%               (suggest at least 1e4).
% Output:   ak, (N+1)-by-1 vector of ak values

% Turn off MATLAB's nchoosek warning
warning('off','MATLAB:nchoosek:LargeCoefficient')

% Number of units, N
[nrows,ncols] = size(p_xi_givenk);
N = min(nrows,ncols);

% Find ak
ak = zeros(N+1,1); % Initialize
ak(1) = 1; % Silent state
ak(end)=1; % All ON state
for k = 2:N

    kactive = k-1;
    
    pvec = p_xi_givenk(:,k); % Select p_xi_givenk vector for correct k
    nwords_wkactive = nchoosek(N,kactive); % Number of words with k active
    
    if nwords_wkactive<brute_thresh
        patternmat = nchoosek([1:N],kactive); % all possible patterns
        cumsumpword = 0;
        for i = 1:nwords_wkactive
            onindsk = patternmat(i,:); % ON neurons
            offindsk = setdiff([1:N],onindsk); % OFF neurons
            cumsumpword = cumsumpword + prod(pvec(onindsk))*prod(1-pvec(offindsk));
        end
        ak(k) = cumsumpword;
    else
        if std(pvec)==0 % If homogeneous (due to lack of data at given k)
            ak(k) = nwords_wkactive*(pvec(1)^kactive) * (1-pvec(1))^(N-kactive);
        else % If heterogeneous
            cumsumpword = 0;
            for i = 1:nsamples
                onindsk = randperm(N,kactive); % Choose k random ON neurons
                offindsk = setdiff([1:N],onindsk); % OFF neurons
                cumsumpword = cumsumpword + prod(pvec(onindsk))*prod(1-pvec(offindsk));
            end
            ak(k) = cumsumpword*(nwords_wkactive/nsamples);
        end
    end
    
end

% Turn MATLAB nchoosek warning back on
warning('on','MATLAB:nchoosek:LargeCoefficient')