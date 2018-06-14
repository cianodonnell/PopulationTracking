function [JSdiv,jsdiv_k] = get_jsdiv(ppop1,pactive1,ppop2,pactive2);
% Get JS divergence from distribution 2 to distribution 1 given parameters
% population activity distribution ppop1&2, and conditional
% probability each neuron fires given population activity, pactive1&2.

nsamples = 1e5; % Number of samples to estimate correlation parameter
nsamples_kactive = 1e5;
nunits = length(ppop1)-1;

% Get middle distribution stats
ppopm = 0.5*(ppop1 + ppop2);

% Function to integrate
intfun = @(x,mu_x,sigma_x,mu_y,sigma_y,rho) ...
    (1/(sigma_x*sqrt(2*pi))).*exp(-((x-mu_x).^2)./(2*sigma_x.^2))...
   .* 2.^(x)...
    .* (mu_y + (sigma_y/sigma_x)*rho.*(x - mu_x) );


% Loop over all possible numbers of active units
jsdiv_k = zeros(nunits+1,1);
jsdiv_k(1) = 0.5*( ppop1(1)*log2(ppop1(1)/ppopm(1)) + ppop2(1)*log2(ppop2(1)/ppopm(1)) );
JSdiv = 0; % initialise
JSdiv = JSdiv + jsdiv_k(1); % Silent state
kvec = find(ppop1~=ppop2);
kvec = kvec(2:end);
for k = kvec
    kactive = k-1;
    
    pvec1 = pactive1(:,k);
    pvec2 = pactive2(:,k);
    
    % Get fraction of words that k neurons active given conditional
    % independent firing probabilities
    pmat1 = repmat(pvec1,1,nsamples_kactive);
    S = floor(rand(length(pvec1),nsamples_kactive)+pmat1);
    sumS = sum(S,1);
    p_kactive1 = sum(sumS==kactive)./nsamples_kactive;
    pmat2 = repmat(pvec2,1,nsamples_kactive);
    S = floor(rand(length(pvec2),nsamples_kactive)+pmat2);
    sumS = sum(S,1);
    p_kactive2 = sum(sumS==kactive)./nsamples_kactive;

    pword1 = zeros(nsamples,1);
    pword2 = zeros(nsamples,1);
    for i = 1:nsamples;
        oninds = randperm(nunits,kactive);
        offinds=setdiff([1:nunits],oninds);
        pword1(i) = ppop1(k)*prod(pvec1(oninds))*prod(1-pvec1(offinds))/p_kactive1;
        pword2(i) = ppop2(k)*prod(pvec2(oninds))*prod(1-pvec2(offinds))/p_kactive2;
    end
    pwordm = 0.5*(pword1+pword2);
    
    %%%%%%%%
    % Get KL divergence between P1 and M
    %%%%%%%%
    x = log2(pword1);
    y = log2(pword1./pwordm);
    
    % Estimate parameters for integral
    mu_x = mean(x);
    sigma_x = std(x);
    mu_y = mean(y);
    sigma_y = std(y);
    rho = corr(x,y);
    
    if sigma_y>0; % if there is a difference in word probabilities
    
        % Integrate over pdf
        q = integral(@(x)intfun(x,mu_x,sigma_x,mu_y,sigma_y,rho),mu_x-10*sigma_x,mu_x+10*sigma_x);

        % Compute KLdiv contribution
        if abs(nunits-kactive)>10;
            binocoef = stir_binom(nunits,kactive);
        else
            binocoef = nchoosek(nunits,kactive);
        end
        kldiv_1m_k = binocoef * q ;
    end
    
    %%%%%%%%
    % Get KL divergence between P2 and M
    %%%%%%%%
    x = log2(pword2);
    y = log2(pword2./pwordm);
    
    % Estimate parameters for integral
    mu_x = mean(x);
    sigma_x = std(x);
    mu_y = mean(y);
    sigma_y = std(y);
    rho = corr(x,y);
    
    if sigma_y>0; % if there is a difference in word probabilities
    
        % Integrate over pdf
        q = integral(@(x)intfun(x,mu_x,sigma_x,mu_y,sigma_y,rho),mu_x-10*sigma_x,mu_x+10*sigma_x);

        % Compute KLdiv contribution
        if abs(nunits-kactive)>10;
            binocoef = stir_binom(nunits,kactive);
        else
            binocoef = nchoosek(nunits,kactive);
        end
        kldiv_2m_k = binocoef * q ;
    end

    jsdiv_k(k) = 0.5*(kldiv_1m_k + kldiv_2m_k);
    JSdiv = JSdiv + jsdiv_k(k);
    fprintf('%1.0f/131 active, JS_k=%1.4f\n',kactive,jsdiv_k(k) )
end
jsdiv_k(end) = 0.5*( ppop1(end)*log2(ppop1(end)/ppopm(end)) + ppop2(end)*log2(ppop2(end)/ppopm(end)) );
JSdiv = JSdiv + jsdiv_k(end); % All active state

% PLOT
figure()
plot([0:nunits],cumsum(jsdiv_k),'-x')
xlabel('Number of units')
ylabel('Cumulative JS divergence')