function [Hiid,Hind,Hpop,Hfull] = get_entropy_allmodels_largeN(p_k,p_xi_givenk,p_xi)
% Get entropy H of pattern probability distribution under 4 different
% models:
% 1) Independent neurons, identical firing rates (Hiid)
% 2) Independent neurons, firing rates matched to data (Hind)
% 3) Population rate model, identical neurons. (Hpop)
% 4) Full model, poprate and conditional firing probabilities (Hfull)
% Use fitting method for full entropy.
%
% Also calculate full pattern probability distribution.


tic;
% Turn off certain warnings
warning('off','MATLAB:nchoosek:LargeCoefficient')
warning('off','stats:kmeans:FailedToConverge')
warning('off','stats:gmdistribution:FailedToConverge')

nwords_sample = 1e5; % Number of pword samples to draw, per k.
%trykclust = 8; % Number of gaussians to try for pword distribution fit

%trykclust_intercept = 4; % Number of gaussians for k = 1
%trykclust_onegaussian = round(N/2); % k=? for one gaussian

trykclust_intercept = 1; % Number of gaussians for k = 1
trykclust_onegaussian = 1; % k=? for one gaussian

% Extract number of units
nunits = length(p_k)-1;

%******
% HIID
%******
pmean = mean(p_xi);
Hiid = - nunits*(pmean*log2(pmean) + (1-pmean)*log2(1-pmean));

%******
% HIND
%******
Hind = - nansum( p_xi.*log2(p_xi) + (1-p_xi).*log2(1-p_xi) );

%******
% HPOP
%******
Hpop = 0;
for k = 1:nunits+1;
    Hpop = Hpop - p_k(k).*log2(p_k(k)) + p_k(k)*log2(nchoosek(nunits,k-1));
end

%********
% HFULL
%********

% Get word probdis
Hfull = - p_k(1)*log2(p_k(1)); % add entropy of silent state
track_Hk = zeros(nunits,1); % for debugging
track_Hjgivenk = zeros(nunits+1,1);
for k = 2:nunits;
    
    % Population rate entropy
    Hk = - p_k(k)*log2(p_k(k));
    
    pvec = p_xi_givenk(:,k);
    kactive = k-1;
    nwords_wkactive = nchoosek(nunits,kactive);
    if nwords_wkactive == Inf % use stirling approximation if n and k too big
        [nwords_wkactive,log_nwords_wkactive]= stir_binom(nunits,kactive);
    end
    
    % Calculate number of clusters to try
    trykclust = max(round(trykclust_intercept - trykclust_intercept*kactive/trykclust_onegaussian) , 2);
    
%     if mod(kactive,2)==0
%         fprintf('%1.0f/%1.0f units\n',kactive,nunits )
%         
%         figure(1)
%         clf
%         plot([0:nunits],track_Hpop,'b-x')
%         hold on
%         plot([0:nunits],track_Hk,'r-x')
%         title(num2str(kactive))
%         legend('Pop','Full')
%         hold off
%     end

    if std(pvec)>0.001; % Check for heterogeneous population

        if nwords_wkactive > nwords_sample; % Do statistical model of popualtion

            % Generate list of word probabilities
            log2pword = zeros(nwords_sample,1);
            for i = 1:nwords_sample
                inds = randperm(nunits,kactive);
                offinds = setdiff([1:nunits],inds);
                log2pword(i) = sum(log2((pvec(inds)))) + sum(log2(1-pvec(offinds)));
            end
            
            % Fit mixture-of-gaussians
            options = statset('Display','off');
            AIC = zeros(1,trykclust);
            GMModels = cell(1,trykclust);
            for kclust = 1:trykclust;
                %fprintf('%1.0f/%1.0f clusters\n',kclust,trykclust )
                
                % k-means for initial guess of gaussian means and sd's
                [idx,kcenters,sumd] = kmeans(log2pword,kclust);
                nkclust = zeros(1,kclust);
                for i = 1:kclust
                    nkclust(i) = length(find(idx==i));
                end
                ksd = ones(1,1,kclust);
                ksd(1,1,:) = sumd'./nkclust;
                meanvar = mean(squeeze(ksd))^2;
                
                % Create struct with initial parameter guesses
                gmS = struct('mu',kcenters,'Sigma',ksd,'PComponents',ones(1,kclust)/kclust);
                
                % Fit with EM
                GMModels{kclust} = fitgmdist(log2pword,kclust,'Options',options,'Start',gmS,'SharedCov',false,'CovType','diagonal','Regularize',0.1*meanvar);
                AIC(kclust) = GMModels{kclust}.AIC;
            end
            [~,bestkclust] = min(AIC); % Find best fit model
            BestModel = GMModels{bestkclust};
            
            % Renormalise so that distribution sums to one (pkactive~=1)
            pwordmin = min(BestModel.mu-5*sqrt(squeeze(BestModel.Sigma)));
            pwordmax = max(BestModel.mu+5*sqrt(squeeze(BestModel.Sigma)));
            pwordvec = linspace(pwordmin,pwordmax,1e4);
            gmpdf = pdf(BestModel,pwordvec'); % Get gaussian mixture pdf
            gmpdf = gmpdf./sum(gmpdf); % Normalise discrete distribution
            if nwords_wkactive ~= Inf % if number of words not too big to represent directly
                pwordsum = sum((2.^pwordvec)*gmpdf*nwords_wkactive);
                pwordvec = pwordvec-log2(pwordsum); % Shift to normalise
            else 
                logpwordsum = log_nwords_wkactive + log(sum((2.^pwordvec)*gmpdf));
                pwordvec = pwordvec-logpwordsum/log(2);
            end
            pwordvec2 = 2.^pwordvec; % Convert back to real p
            
            % Get entropy
            if nwords_wkactive ~= Inf % if number of words not too big to represent directly
                Hjgivenk = -p_k(k)*nwords_wkactive*sum(gmpdf'.*pwordvec2.*log2(pwordvec2));
            else
                Hjgivenk = -exp( log_nwords_wkactive + log( p_k(k)*sum(gmpdf'.*pwordvec2.*log2(pwordvec2)) ) );
            end

        else % Enumerate all word probabilities
            pword = zeros(nwords_sample,1);
            onmat = nchoosek(1:nunits,kactive);
            for i = 1:nwords_wkactive
                offinds = setdiff([1:nunits],onmat(i,:));
                pword(i) = prod(pvec(onmat(i,:)))*prod(1-pvec(offinds));
            end
            pword = pword./sum(pword); % Normalise
            Hjgivenk = -p_k(k)*nansum(pword.*log2(pword));
        end
        
    else % If homogeneous population
        Hjgivenk = p_k(k)*log2(nwords_wkactive);
    end

    % Add entropies to total entropy count
    if isnan(Hjgivenk)==0
        Hfull = Hfull + Hk + Hjgivenk;
        track_Hk(k) = Hk + Hjgivenk;
        track_Hjgivenk(k) = Hjgivenk/p_k(k);
    end
    
end

% Turn warnings back on
warning('on','MATLAB:nchoosek:LargeCoefficient')
warning('on','MATLAB:stats:kmeans:FailedToConverge')
warning('on','MATLAB:stats:gmdistribution:FailedToConverge')

%fprintf('Total runtime = %1.0f mins\n',floor(toc/60) )