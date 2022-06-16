% Bayesian update for the posterior
function [a_vec, posteriors, posterior_doms] = BayesianUpdate(l_vec, B_gamma, C, priors, prior_doms, N_MCMC, N_burn, start)
% Updates the estimate for the constrained parameter vector a_vec using a
% Bayesian inference scheme. The posterior distribution is also approximated
% from MCMC samples. The Likelihood is computed from the most recent measurement
% outcome of the joint parameter estimator.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% l_vec     : The measurement vector containing the number of photons in
%             each eigenstate of the joint parameter estimator B_gamma.
% B_gamma   : the joint parameter estimator used to collect the measurement
% C         : a stack of the transformed wavelet operators C =  W * A
% priors    : A matrix of dimensions [n_params-1,100]. The i'th row contains
%             points on the i'th parameter's prior distribution.
% prior_doms: A matrix of dimensions [numel(a_vec),100]. Each row
%             contains the domain points associated with points on the prior
%             distribution.
% N_MCMC    : number of Metropolis-Hastings MCMC samples used to compute
%             approximate the posterior.
% N_burn    : number of Metropolis-Hasting MCMC samples to discard in order
%             to let the Markov Chain reach steady state.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% a_vec             : the updated estimate on the constrained parameter vector
% posteriors        : A matrix of dimensions [n_params-1,100]. The i'th row contains
%                     points on the i'th parameter's posterior distribution. 
% posterior_domains : A matrix of dimensions [n_params-1,100]. Each row
%                     contains the domain points associated with points on
%                     the posterior distribution. 

n_as = size(priors,1);
use_MCMC = 0;

%% MCMC Sampling Methods
if use_MCMC


    % Slice Sampling
    %pdf = @(x)max(likelihood(l_vec, B_gamma, x', C) * prior(x', priors, prior_doms), realmin); % pdf cannot be 0 for MCMC methods
    pdf = @(x)likelihood(l_vec, B_gamma, x', C) * prior(x', priors, prior_doms);
    samples = slicesample(start,N_MCMC,'pdf',pdf,'burnin',N_burn);


    %{
    % Metropolis Hastings Sampling
    % proportional probability density to posterior
    n_as = numel(start);
    pdf = @(x)max(likelihood(l_vec, B_gamma, x', C) * prior(x', priors,
    prior_doms), realmin); % pdf cannot be 0 for MCMC methods
    proppdf = @(x,y) mvnpdf(y,x);
    proprnd = @(x) mvnrnd(x, eye(n_as));
    %samples = mhsample(start, N_MCMC, 'pdf', pdf, 'proppdf', proppdf, 'proprnd', proprnd, 'burnin', N_burn,'nchain',1);
    samples = mhsample(start, N_MCMC, 'pdf', pdf, 'proprnd', proprnd, 'burnin', N_burn,'symmetric',1,'nchain',1);
    %}

    % Use kernel density estimation to generate a smooth approximation of the
    % posterior
    posteriors = zeros([n_as,100]);
    posterior_doms = zeros([n_as,100]);

    for i = 1:n_as
        [posteriors(i,:), posterior_doms(i,:)] = ksdensity(samples(:,i));
    end
else
    % Interior Sampling (only works for low-dimensional cases)
    pdf = @(x)max(likelihood(l_vec, B_gamma, x, C) * prior(x, priors, prior_doms), realmin);
    [posteriors,posterior_doms] = interior_sampling(pdf,n_as);
end


% normalize posteriors (technically ksdensity discretely approximates a pdf)
posteriors = posteriors./sum(posteriors,2);

% MMSE estimator is given by the mean of the posterior
a_vec = sum(posteriors.*posterior_doms,2);

end

function p_l = likelihood(l_vec, B_gamma, a_vec, C)
% The likelihood (probability) of observing l_vec under the measurement
% operator B_gamma (the joint estimator). The likelihood function is a
% multinomial over the probabilty distribution of the outcomes.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% l_vec     : measurement vector containing the number of photons detected in each
%             of the eigenstates of the joint estimator B_gamma l_vec = [n_1,n_2,...,n_N]
% B_gamma   : The measurement matrix for the joint estimator.
% C         : a stack of the transformed wavelet operators C =  W * A
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_l       : the probability of the outcome l_vec

[~,max_dim_idx] = max(size(a_vec));    
if max_dim_idx == 2
    a_vec = a_vec';
end

% get the density matrix
aa_vec = [a_vec;1];
rho = rho_a_HG(aa_vec, C);

% get the probabilities of each measurement outcome
[V,~] = eig(B_gamma);
p_outcomes = diag(V'*rho*V);

% take absolute value for numerical stability
p_outcomes = abs(p_outcomes);

% include the outcome probability for modes greater the N_HG_modes
p_outcomes = [p_outcomes; max(0,1-sum(p_outcomes))];

% normalize probability
p_outcomes = p_outcomes/sum(p_outcomes);
    
% likelihood is a multinomial
p_l = mnpdf(l_vec, p_outcomes);

% handle numerically unstable cases
if sum(isnan(p_l))>0
    p_l = 0;
end
end

function p_a = prior(a_vec, priors, prior_doms)
% Calculates the probability of observing a_vec given
% a generalized probability density approximation for each parameter.
% This prior assumes the parameters are independent.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec         : An instance of the parameter vector with prior given by the
%                 other inputs (priors, prior_doms)
% priors        : A matrix of dimensions [numel(a_vec),100]. The i'th row contains
%                 points on the i'th parameter's prior distribution.
% prior_doms    : A matrix of dimensions [numel(a_vec),100]. Each row
%                 contains the domain points associated with points on the prior
%                 distribution.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a           : probability P(a_vec)


%{
% Discrete Distribution Case
% get domain points nearest to sample point (in a Manahatten distance
% sense)
[~,p_ind] = min(abs(prior_doms-a_vec),[],2);
% determine the prior probabilites of the sample for each of the parameters
p_a_vec = priors(sub2ind(size(priors),1:size(priors,1),p_ind'));
%}

% Continuous Distribution Case
p_a_vec = zeros([numel(a_vec),1]);
for i = 1:numel(a_vec)
   p_a_vec(i) = interp1(prior_doms(i,:),priors(i,:),a_vec(i));
end

% handle numerically unstable cases
if sum(isnan(p_a_vec))>0
    p_a = 0;
else
    p_a = prod(p_a_vec);
end
end

function [posteriors,posterior_doms] = interior_sampling(pdf,n_as)
% Brute force samples the pdf p(l|a)*p(a) over a domain of
% interest and marginalizes to get posteriors.
    %N_samples = 1e6;                    % use a million samples uniformly distributed over the volume of interest
    N_samples = 36^3;
    N = floor(N_samples^(1/n_as)); 
    a_rng = linspace(-.5,.5,N);
    [A{1:n_as}] = ndgrid(a_rng);
    
    p_a_vec = zeros(size(A{1}));
    
    a_vec = zeros([n_as,1]);
    for i = 1:numel(p_a_vec)
        for j = 1:n_as
            A_j = A{j}; 
            a_vec(j) = A_j(i);
        end
        p_a_vec(i) = pdf(a_vec);
    end
    
    % normalize
    p_a_vec = p_a_vec / sum(p_a_vec(:));
    
    % marginalize
    posteriors = zeros(n_as,N);
    for i = 1:n_as
        sumover_idx = 1:n_as;
        sumover_idx(i) = [];
        posteriors(i,:) = reshape(sum(p_a_vec,sumover_idx),[1,N]);
    end
    
    posterior_doms = repmat(a_rng,[n_as,1]);
end



function F = check_prop_posterior(l_vec, B_gamma, C, priors, prior_doms, posteriors, posterior_doms)
    
    N = 51;
    a1 = linspace(-.5,.5,N); 
    a2 = linspace(-.5,.5,N);
    a3 = linspace(-.5,.5,N);
    
    [A1,A2,A3] = meshgrid(a1,a2,a3);

    F = zeros(size(A1));
    for i = 1:N^3
        a_vec  = [A1(i),A2(i),A3(i)]';
        F(i) = likelihood(l_vec, B_gamma, a_vec, C)*prior(a_vec, priors, prior_doms);
    end
    
    
    figure
    title('Uniform volumeteric sampling of posterior')
    FF = (F-min(F(:)))/max(F(:));
    scatter3(A1(:),A2(:),A3(:),FF(:)*36+1e-5,FF(:))
    xlabel('a1')
    ylabel('a2')
    zlabel('a3')
    
    % marginalize
    FF_a1 = squeeze(sum(FF,[2,3]));
    FF_a2 = squeeze(sum(FF,[1,3]));
    FF_a3 = squeeze(sum(FF,[1,2]));
    
    % compare prop_posterior to posterior found by MCMC
    
    % MCMC posteriors
    figure
    title('MCMC Posteriors')
    subplot(1,3,1)
    plot(posterior_doms(1,:),posteriors(1,:))
    xlabel('a_1')
    subplot(1,3,2)
    plot(posterior_doms(2,:),posteriors(2,:))
    xlabel('a_2')
    subplot(1,3,3)
    plot(posterior_doms(3,:),posteriors(3,:))
    xlabel('a_3')
    
    
    % unnormalized uniform sampling posteriors
    figure
    subplot(1,3,1)
    plot(a1,FF_a1)
    xlabel('a_1')
    subplot(1,3,2)
    plot(a2,FF_a2)
    xlabel('a_2')
    subplot(1,3,3)
    plot(a3,FF_a3)
    xlabel('a_3')
    
    
end



function p_a = GBM_prior(a_vec,mu,z,q)
% Calculates the probability of observing a_vec given the GBM prior
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec:  the [N-1 x 1] parameter vector 
% mu:     the [N-1 x 2] mean vector containing the means of each normal distribution 
% in the GBM prior. [mu_1,0;mu_2,0; ...]. In general, this generates a bimodal distribution for each parameter as the
% prior updates in each Bayesian iteration. 
% z:      the [N-1 x 2] variance vector contianing the two variances of each
% normal distribution in the GBM prior. [zmax_1,zmin_1;z_max_2,zmin_2; ...]In general, the variances reduce
% every iteration of the Bayesian update to the prior.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a: the prior probability for the parameter values.
    [~,max_dim_idx] = max(size(a_vec));    
    if max_dim_idx == 2
        a_vec = a_vec';
    end
    
    p_a_vec = zeros(size(a_vec));   % probability for each independent GBM random variable
    
    % binary indices to the dirac delta cases
    dd1 = z(:,1)==0;
    dd2 = z(:,2)==0;
    
    % get the priors for each independendent GBM random variable
    p_a_vec(dd1) = q*(a_vec(dd1) == mu(dd1,1));
    p_a_vec(~dd1) = q*normpdf(a_vec(~dd1), mu(~dd1,1),z(~dd1,1));
    p_a_vec(dd2) = p_a_vec(dd2) + (1-q)*(a_vec(dd2)== mu(dd2,2));
    p_a_vec(~dd2) = p_a_vec(~dd2) + (1-q)*normpdf(a_vec(~dd2),mu(~dd2,2),z(~dd2,2));
    
    % joint probability of all the independent random variables
    p_a = prod(p_a_vec);
    
    % handle numerically unstable cases
    if sum(isnan(p_a))>0
        p_a = 0;
    end
    
end