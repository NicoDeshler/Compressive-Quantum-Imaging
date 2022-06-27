% Bayesian update for the posterior
function [a_vec, a_mu_post, a_cov_post, posterior, posterior_dom] = BayesianUpdate_mvksdensity(l_vec, B_gamma, C,...
                                                                prior, prior_dom,...
                                                                N_samples, method,...
                                                                W,wv_idx,WaveletName,...
                                                                varargin)
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
% prior     : a vector of probabilities for the posterior
% prior_dom : a matrix of posterior domain points at which the posterior is
%             is evaluated. Each row corresponds to a new unique variable.
% N_samples : number of samples used to approximate the posterior.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% a_vec             : the updated estimate on the constrained parameter vector
% a_mu_post         :
% a_cov_post        :
% posterior        : A matrix of dimensions [n_params-1,100]. The i'th row contains
%                     points on the i'th parameter's posterior distribution. 
% posterior_dom : A matrix of dimensions [n_params-1,100]. Each row
%                     contains the domain points associated with points on
%                     the posterior distribution. 
% ----------------------------------------------------------------
% NAME/VALUE:
% ----------------------------------------------------------------


p = inputParser;
n_as = size(priors,1);

prior_fn = @(x) mvksdensity_prior(x,prior,prior_dom);
pdf = @(x) likelihood(l_vec, B_gamma, x, C) * constrain_prior(x, prior_fn, W, wv_idx, WaveletName);
MCMC_pdf = @(x) max(pdf(x),realmin);

switch method
    case 'interior'
        % Interior Sampling (works well for low-dimensional cases)
        addRequired(p,'a_min')
        addRequired(p,'a_max')
        parse(p, varargin{:})
        
        a_min = p.Results.a_min;
        a_max = p.Results.a_max;
        
        [posterior, posterior_dom] = interior_sampling(pdf,N_samples,n_as,a_min,a_max);
        
    case 'importance'
        % Importance Sampling      
        addRequired(p,'ref_mu')
        addReqired(p,'ref_sigma')
        parse(p, varargin{:})
        
        ref_mu = p.Results.ref_mu;
        ref_sigma = p.Results.ref_sigma;        
        
        [posterior, posterior_dom] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sigma);
    
    case 'slice'
        % Slice MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})
        
        N_burn = p.Results.N_burn;
        start = p.Results.start;
        
        [posterior, posterior_dom] = slice_sampling(MCMC_pdf,N_samples,n_as,N_burn,start);
    
    case 'MH'
        % Metropolis-Hastings MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})

        N_burn = p.Results.N_burn;
        start = p.Results.start;
        
        [posterior, posterior_dom] = MH_sampling(MCMC_pdf,N_samples,n_as,N_burn,start);
        
    otherwise
        error('Unsupported sampling method.')        
end


% compute the variances and the means from the posteriors
a_mu_post = sum(posterior.*posterior_dom,1)';


a_var_post = sum(posteriors.*((posterior_doms-a_mu_post).^2),2);
a_cov_post = diag(a_var_post);

% MMSE estimator is given by the mean of the posterior
a_vec = a_mu;

end

%%%%%%%%%%%%%%%%%%%%%% LIKELIHOOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p_a = constrain_prior(a_vec,prior_fn, W,wv_idx,WaveletName)
    % imposes non-negativity constraint
    p_a = prior_fn(a_vec) * non_neg(a_vec,W,wv_idx,WaveletName);
end


function p_a = mvksdensity_prior(a_vec, prior, prior_dom)
% Calculates the probability of observing a_vec given
% a generalized approximation for the joint prior probability over the
% parameters.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec         : An instance of the parameter vector with prior given by the
%                 other inputs (priors, prior_doms)
% prior        : A vector of probability densities
% prior_dom    : A matrix of points in the domain of a_vec corresponding to
%               the probability densities in 'prior'. Each column
%               corresponds to a unique variable.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a           : probability P(a_vec)

p_a = interpn(prior_dom,prior,a_vec,'nearest');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAMPLING METHODS %%%%%%%%%%%%%%%%%%%%%%%

function [posterior,posterior_dom]= MH_sampling(pdf,N_samples,n_as,N_burn,start)
    % Metropolis Hastings Sampling
    % proposal probability density 
    proppdf = @(x) mvnpdf(x,a_mu_prior, a_cov_prior);
    proprnd = @(x) mvnrnd(a_mu_prior, a_cov_prior);
    % MH samples
    samples = mhsample(start, N_samples, 'pdf', pdf,'proppdf',proppdf,'proprnd', proprnd, 'burnin', N_burn,'nchain',1);

    % get first and second moments of the samples
    a_mu_post = mean(samples,2)';
    a_cov_post = cov(samples);

    % Use kernel density estimation to generate a smooth approximation of the
    % posterior

    % range to evaluate the posterior over
    linear_dom = linspace(a_min,a_max,N);
    [posterior_dom{1:n_as}] = ndgrid(linear_dom);
    posterior_dom = arrayfun(@(x) {reshape(cell2mat(x),[numel(cell2mat(x))])},posterior_dom);
    
    % Silverman's rule of thumb for the kernel bandwidth
    silvermans_bw = sqrt(diag(a_cov_post)) .* ((4/(n_as+2)/N_samples)^(1/(n_as + 4)));
    
    % get the approximation to the pdf
    posterior = mvksdensity(samples,posterior_dom,'Bandwidth',silvermans_bw);

end






