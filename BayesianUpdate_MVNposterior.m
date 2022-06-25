% Bayesian update for the posterior
function [a_vec, a_mu_post, a_cov_post] = BayesianUpdate_MVNposterior(...
    l_vec, B_gamma, C,...
    a_mu_prior, a_cov_prior,...
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
% a_mu_prior : 
% a_cov_prior:
% N_samples    : number of Metropolis-Hastings MCMC samples used to compute
%             approximate the posterior.
% N_burn    : number of Metropolis-Hasting MCMC samples to discard in order
%             to let the Markov Chain reach steady state.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% a_mu_post              : the updated estimate on the constrained parameter vector
% a_cov_post             : covariance matrix of the posterior 
% ----------------------------------------------------------------
% NAME/VALUE:
% ----------------------------------------------------------------


p = inputParser;
n_as = numel(a_mu_prior);

prior_fn = @(x) mvn_prior(x,a_mu_prior,a_cov_prior);
pdf = @(x) likelihood(l_vec, B_gamma, x, C) * constrain_prior(x, prior_fn, W, wv_idx, WaveletName);

switch method
    case 'interior'
        % Interior Sampling (works well for low-dimensional cases)
        addRequired(p,'a_min')
        addRequired(p,'a_max')
        parse(p, varargin{:})
        
        a_min = p.Results.a_min;
        a_max = p.Results.a_max;
        
        [a_mu_post,a_cov_post] = interior_sampling(pdf,N_samples,n_as,a_min,a_max);
        
    case 'importance'
        % Importance Sampling 
        % Approximates posterior as a multi-variate gaussian distribution      
        addRequired(p,'ref_mu')
        addRequired(p,'ref_sigma')
        parse(p, varargin{:})
        
        ref_mu = p.Results.ref_mu;
        ref_sigma = p.Results.ref_sigma;        
        
        [a_mu_post,a_cov_post] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sigma);
    
    case 'slice'
        % Slice MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})
        
        N_burn = p.Results.N_burn;
        start = p.Results.start;

        [a_mu_post, a_cov_post] = slice_sampling(pdf,N_samples,n_as,N_burn,start);
    
    case 'MH'
        % Metropolis-Hastings MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})

        N_burn = p.Results.N_burn;
        start = p.Results.start;
        
        [a_mu_post, a_cov_post] = MH_sampling(pdf,N_samples,n_as,N_burn,start);
        
    otherwise
        error('Unsupported sampling method.')        
end

% MMSE estimator is given by the mean of the posterior
a_vec = a_mu_post;

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

%disp(['Prob Sum: ', num2str(sum(p_outcomes))]) 

% take absolute value for numerical stability
p_outcomes = abs(p_outcomes);

% include the outcome probability for modes greater the N_HG_modes
p_outcomes = [p_outcomes; max(0,1-sum(p_outcomes))];

% normalize probability
p_outcomes = p_outcomes/sum(p_outcomes);
    
% likelihood is a multinomial
p_l = mnpdf(l_vec, p_outcomes);

% handle numerically unstable cases
%if sum(isnan(p_l))>0
%    p_l = 0;
%end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p_a = constrain_prior(a_vec,prior_fn,W,wv_idx,WaveletName)
    % imposes non-negativity constraint
    %p_a = prior_fn(a_vec) * non_neg(a_vec,W,wv_idx,WaveletName);
    p_a = prior_fn(a_vec);
end


function p_a = mvn_prior(a_vec, a_mu, a_cov)
% Calculates the probability of observing a_vec given
% a generalized probability density approximation for each parameter.
% This prior assumes the parameters are independent.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec         : An instance of the parameter vector with prior given by the
%                 other inputs (priors, prior_doms)
% a_mu_prior    : .
% a_cov_prior   :
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a           : probability P(a_vec)

p_a = mvnpdf(a_vec,a_mu,a_cov);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAMPLING METHODS %%%%%%%%%%%%%%%%%%%%%%%
function [a_mu_post, a_cov_post] = interior_sampling(pdf,N_samples,n_as,a_min,a_max)
% Brute force samples the pdf p(l|a)*p(a) over a domain of
% interest and marginalizes to get posteriors.
    N = floor(N_samples^(1/n_as)); 
    dom = linspace(a_min,a_max,N);
    [a_vec_dom{1:n_as}] = ndgrid(dom);
    
    % probability grid container
    p_a_vec = zeros(size(a_vec_dom{1}));
    
    % first and second moments container
    E1_a = zeros([n_as,1]);
    E2_a = zeros(n_as);
    
    % vector container
    a_vec = zeros([n_as,1]);
    for i = 1:numel(p_a_vec)
        for j = 1:n_as
            a_j = a_vec_dom{j}; 
            a_vec(j) = a_j(i);
        end
        p_a_vec(i) = pdf(a_vec);
        E1_a = E1_a + p_a_vec(i) * a_vec;
        E2_a = E2_a + p_a_vec(i) * (a_vec*a_vec');
    end
       
    % normalization constant
    C = sum(p_a_vec(:));
    
    % normalize
    % p_a_vec = p_a_vec/C;
    E1_a = E1_a/C;
    E2_a = E2_a/C;
    
    % outputs
    a_mu_post = E1_a;
    a_cov_post = E2_a - (E1_a*E1_a');
    
end


function [a_mu_post, a_cov_post] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sig)
    
    % reference distribution (Multi-variate gaussian)
    ref_pdf = @(x)mvnpdf(x,ref_mu',ref_sig);
    
    % get N samples from reference distribution
    ref_samples = mvnrnd(ref_mu',ref_sig,N_samples);
    
    % ratio of pdfs
    sample_pdf = cellfun(pdf,num2cell(ref_samples',1))';
    pdf_ratio = sample_pdf./ref_pdf(ref_samples);
    
    % remove numerically unstable points
    pdf_ratio = pdf_ratio(~isnan(pdf_ratio));
    
    % approximate normalizing constant of pdf
    C = mean(pdf_ratio);
    
    % probability ratio
    prob_ratio = pdf_ratio/C;
    
    % approximate expected value of the normalized pdf
    a_mu_post = mean(ref_samples.*prob_ratio,1);
    
    % approximate the covariance matrix
    delta = ref_samples - a_mu_post;
    d_vert = reshape(delta',[n_as,1,N_samples]);
    d_horz = reshape(delta',[1,n_as,N_samples]);
    dyad_stack = pagemtimes(d_vert,d_horz);
    prob_ratio = reshape(prob_ratio,[1,1,N_samples]);
    a_cov_post = mean(dyad_stack.*prob_ratio,3);
    a_mu_post = a_mu_post';
    
end

function [a_mu_post, a_cov_post] = slice_sampling(pdf,N_samples,n_as,N_burn,start)
    % Slice Sampling
    samples = slicesample(start, N_samples,'pdf',pdf,'burnin',N_burn)';
    
    a_mu_post = mean(samples,2);
    a_cov_post = cov(samples);
end

function [a_mu_post,a_cov_post]= MH_sampling(pdf,N_samples,n_as,N_burn,start)
    % Metropolis Hastings Sampling
    % proportional probability density to posterior
    proprnd = @(x) mvnrnd(x, eye(n_as));
    samples = mhsample(start, N_samples, 'pdf', pdf, 'proprnd', proprnd, 'burnin', N_burn,'symmetric',1,'nchain',1)';
    
    a_mu_post = mean(samples,2);
    a_cov_post = cov(samples);
end

function is_non_neg = non_neg(a_vec,W,wv_idx,WaveletName)
% returns 1 if the predicted image is non-negative
% returns 0 otherwise
    aa_vec = [a_vec; 1];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    is_non_neg = min(img_est(:)) >= 0;
end

