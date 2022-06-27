% Bayesian update for the posterior
function [a_vec, a_mu_post, a_cov_post, posteriors, posterior_doms] = BayesianUpdate(l_vec, B_gamma, C, priors, prior_doms,...
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
% priors    : A matrix of dimensions [n_params-1,100]. The i'th row contains
%             points on the i'th parameter's prior distribution.
% prior_doms: A matrix of dimensions [numel(a_vec),100]. Each row
%             contains the domain points associated with points on the prior
%             distribution.
% N_samples    : number of Metropolis-Hastings MCMC samples used to compute
%             approximate the posterior.
% N_burn    : number of Metropolis-Hasting MCMC samples to discard in order
%             to let the Markov Chain reach steady state.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% a_vec             : the updated estimate on the constrained parameter vector
% a_mu_post         :
% a_cov_post        :
% posteriors        : A matrix of dimensions [n_params-1,100]. The i'th row contains
%                     points on the i'th parameter's posterior distribution. 
% posterior_domains : A matrix of dimensions [n_params-1,100]. Each row
%                     contains the domain points associated with points on
%                     the posterior distribution. 
% ----------------------------------------------------------------
% NAME/VALUE:
% ----------------------------------------------------------------


p = inputParser;
n_as = size(priors,1);

prior_fn = @(x) ksdensity_prior(x,priors,prior_doms);
pdf = @(x) likelihood(l_vec, B_gamma, x, C) * constrain_prior(x, prior_fn, W, wv_idx, WaveletName);

switch method
    case 'interior'
        % Interior Sampling (works well for low-dimensional cases)
        addRequired(p,'a_min')
        addRequired(p,'a_max')
        parse(p, varargin{:})
        
        a_min = p.Results.a_min;
        a_max = p.Results.a_max;
        
        [posteriors, posterior_doms] = interior_sampling(pdf,N_samples,n_as,a_min,a_max);
        
    case 'importance'
        % Importance Sampling      
        addRequired(p,'ref_mu')
        addReqired(p,'ref_sigma')
        parse(p, varargin{:})
        
        ref_mu = p.Results.ref_mu;
        ref_sigma = p.Results.ref_sigma;        
        
        [posteriors, posterior_doms] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sigma);
    
    case 'slice'
        % Slice MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})
        
        N_burn = p.Results.N_burn;
        start = p.Results.start;
        
        [posteriors, posterior_doms] = slice_sampling(pdf,N_samples,n_as,N_burn,start);
    
    case 'MH'
        % Metropolis-Hastings MCMC method
        addRequired(p,'N_burn')
        addRequired(p,'start')
        parse(p, varargin{:})

        N_burn = p.Results.N_burn;
        start = p.Results.start;
        
        [posteriors, posterior_doms] = MH_sampling(pdf,N_samples,n_as,N_burn,start);
        
    otherwise
        error('Unsupported sampling method.')        
end


% compute the variances and the means from the posteriors
a_mu_post = sum(posteriors.*posterior_doms,2);
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


function p_a = ksdensity_prior(a_vec, priors, prior_doms)
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

p_a_vec = zeros([numel(a_vec),1]);
for i = 1:numel(a_vec)
   p_a_vec(i) = interp1(prior_doms(i,:),priors(i,:),a_vec(i),'nearest','extrap');
end

p_a = prod(p_a_vec);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAMPLING METHODS %%%%%%%%%%%%%%%%%%%%%%%
function [posteriors, posterior_doms] = interior_sampling(pdf,N_samples,n_as,a_min,a_max)
% Brute force samples the pdf p(l|a)*p(a) over a domain of
% interest and marginalizes to get posteriors.
    N = floor(N_samples^(1/n_as)); 
    a_rng = linspace(a_min,a_max,N);
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


function [posteriors, posterior_doms] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sig)
    
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
    x_mu = mean(ref_samples.*prob_ratio,1);
    
    % approximate the covariance matrix
    delta = ref_samples - x_mu;
    d_vert = reshape(delta',[n_as,1,N_samples]);
    d_horz = reshape(delta',[1,n_as,N_samples]);
    
    % (compatible with Matlab 2021a)
    % dyad_stack = pagemtimes(d_vert,d_horz);
    
    % (compatible with Matlab 2017)
    dyad_stack = zeros([n_as,n_as,N_samples]);
    for i=1:N_samples
        dyad_stack(:,:,i) = d_horz(:,:,i)*d_vert(:,:,i);
    end
       
    prob_ratio = reshape(prob_ratio,[1,1,N_samples]);
    x_sig = mean(dyad_stack.*prob_ratio,3);
    
    % assume the posterior follows a gaussian
    samples = mvnrnd(x_mu,x_sig,N_samples);
    
    % Use kernel density estimation to generate a smooth approximation of the
    % posterior
    posteriors = zeros([n_as,100]);
    posterior_doms = zeros([n_as,100]);
    for i = 1:n_as
        [posteriors(i,:), posterior_doms(i,:)] = ksdensity(samples(:,i));
    end
    
    % normalize posteriors
    posteriors = posteriors./sum(posteriors,2);
    
end

function [posteriors,posterior_doms] = slice_sampling(pdf,N_samples,n_as,N_burn,start)
    % Slice Sampling
    samples = slicesample(start, N_samples,'pdf',pdf,'burnin',N_burn);

    % Use kernel density estimation to generate a smooth approximation of the
    % posterior
    posteriors = zeros([n_as,100]);
    posterior_doms = zeros([n_as,100]);
    for i = 1:n_as
        [posteriors(i,:), posterior_doms(i,:)] = ksdensity(samples(:,i));
    end
    
    % normalize posteriors
    posteriors = posteriors./sum(posteriors,2);
end

function [posteriors,posterior_doms]= MH_sampling(pdf,N_samples,n_as,N_burn,start)
    % Metropolis Hastings Sampling
    % proportional probability density to posterior
    proprnd = @(x) mvnrnd(x, eye(n_as));
    samples = mhsample(start, N_samples, 'pdf', pdf, 'proprnd', proprnd, 'burnin', N_burn,'symmetric',1,'nchain',1);
    
    % Use kernel density estimation to generate a smooth approximation of the
    % posterior
    posteriors = zeros([n_as,100]);
    posterior_doms = zeros([n_as,100]);
    for i = 1:n_as
        [posteriors(i,:), posterior_doms(i,:)] = ksdensity(samples(:,i));
    end
    
    % normalize posteriors
    posteriors = posteriors./sum(posteriors,2);
end

function is_non_neg = non_neg(a_vec,W,wv_idx,WaveletName)
% returns 1 if the predicted image is non-negative
% returns 0 otherwise
    aa_vec = [a_vec; 1];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    is_non_neg = min(img_est(:)) >= 0;
end


%% LEGACY SCRIPTS
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