function [a_vec, a_mu_post, a_cov_post] = BayesianUpdate_MVGBMposterior(...
    l_vec, B_gamma, C,...
    a_mu0, a_cov0, a_mu1, a_cov1, q,...
    N_samples,...
    W,wv_idx,WaveletName)

n_as = numel(a_mu0);

prior_fn = @(x) MVGBM_prior(x,a_mu0,a_cov0,a_mu1,a_cov1,q);
pdf = @(x) likelihood(l_vec, B_gamma, x, C) * constrain_prior(x, prior_fn, W, wv_idx, WaveletName);

ref_mu = (1-q)*a_mu0 + (q)*a_mu1;
ref_sigma = (1-q)*(a_cov0 + a_mu0*a_mu0') + (q)*(a_cov1 + a_mu1*a_mu1') - ref_mu*ref_mu';

[a_mu_post,a_cov_post] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sigma);
    
% MMSE estimator is given by the mean of the posterior
a_vec = a_mu_post;

end


%%%%%%%%%%%%%%%%%%%%% PRIOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p_a = constrain_prior(a_vec,prior_fn,W,wv_idx,WaveletName)
    % imposes non-negativity constraint
    %p_a = prior_fn(a_vec) * non_neg(a_vec,W,wv_idx,WaveletName);
    p_a = prior_fn(a_vec);
end

function p_a = MVGBM_prior(a_vec, mu0, cov0, mu1, cov1, q)
% A multivariate gaussian binomial mixture prior
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec       : An instance of the parameter vector at which to evaluate
%                 the prior
% mu0         : the mean of the 
% mu1         : 
% cov0        :
% cov1        :
% q           : the sparsity coefficient
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a         : the probability of a_vec

p_a = (1-q)*mvnpdf(a_vec, mu0, cov0) + (q)*mvnpdf(a_vec, mu1, cov1);
end

%%%%%%%%%%%%%%%%%%%% SAMPLING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    % (compatible with Matlab 2021a)
    % dyad_stack = pagemtimes(d_vert,d_horz);
    
    % (compatible with Matlab 2017)
    dyad_stack = zeros([n_as,n_as,N_samples]);
    for i=1:N_samples
        dyad_stack(:,:,i) = d_vert(:,:,i)*d_horz(:,:,i);
    end
    
    prob_ratio = reshape(prob_ratio,[1,1,N_samples]);
    a_cov_post = mean(dyad_stack.*prob_ratio,3);
    disp(a_cov_post)
    a_mu_post = a_mu_post';
end
