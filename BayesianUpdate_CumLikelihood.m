function [a_vec, a_mu_post, a_cov_post] = BayesianUpdate_CumLikelihood(...
    l_vec_stack, B_gamma_stack, C,...
    a_mu0, a_cov0, a_mu1, a_cov1, q,...
    N_samples, ref_mu, ref_sigma,...
    W,wv_idx,WaveletName)


n_as = numel(a_mu0);

% define function handles from computing the pdf
prior_fn = @(x) MVGBM_prior(x,a_mu0,a_cov0,a_mu1,a_cov1,q);

% pdf with non-negativity penalty (soft simplex boundary)
% pdf = @(x) cumulative_likelihood(l_vec_stack, B_gamma_stack, x, C) * prior_fn(x) * negativity_penalty(x, W, wv_idx, WaveletName);
% pdf with non-negativity constraint (hard simplex boundary)
pdf = @(x) cumulative_likelihood(l_vec_stack, B_gamma_stack, x, C) * prior_fn(x) * non_neg(x, W, wv_idx, WaveletName);

% use importance sampling to estimate mean and covariance of posterior
[a_mu_post,a_cov_post] = importance_sampling(pdf,N_samples,n_as,ref_mu,ref_sigma);
    
% MMSE estimator is given by the mean of the posterior
a_vec = a_mu_post;

end


%%%%%%%%%%%%%%%%%%%%% PRIOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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



%%%%%%%%%%%%%%%%%%%% CUMULATIVE LIKELIHOOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p_l = cumulative_likelihood(l_vec_stack,B_gamma_stack, a_vec, C)
    num_measurements = size(B_gamma_stack,3);
         
    % likelihood is conditional probability of observations with
    % measurement operators used up to the current point
    p_l = 1;
    for i = 1:num_measurements
        p_l = p_l * likelihood(l_vec_stack(:,i), B_gamma_stack(:,:,i), a_vec, C); 
    end
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
    %disp(a_cov_post)
    a_mu_post = a_mu_post';
end

%%%%%%%%%%%%%%%%%%%%%%% CONSTRAINT FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%
function p_penalty = negativity_penalty(a_vec,W,wv_idx,WaveletName)
    % calculates a probability factor to exponentially penalize proposal
    % samples of a_vec that lie outside the parameter vector simplex which
    % preserves the positivity of the reconstructed image.
    
    aa_vec = [a_vec; 1];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    fractional_neg = sum(img_est(img_est<0)) / max(sum(img_est(img_est>=0)),realmin);
    p_penalty = exp(fractional_neg * 1e3);
end

function is_non_neg = non_neg(a_vec,W,wv_idx,WaveletName)
% returns 1 if the predicted image is non-negative
% returns 0 otherwise
    aa_vec = [a_vec; 1];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    is_non_neg = min(img_est(:)) >= 0;
end


