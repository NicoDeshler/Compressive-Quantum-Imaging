
% Generate images that are sparse in wavelet domain
%[img,coeffs] = gen_wavelet_sparse_img('db1',1,[2,2],.75);


% Test importance sampling
pdf_mu = [1,1];
pdf_sig = eye(2);
pdf = @(x) 5 * mvnpdf(x,pdf_mu,pdf_sig);
N_samples = 1e6;
ref_mu = [0,0];
ref_sig = eye(2);

[x_mu,x_sig] = importance_sampling(pdf,N_samples,ref_mu,ref_sig);







function sample = sampleGBMPrior(q,mu,z)
    x1 = normrnd(mu(:,1),z(:,1));
    x2 = normrnd(mu(:,2),z(:,2));
    coinflips = binornd(1,q,[numel(x1),1]);
    
    aa_vec = coinflips.*x1 + ~coinflips.*x2;
    aa_vec(end)=1;
end


function [img_sparse, ws] =  gen_wavelet_sparse_img(WaveletName,WaveletLevel,img_dims,q)

    %%
    % WaveletName - name of wavelet decomposition
    % Wavlet Level - level of decomposition desired
    % img_dims - [x,y] dimensions of the input image undergoing the
    % transform
    % q - sparsity coefficients q = # zeros/#wavelets
    % ---------------------------------------------
    % img_sparse - sparse image in wavelet domain
    % ws - sparse wavelet coefficient vector for img
    %%
    img = abs(randn(img_dims));
    [C,S] = wavedec2(img,WaveletLevel,WaveletName);
    ws = C;                       % sparsified wavelet coefficient vector
    [~,sort_idx] = sort(abs(ws),'descend');
    K = floor(q*numel(C));
    ws(sort_idx(K+1:end)) = 0;
    img_sparse = waverec2(ws,S,WaveletName);
end

function Y_vec = MatMulVecOp(A,X_vec)
    % performs matrix-vector multiplication between a matrix and a vector
    % operator represented by a stack of matrices.
    % Operation in LaTex notation:
    %                   $$ \hat{\vec{Y} = A \hat{\vec{X}} $$
    % --------
    % Inputs:
    % --------
    % A     : a [p, n] matrix
    % X_vec : a [m, m, n] stack of matrices representing the input vector
    %         operator. The operators are indexed along the 3rd dimension.
    % --------
    % Outputs:
    % --------
    % Y_vec : a [m, m, p] stack of matrices representing the output vector
    %         operator. The operators are indexed along the 3rd dimension.

    Y_vec = squeeze(sum(reshape(A',[1,1,size(A')]).*reshape(X_vec,[size(X_vec),1]),3));

end



function samples = MCMC_sampling(start,pdf,method,Ns,Nb)


samples = zeros([numel(start),Ns+Nb]);

    
    
end


function [x_mu,x_sig] = importance_sampling(pdf,N,ref_mu,ref_sig)
    
    % number of random variables
    n = numel(ref_mu);
    
    % reference distribution (Multi-variate gaussian)
    ref_pdf = @(x)mvnpdf(x,ref_mu,ref_sig);
    
    % get N samples from reference distribution
    f_samples = mvnrnd(ref_mu,ref_sig,N);
    
    % ratio of pdfs
    pdf_ratio = pdf(f_samples)./ref_pdf(f_samples);
    
    % approximate normalizing constant of pdf
    C = mean(pdf_ratio);
    
    % probability ratio
    prob_ratio = pdf_ratio/C;
    
    % approximate expected value of the normalized pdf
    x_mu = mean(f_samples.*prob_ratio,1);
    
    % approximate the covariance matrix
    delta = f_samples - x_mu;
    d_vert = reshape(delta',[n,1,N]);
    d_horz = reshape(delta',[1,n,N]);
    dyad_stack = pagemtimes(d_vert,d_horz);
    prob_ratio = reshape(prob_ratio,[1,1,N]);
    x_sig = mean(dyad_stack.*prob_ratio,3);
end


function is_pos = non_neg(a_vec, wv_idx,WaveletName)
% returns 1 if the predicted image is non-negative
% returns 0 otherwise
    aa_vec = [a_vec; 0];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    is_pos = min(img_est(:)) >= 0;
end