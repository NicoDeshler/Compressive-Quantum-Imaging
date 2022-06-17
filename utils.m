
[img,coeffs] = gen_wavelet_sparse_img('db1',1,[2,2],.75);

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


function x_mu = importance_sampling(pdf,N,n_as)

    % reference distribution
    f_mu = ones([n_as,1]);
    f_sig = eye(n_as);    
    f = @(x)mvnpdf(x,f_mu,f_sig);
    
    % get N samples from reference distribution
    f_samples = mvnrnd(f_mu,f_sig,N);
    
    % approximate normalizing constant of pdf
    C = 1/N * sum(pdf(f_samples)./f(f_samples));
    
    % approximate expected value of the normalized pdf
    x_mu = 1/N * sum(f_samples.*pdf(f_samples)./(f_samples))/C;
    
    % approximate the variance of the normalized pdf
    x_mu2 = 1/N * sum(f_samples.^2.*pdf(f_samples)./(f_samples))/C;
    x_var = x_mu2 - x_mu.^2;
    
end
        



function is_pos = non_neg(a_vec, wv_idx,WaveletName)
% returns 1 if the predicted image is non-negative
% returns 0 otherwise
    aa_vec = [a_vec; 0];
    theta_vec = W*aa_vec;
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    is_pos = min(img_est(:) >= 0;
end