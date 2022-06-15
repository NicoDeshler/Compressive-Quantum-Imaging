
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


