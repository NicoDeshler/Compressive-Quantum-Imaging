
gen_wavelet_sparse_img('db1',2,[4,4],

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
    K = floor((1-q)*numel(C));
    ws(sort_idx(K+1:end)) = 0;
    img_sparse = waverec2(ws,S,WaveletName);
end


