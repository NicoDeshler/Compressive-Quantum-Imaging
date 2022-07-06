function [img_sparse, ws] =  gen_wavelet_sparse_img(WaveletName,WaveletLevel,img_dims,q)

    % Generates a target image image that is q sparse in the wavelet domain
    % of choice.
    %
    % ------------------------------------------------------------
    % INPUTS:
    % ------------------------------------------------------------
    % WaveletName   - name of wavelet decomposition
    % WavletLevel   - level of decomposition desired
    % img_dims      - [x,y] dimensions of the input image undergoing the
    %                   transform
    % q             - fractional sparsity  K/N = (# non-zero params/ # params)
    % ------------------------------------------------------------
    % OUTPUTS:
    % ------------------------------------------------------------
    % img_sparse    - sparse image in wavelet domain
    % ws            - sparse wavelet coefficient vector for img
    
    
    img = abs(randn(img_dims));
    [C,S] = wavedec2(img,WaveletLevel,WaveletName);
    ws = C;                       % sparsified wavelet coefficient vector
    [~,sort_idx] = sort(abs(ws),'descend');
    K = floor(q*numel(C));
    ws(sort_idx(K+1:end)) = 0;
    img_sparse = waverec2(ws,S,WaveletName);
end