function Gamma_i1_stack = Gamma_i1_HG(A,W,aa_mu,aa_var)
% --------
% Inputs:
% --------
% --------
% Outputs:
% --------
% NOTE: Gamma_i1 = Gamma_i1_stack(:,:,i)  
    M = diag(aa_var) + aa_mu*aa_mu';
    WM  = W*M;
    n_params = numel(aa_mu);
    Gamma_i1_stack = zeros(A);
    
    for i = 1:n_params
        Gamma_i1_stack(:,:,i) = sum(reshape(WM(:,i),[1,1,n_params]).*A,3);
    end
    %Gamma_i1_stack = squeeze(sum(reshape(A,[size(A),1]).* reshape(W*M,[1,1,size(W*M)]),3));
end