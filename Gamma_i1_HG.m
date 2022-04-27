function Gamma_i1_stack = Gamma_i1_HG(A,W,aa_mu,aa_var)
% --------
% Inputs:
% --------
% --------
% Outputs:
% --------
    M = diag(aa_var) + aa_mu*aa_mu';
    Gamma_i1_stack = squeeze(sum(reshape(A,[size(A),1]).* reshape(W*M,[1,1,size(W*M)]),3));
    %NOTE: Gamma_i1 = Gamma_i1_stack(:,:,i)  
end