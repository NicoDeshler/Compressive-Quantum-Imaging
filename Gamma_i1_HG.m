function Gamma_i1 = Gamma_i1_HG(C_vec,aa_mu,aa_cov)
% Computes the stack of first moment operators for each the 
% unconstrained parameters.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% C_vec     :  a stack of matrices representing the wavelet operators transformed by W matrix
% aa_mu     :  expectations of the unconstrained parameters
% aa_cov    :  covariance matrix of the parameters
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% Gamma_i1  : the stack of first moment operators. Gamma_i1 = Gamma_i1_stack(:,:,i) 
    
% The second moment matrix E[a a']
E_aa = aa_cov + aa_mu*aa_mu';


% matrix-vector multiplication with vector operator Gamma_1_vec =  W * C_vec
Gamma_i1 = MatMulVecOp(E_aa,C_vec);

end


