function Gamma_i1 = Gamma_i1_HG(C,aa_mu,aa_var)
% Computes the stack of first moment operators for each the 
% unconstrained parameters.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% C         :  a stack of matrices representing the wavelet operators transformed by W matrix
% aa_mu     :  expectations of the unconstrained parameters
% aa_var    :  variances of the parameters
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% Gamma_i1  : the stack of first moment operators. Gamma_i1 = Gamma_i1_stack(:,:,i) 
    
    % get the matrix E[a_i a_j] of expected values for pair-wise parameter 
    % products
    M = diag(aa_var) + aa_mu*aa_mu';   
    
    % matrix-vector multiplication with vector operator Gamma_1_vec =  W * C_vec
    Gamma_i1 = MatMulVecOp(M,C);
    
end


