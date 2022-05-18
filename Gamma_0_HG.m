function Gamma_0 = Gamma_0_HG(C,aa_mu)
% Computes the 0th-index Personick Operator represented in the
% Hermite-Gauss basis for a scene described via a wavelet transform.
% This operator is used for solving the equation:
%           2 x Gamma_1 = B Gamma_0 + Gamma_0 B 
% where B is the Quantum Minimum Mean-Squared Estimator (QMMSE) for the
% joint parameter gamma = dot(h_vec,a_vec). Note that this function only
% applies to constrained parameters a_vec which are INDPENDENT random
% variables.
%
% --------
% Inputs:
% --------
% C     - the transformed wavelet operators
% aa_mu - a vector containing the expected values of the parameters in the
% augmented parameter a_vec given the prior.
%
% --------
% Outputs:
% --------
% Gamma_0 - A matrix with dimensions equal to size(A(:,:,1))

Gamma_0 = MatMulVecOp(aa_mu',C);
    
end