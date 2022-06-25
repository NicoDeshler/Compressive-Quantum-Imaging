function Gamma_1 = Gamma_1_HG(h, Gamma_1_vec)
% Computes the 1th-index Personick Operator represented in the
% Hermite-Gauss basis for a scene described via a wavelet transform.
% This operator is used for solving the equation:
%           2 x Gamma_1 = B Gamma_0 + Gamma_0 B 
% where B is the Quantum Minimum Mean-Squared Estimator (QMMSE).
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% h             : a unit vector for computing the joint parameter dot(h,a_vec)
% Gamma_1_vec   : a stack of first-moment operators for each parameter
%
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% Gamma_1   : a matrix with dimensions equal to size(C(:,:,1))

% take inner product
Gamma_1 = MatMulVecOp(h',Gamma_1_vec);
end
