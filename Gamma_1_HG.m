function Gamma_1 = Gamma_1_HG(C_vec,h,aa_mu,aa_var)
% Computes the 1th-index Personick Operator represented in the
% Hermite-Gauss basis for a scene described via a wavelet transform.
% This operator is used for solving the equation:
%           2 x Gamma_1 = B Gamma_0 + Gamma_0 B 
% where B is the Quantum Minimum Mean-Squared Estimator (QMMSE).
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% C_vec     :  a stack of matrices representing the wavelet operators transformed by W matrix
% h         : a unit vector for computing the joint parameter dot(h,a_vec)
% aa_mu     : a vector containing the expected values of the 'a' parameters
% aa_var    : a vector containing the variances of the 'a' parameters
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% Gamma_1   : a matrix with dimensions equal to size(C(:,:,1))

% The second moment matrix M = E[a a'] (assumes parameters are independent)
M = diag(aa_var) + aa_mu*aa_mu';

% intermediate vector
x = M*h;

% take inner product
Gamma_1 = MatMulVecOp(x',C_vec);
end
