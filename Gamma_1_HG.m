%function Gamma_1 = Gamma_1_HG(A,W,h,aa_mu,aa_var)
function Gamma_1 = Gamma_1_HG(C,h,aa_mu,aa_var)
% Computes the 1th-index Personick Operator represented in the
% Hermite-Gauss basis for a scene described via a wavelet transform.
% This operator is used for solving the equation:
%           2 x Gamma_1 = B Gamma_0 + Gamma_0 B 
% where B is the Quantum Minimum Mean-Squared Estimator (QMMSE).
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% A         : a stack of matrices containing the OTF projectors after a wavelet
%             transform represented in the HG basis.
% W         : the transformation matrix that takes a_vec to theta_vec where a_vec
%             contains a list of unconstrained (yet lower dimensional) parameters
%             and theta_vec contains (constrained) wavelet coefficients
% h         : a unit vector for computing the joint parameter dot(h,a_vec)
% aa_mu     : a vector containing the expected values of the 'a' parameters
% aa_var    : a vector containing the variances of the 'a' parameters
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% Gamma_1   : a matrix with dimensions equal to size(C(:,:,1))

% get the matrix E[a_i a_j] of expected values for pair-wise parameter 
% products
M = diag(aa_var) + aa_mu*aa_mu';

% intermediate vector
x = M*h;

% take inner product
Gamma_1 = MatMulVecOp(x',C);
end
