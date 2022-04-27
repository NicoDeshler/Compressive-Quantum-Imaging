function Gamma_0 = Gamma_0_HG(A,W,aa_mu)
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
% A - A stack of matrices containing the OTF projectors after a wavelet
%     transform represented in the HG basis.
% W - the transformation matrix that takes a_vec to theta_vec where a_vec
%     contains a list of unconstrained (yet lower dimensional) parameters
%     and theta_vec contains (constrained) wavelet coefficients
% aa_mu - a vector containing the expected values of the parameters in the
% augmented parameter a_vec given the prior.
%
% --------
% Outputs:
% --------
% Gamma_0 - A matrix with dimensions equal to size(A(:,:,1))
theta_mu = W * aa_mu;
Gamma_0 = sum(A.*reshape(theta_mu,1,1,numel(theta_mu)),3); % Equals sum_j E[theta_j]*A_j


