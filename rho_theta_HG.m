function rho_1 = rho_theta_HG(theta_vec, A)
% Computes the single-photon excitation density operator in 
% the Hermite-Gauss (HG) mode representation for an incoherent scene
% expressed in the wavelet domain. The PSF of the system is assumed to 
% be a gaussian psf of width sigma_x = sigma_y.
%
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% theta_vec     : a vector containing the wavelet coefficients 
% A             : a stack of matrices containing the wavelet operators
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% rho_1         : the density matrix

rho_1 = MatMulVecOp(theta_vec',A);

end