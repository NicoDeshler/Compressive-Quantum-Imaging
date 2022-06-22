function rho_1 = rho_a_HG(aa_vec, C)
% Computes the single-photon excitation density operator in 
% the Hermite-Gauss (HG) mode representation for an incoherent scene
% expressed in the wavelet domain. The PSF of the system is assumed to 
% be a gaussian psf of width sigma_x = sigma_y.
%
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% aa_vec     : the augmented parameter vector related to the wavelet coefficients through the W matrix 
% C          : a stack of matrices containing the wavelet operator vector
%              transfromed by W
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% rho_1      : the density matrix

rho_1 = MatMulVecOp(aa_vec',C);
%disp(['rho trace: ', num2str(trace(rho_1))])
end

