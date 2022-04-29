function rho_1 = rho_wavelet_HG(A, theta_vec)
% Computes the single-photon excitation density operator in 
% the Hermite-Gauss (HG) mode representation for an incoherent scene
% expressed in the wavelet domain. The OTF of the system is assumed to 
% be a gaussian psf of width sigma_x = sigma_y.
%
% --------
% Inputs:
% --------
% theta_vec     - A vector containing the wavelet coefficients 
% A             - A stack of matrices containing the OTF projectors after a wavelet
%                 transform represented in the HG basis. 
% --------
% Outputs:
% --------
% rho_1         - The density matrix with dimensions equal to A(:,:,1)

rho_1 = sum(reshape(theta_vec,1,1,numel(theta_vec)).*A,3);
rho_1 = rho_1/trace(rho_1);             % artificially make density matrix trace 1
end
