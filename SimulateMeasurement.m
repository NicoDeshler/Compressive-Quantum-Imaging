function l_vec = SimulateMeasurement(B_gamma, N_photons, A, gt_theta)
% Simulates the result of a photon counting measurement with the
% measurement operator B_gamma (the joint parameter estimator).
% The interrogated mixed state is the ground-truth state parametrized by
% the ground-truth wavelet coefficients. Each element of the output is the 
% number of photons observed in each eigenstate of B_gamma. 
%
% --------
% Inputs:
% --------
% N_photons - the transformation matrix that takes a_vec to theta_vec where a_vec
%     contains a list of unconstrained (yet lower dimensional) parameters
%     and theta_vec contains (constrained) wavelet coefficients
% B_gamma - a Hermitian matrix corresponding to an observable. In this case
%           it is the joint parameter estimator.
% A - A stack of matrices containing the OTF projectors after a wavelet
%     transform represented in the HG basis.
% gt_theta - the ground truth parameters of the scene (ground truth wavelet
% coefficients.
% --------
% Outputs:
% --------
% l_vec - a vector containing the number of photons observed in different 
%          eigenstates of B_gamma. The dimensionality of l_vec is
%          N_HG_modes x1.
%%

% compute the ground truth density matrix using the ground truth wavelet
% parameters
gt_rho = rho_wavelet_HG(A, gt_theta);

% Get measurement projectors (eigenstates) and possible measurement
% outcomes(eigenvalues) from the joint parameter represented in the HG
% basis.
[V,~] = eig(B_gamma,'vector');

% From the measurement projectors, get the probability distribution over
% eigenvalues of B_gamma of observing B_gamma. This is the true likelihood
% p(l_vec | gt_theta)

% Two requirements for the probability to be valid
%-----assert(all(V'*V == eye(size(V,1))));
%-----assert(trace(gt_rho)==1)

p_outcomes = diag(V'*gt_rho*V);

% take absolute value for numerical stability
p_outcomes = abs(p_outcomes);

% include the outcome probability for modes greater the N_HG_modes
p_outcomes = [p_outcomes; max(0,1-sum(p_outcomes))];

% normalize probability
p_outcomes = p_outcomes/sum(p_outcomes);
    
%% Generate samples from the likelihood
% sample the indices of each outcome from the discrete distribution
% p_outcomes associated with the measurement
ind = datasample(1:numel(p_outcomes),N_photons,'weights',p_outcomes)';

% l_vec contains the count of the number of times each outcome appeared
l_vec = accumarray(ind, 1);

% fill in the cases where no counts appeared
min_ind = min(ind);
max_ind = max(ind);
l_vec = [zeros(1-min_ind,1);l_vec;zeros(numel(p_outcomes)-max_ind,1)];