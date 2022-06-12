function h = h_proj(Gamma_0, B, aa_mu, aa_var)
% Compute joint parameter projection vector h
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% aa_var    : augmented unconstrained parameter vector variances
% aa_mu     : augmented uncontrained prameter vector means
% Gamma_0   : the expectation of the density operator
% B         : a matrix stack of the optimal parameter estimators in HG
% representation
% 
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% h         : the joint parameter projection vector

M = diag(aa_var) + aa_mu*aa_mu'; 

g = size(B,3);
G = zeros([g,g]);

for i = 1:g
    for j = 1:g
    G(i,j) = 1/2 * trace(Gamma_0*(B(:,:,i)*B(:,:,j)+B(:,:,j)*B(:,:,i))/2);
    end
end

% Quantum Bayesian Cramer-Rao Lower Bound (QBCRLB)
Sigma_Q = M - G;

% Remove matrix elements corresponding to the augmented parameter
Sigma_Q = Sigma_Q(1:end-1,1:end-1);

% Get eigenvector corresponding to maximum eigenvalue of the QBCRLB
[V,lam] = eig(Sigma_Q,'vector');

[~, min_eigval_idx] = min(lam);
h = V(:,min_eigval_idx(1)); % joint parameter vector

% choose max eigenvector
%[~, max_eigval_idx] = max(lam);
%h = V(:,max_eigval_idx(1)); % joint parameter vector

% augment the projection vector
h = [h;0];
end
