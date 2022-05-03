function h = h_proj(Gamma_0,B_stack, aa_mu, aa_var)
% Compute joint parameter projection vector h

% Inputs:
% aa_var: augmented unconstrained parameter vector variances
% aa_mu : augmented uncontrained prameter vector means
% Gamma_0 : The mena density operator
% B_stack : A matrix stack of the optimal parameter estimators in HG
% representation
% 

% Outputs:
% h : The joint parameter projection vector

M = diag(aa_var) + aa_mu*aa_mu'; 

g = size(B_stack,3);
G = zeros([g,g]);

for i = 1:g
    for j = 1:g
    G(i,j) = 1/2 * trace(Gamma_0*(B_stack(:,:,i)*B_stack(:,:,j)+B_stack(:,:,j)*B_stack(:,:,i))/2);
    end
end

% Quantum Bayesian Cramer-Rao Lower Bound (QBCRLB)
Sigma_Q = M - G;

% Get eigenvector corresponding to maximum eigenvalue of the QBCRLB
[V,lam] = eig(Sigma_Q);
[~, max_eigval_idx] = max(lam);
h = V(:,max_eigval_idx); % joint parameter vector
end
