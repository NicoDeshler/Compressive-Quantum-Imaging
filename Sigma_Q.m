function E_Q = Sigma_Q(Gamma_0, B, aa_mu, aa_var)

% Computes the Bayesian Quantum Cramer-Rao Lower Bound matrix
% which is similar to the covariance matrix of the estimators.
% Reference: "Bayesian multiparameter Quatnum Metrology with Limited Data"
% Rubio and Dunningahm (2020)
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
% E_Q         : the BQCRLB

E_aa = diag(aa_var) + aa_mu*aa_mu'; 

g = size(B,3);
K = zeros([g,g]);

for i = 1:g
    for j = 1:g
    K(i,j) = 1/2 * trace(Gamma_0*(B(:,:,i)*B(:,:,j)+B(:,:,j)*B(:,:,i))/2);
    end
end

% Quantum Bayesian Cramer-Rao Lower Bound (QBCRLB)
E_Q = E_aa - K;
end