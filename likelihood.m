function p_l = likelihood(l_vec, B_gamma, a_vec, C)
% The likelihood (probability) of observing l_vec under the measurement
% operator B_gamma (the joint estimator). The likelihood function is a
% multinomial over the probabilty distribution of the outcomes.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% l_vec     : measurement vector containing the number of photons detected in each
%             of the eigenstates of the joint estimator B_gamma l_vec = [n_1,n_2,...,n_N]
% B_gamma   : The measurement matrix for the joint estimator.
% C         : a stack of the transformed wavelet operators C =  W * A
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_l       : the probability of the outcome l_vec

% get the density matrix
aa_vec = [a_vec;1];
rho = rho_a_HG(aa_vec, C);

% get the probabilities of each measurement outcome
[V,~] = eig(B_gamma);
p_outcomes = diag(V'*rho*V);

% take absolute value for numerical stability
p_outcomes = abs(p_outcomes);

% include the outcome probability for modes greater the N_HG_modes
p_outcomes = [p_outcomes; max(0,1-sum(p_outcomes))];

% normalize probability
p_outcomes = p_outcomes/sum(p_outcomes);
    
% likelihood is a multinomial
p_l = mnpdf(l_vec, p_outcomes);
end