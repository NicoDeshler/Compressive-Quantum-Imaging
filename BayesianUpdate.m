% Bayesian update for the posterior
function [a_vec, posteriors, posterior_doms] = BayesianUpdate(l_vec, B_gamma, W, A, priors, prior_doms, N_MCMC, N_burn, start)
% Updates the estimate for the constrained parameter vector a_vec using a
% Bayesian inference scheme. The hyperparameters for the prior are also
% updated to serve the subsequeny iteration. The Likelihood is computed 
% from the most recent measurment outcome of the joint parameter estimator.
% The prior is a gaussian binomial mixture. To estimate the posterior, 
% the Metropolis-Hasting algorithm is employed.
%
% The hyperparameters for the subsequent iteration (mu_new,
% z_new) are computed as follows. 
% After sampling the posterior, we update the mean of the second binomial
% mixture to the empirical mean of the
% --------
% Inputs:
% --------
% l_vec     - 
% B_gamma   - the joint parameter estimator used to collect the measurement
% A         - a stack of wavelet operators
% mu        - a Nx2 matrix containing the means of the GBM prior for
%             each parameter
% z         - a Nx2 matrix containing the variances of the GBM prior for
%             each parameter
% q         - sparsity toggle for the prior
% N_MCMC    - number of Metropolis-Hastings MCMC samples used to compute
%             approximate the posterior.
% N_burn    - number of Metropolis-Hasting MCMC samples to discard in order
%             to let the Markov Chain reach steady state.
% --------
% Outputs:
% --------
% a_vec     - the updated estimate on the constrained parameter vector
% posterior - a vector of function fit samples from the posterior distribution 
% x         - a vecor 

% REDACTED OUTPUTS%%
% mu_new    - the updated matrix of means constituting the hyperparameters
%             for the prior. These are used in the next iteration.
% z_new     - the updated matrix of variances constituting the
%             hyperparameters for the GBM prior. These are used in the next
%             iteration.

% use metropolis-hastings to sample the joint distribution and estimate the new mean and variance
n_as = size(priors,1);
pdf = @(x)max(likelihood(l_vec, B_gamma, x', W, A) * prior(x', priors, prior_doms), realmin); % pdf cannot be 0 for MCMC
proppdf = @(x,y) mvnpdf(y,x);
proprnd = @(x) mvnrnd(x, eye(n_as));
samples = mhsample(start, N_MCMC, 'pdf', pdf, 'proppdf', proppdf, 'proprnd', proprnd, 'burnin', N_burn);

% Use kernel density estimation to generate a smooth approximation of the
% posterior
posteriors = zeros([n_as,100]);
posterior_doms = zeros([n_as,100]);
for i = 1:n_as
    [posteriors(i,:),posterior_doms(i,:)] = ksdensity(samples(:,i));
end

%samples = samples/1000;
%mu_new = [mu(:,1),mean(samples)'];
%mu_new = [mean(samples)',mean(samples)'];
%z_new = [z(:,1),var(samples)'];
%z_new = [var(samples)',var(samples)'];
a_vec = sum(posteriors.*posterior_doms,2); % estimator is the mean given the posterior

%{
c = clock;
fig1 = figure;
histogram(samples(:,1))
title('$P(\tilde{\vec{a}}_1 | \vec{l})$ ','interpreter','latex')
saveas(fig1,fullfile('a1',[num2str(c),'.png']))
close(fig1)

fig2 = figure; 
histogram(samples(:,2))
title('$P(\tilde{\vec{a}}_2 | \vec{l})$ ','interpreter','latex')
saveas(fig2,fullfile('a2',[num2str(c),'.png']))
close(fig2)

fig3 = figure; 
histogram(samples(:,3))
title('$P(\tilde{\vec{a}}_3 | \vec{l})$ ','interpreter','latex')
saveas(fig3,fullfile('a3',[num2str(c),'.png']))
close(fig3)
%}

end

function p_l = likelihood(l_vec, B_gamma, a_vec, W, A)
% The likelihood (probability) of observing l_vec under the measurement
% operator B_gamma (the joint estimator). The likelihood function is a
% multinomial over the probabilty distribution of the outcomes.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% l_vec: measurement vector containing the number of photons detected in each
% of the eigenstates of the joint estimator B_gamma l_vec = [n_1,n_2,...,n_N]
% B_gamma: The measurement matrix for the joint estimator.

% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_l : the probability of the outcome l_vec

[~,max_dim_idx] = max(size(a_vec));    
if max_dim_idx == 2
    a_vec = a_vec';
end

% get the density matrix
aa_vec = [a_vec;1];
theta_vec = W\aa_vec;
rho = rho_wavelet_HG(A,theta_vec);

% get the probabilities of each measurement outcome
[V,~] = eig(B_gamma);
p_outcomes = diag(V'*rho*V);

% Likelihood is a multinomial
p_l = mnpdf(l_vec, p_outcomes);

% handle numerically unstable cases
if sum(isnan(p_l))>0
    p_l = 0;
end
end

function p_a = prior(a_vec, priors, prior_doms)
% Calculates the probability of observing a_vec given
% a generalized probability density approximation for each parameter.
% This prior assumes the parameters are independent.
%
% INPUTS:
%   a_vec - An instance of the parameter vector with prior given by the
%           other inputs (priors, prior_doms)
%   priors - A matrix of dimensions [numel(a_vec),100]. The i'th row contains
%            points on the i'th parameter's prior distribution.
%   prior_doms - A matrix of dimensions [numel(a_vec),100]. Each row
%           contains the domain points associated with points on the prior
%           distribution.
% OUTPUT:

p_a_vec = zeros(size(a_vec));
for i = 1:numel(a_vec)
    p_a_vec(i) = interp1(prior_doms(i,:),priors(i,:),a_vec(i));
end

% handle numerically unstable cases
if sum(isnan(p_a_vec))>0
    p_a = 0;
else
    p_a = prod(p_a_vec);
end
end


function p_a = GBM_prior(a_vec,mu,z,q)
% Calculates the probabbility of observing a_vec given the GBM prior
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% a_vec:  the [N-1 x 1] parameter vector 
% mu:     the [N-1 x 2] mean vector containing the means of each normal distribution 
% in the GBM prior. [mu_1,0;mu_2,0; ...]. In general, this generates a bimodal distribution for each parameter as the
% prior updates in each Bayesian iteration. 
% z:      the [N-1 x 2] variance vector contianing the two variances of each
% normal distribution in the GBM prior. [zmax_1,zmin_1;z_max_2,zmin_2; ...]In general, the variances reduce
% every iteration of the Bayesian update to the prior.
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% p_a: the prior probability for the parameter values.
    [~,max_dim_idx] = max(size(a_vec));    
    if max_dim_idx == 2
        a_vec = a_vec';
    end
    
    p_a_vec = zeros(size(a_vec));   % probability for each independent GBM random variable
    
    % binary indices to the dirac delta cases
    dd1 = z(:,1)==0;
    dd2 = z(:,2)==0;
    
    % get the priors for each independendent GBM random variable
    p_a_vec(dd1) = q*(a_vec(dd1) == mu(dd1,1));
    p_a_vec(~dd1) = q*normpdf(a_vec(~dd1), mu(~dd1,1),z(~dd1,1));
    p_a_vec(dd2) = p_a_vec(dd2) + (1-q)*(a_vec(dd2)== mu(dd2,2));
    p_a_vec(~dd2) = p_a_vec(~dd2) + (1-q)*normpdf(a_vec(~dd2),mu(~dd2,2),z(~dd2,2));
    
    % joint probability of all the independent random variables
    p_a = prod(p_a_vec);
    
    % handle numerically unstable cases
    if sum(isnan(p_a))>0
        p_a = 0;
    end
    
end
