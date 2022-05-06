% INITIALIZATION

%load('db1_4sparse_4x4_img.mat');       % load in image
load('db1_2sparse_2x2_img.mat');       

% Image variables
img_dims = size(img);                  % image dimension vector [y pixels, x pixels]
img = img/sum(img,'all');             % normalized scene intensity distribution 

% HG mode state space representation
n_HG_modes = 4;                        % max number of 1D Hermite-Gauss modes to consider
N_HG_modes = n_HG_modes*(n_HG_modes+1)/2;% total number of Hermite-Gauss modes considered

% Wavelet decomposition
WaveletName = 'db1';                   % wavelet type
WaveletLevel = 1;                      % wavelet decomposition level (complete decomp)
[gt_theta_vec, wv_idx] = wavedec2(img,WaveletLevel,WaveletName);                 % ground truth wavelet coefficients
n_thetas = numel(gt_theta_vec);        % number of wavelet coefficients
n_as = n_thetas-1;                     % number of unconstrained parameters 

% Wavelet integrals
[f_vec,~] = wavedec2(ones(img_dims),WaveletLevel,WaveletName);              % wavelet integrals for normalization constraint
f_vec = f_vec';
ff_vec = f_vec/(f_vec'*f_vec);                                              % wavelet integrals rescaled for W matrix

% W matrix for transforming a_vec into theta_vec
W = W_matrix(n_thetas, ff_vec);

% normalize ground truth parameters to make scene intensity a distribution
% (density operator trace 1)
gt_theta_vec = gt_theta_vec';
gt_theta_vec = gt_theta_vec/(gt_theta_vec'*f_vec);

% ground truth unconstrained parameter vector
gt_aa_vec = W\gt_theta_vec;

% wavelet operators (pairs with theta_vec)
A_i = A_stack_HG(img_dims,n_HG_modes,n_thetas,WaveletName,WaveletLevel);

% transformed wavelet operators (pairs with aa_vec)
C_i = squeeze(sum(reshape(A_i,[size(A_i),1]).*reshape(W,[1,1,size(W)]),3));

% photon collection variables
N_iter = 10000;                     % number of photons collected per bayesian update iteration
max_iter = 500;                     % number of Bayesian updates to perform

% Metropolis-Hastings parameters
n_MCMC = 10000;                   % number of MCMC samples of the posterior distribution
n_burn = 1000;                     % number of MCMC burnin samples (number of samples discarded while Markov Chain converges to stationary distribution)

% GBM prior parameters
q = .1;                                         % fractional sparsity  (# zero-valued params/# params)
z_min = 0.001;                                  % min variance
z_max = 1;                                      % max variance
mu = zeros([n_thetas-1,2]);                     % means for gaussian mixture random variables
z = [z_min*ones([n_as,1]),z_max*ones([n_as,1])];    % variances for gaussian mixture randomv variables

% Generate samples of GBM prior for creating probability density estimate
x1 = mvnrnd(mu(:,1),diag(z(:,1)),n_MCMC)';
x2 = mvnrnd(mu(:,2),diag(z(:,2)),n_MCMC)';
coinflips = binornd(1,q,[n_as,n_MCMC]);
GBM_samples = coinflips.*x1 + ~coinflips.*x2;

GBM_priors = zeros(n_as,100);
GBM_prior_doms = zeros(n_as,100);
a_vec = zeros([n_as,1]);
for i = 1:n_as
    % generate probability density estimate
    [GBM_priors(i,:), GBM_prior_doms(i,:)] = ksdensity(GBM_samples(i,:)); 
    
    % sample the unconstrained parameter vector from the initial prior
    a_vec(i) = datasample(GBM_prior_doms(i,:), 1,'weights',GBM_priors(i,:));
end
aa_vec = [a_vec; 1];                           % augmented unconstrained parameter vector


%{
% Sample the unconstrained parameter vector from the prior
a_vec(i) = datasample(GBM_prior_doms(:,i), N_photons,'weights',GBM_priors(:,i))';
a_vec = coinflips.*x1 + ~coinflips.*x2;        % unconstrained parameter vector
aa_vec = [a_vec; 1];                           % augmented unconstrained parameter vector
%}   

% initialize the wavelet vector estimator
theta_vec = W*aa_vec;                           % wavelet vector estimator
                     
% initialize measurement matrix as the identity - measure directly the HG
% modes
B_gamma = eye(N_HG_modes);
assert(ishermitian(B_gamma));

%% Display Initialization Figures

% Display input image
imagesc(img);
title('Input Image')
xticks((1:img_dims(1))-.5);
yticks((1:img_dims(2))-.5);
xticklabels({})
yticklabels({})
grid on
axis('square')

% display ground truth wavelet coefficients
figure;
stem(gt_theta_vec)
xlabel('Wavelet Linear Index')
ylabel('Coefficient Value')
title([WaveletName,' Decomposition'])

% plot the wavelet decomposition tree
figure;
plotwavelet2(gt_theta_vec, wv_idx, WaveletLevel, WaveletName,16,'square')
title('Db1 Wavelet Decomposition Tree')
xticks((1:img_dims(1))-.5);
yticks((1:img_dims(2))-.5);
xticklabels({})
yticklabels({})
grid on
axis('square')

%% Run Adaptive Bayesian Inference Algorithm 
% make video
v = VideoWriter('UnconstrainedCoeffs.avi');
v.FrameRate = 1;
open(v)
figure

% make coefficient convergence stack
theta_evo = zeros(n_thetas, max_iter);

% Bayesian Update
priors = GBM_priors;
prior_doms = GBM_prior_doms;


N_collected = 0;                                % number of photons collected
iter = 1;
while iter <= max_iter
    % simulate a measurement
    l_vec = SimulateMeasurement(B_gamma, N_iter, A_i, gt_theta_vec);
    
    % estimate the wavelet coefficients
    MCMC_start = mvnrnd(zeros(size(a_vec)), eye(n_as));
    [a_vec, posteriors, posterior_doms] = BayesianUpdate(l_vec,B_gamma, W,...
                                          A_i, priors, prior_doms, n_MCMC, n_burn, MCMC_start);
    aa_vec = [a_vec ; 1];
    theta_vec = W*aa_vec;

    % compute the variances and the means from the posteriors
    a_mu = sum(posteriors.*posterior_doms,2);
    a_var = sum(posteriors.*((posterior_doms-a_mu).^2),2);
    aa_mu = [a_mu; 1];
    aa_var = [a_var; 0];
    
    % set the posteriors of current iteration as the priors for the next
    % iteration
    priors = posteriors;
    prior_doms = posterior_doms;
    

    %{
    % get mean and variance of the augmented parameter vector
    a_mu = mu*[q;(1-q)];                          % expected value of a_vec under GBM prior
    aa_mu = [a_mu; 1];
    a_var = (z + mu.^2)*[q;(1-q)] - a_mu.^2;      % variance of a_vec under GBM prior  
    aa_var = [a_var; 0];
    %}
    
    % compute Gamma_0 and Gamma_i1
    Gamma_0 = Gamma_0_HG(A_i,W,aa_mu);
    Gamma_i1 = Gamma_i1_HG(A_i,W,aa_mu,aa_var);
    
    % compute the optimal parameter estimators {B_i} with the implicit SLD equation.
    B_i = SLD_eval(Gamma_i1,Gamma_0);
    
    % compute the SLD eigenprojection vector
    h = h_proj(Gamma_0, B_i, aa_mu, aa_var);
    
    % calculate Gamma_1
    Gamma_1 = Gamma_1_HG(A_i,W,h,aa_mu,aa_var);
    
    % update the joint parameter estimator (measurement matrix) 
    B_gamma = SLD_eval(Gamma_1,Gamma_0);
    assert(ishermitian(B_gamma));
        
    % update the number of photons
    N_collected = N_collected + N_iter;
    
    %{
    % display diagonalized measurement matrix (projections in HG modes)
    [U,D] = eig(U);
    imagesc(B_gamma)
    xlabel('Measurement Projectors (Columns)')
    ylabel('HG Mode')
    title(['$B_{\gamma}^{[',num2str(iter),']}$'],'interpreter','latex')
    %}
    
    %{
    stem(aa_vec,'filled')
    xlabel('Linear Index')
    ylabel('Coefficient Magnitude')
    title(['$\hat{a}^{[',num2str(iter),']}$'],'interpreter','latex')
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame)
    %}
  
    
    stem(theta_vec,'filled')
    xlabel('Wavelet Linear Index')
    ylabel('Coefficient Value')
    title(['$\hat{\theta}^{[',num2str(iter),']}$'],'interpreter','latex')
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame)
    
    
    % update theta evolution stack (each time step is a new column)
    theta_evo(:,iter) = theta_vec;
    
    % update iteration index
    iter = iter + 1;
end
close(v)

% Compare output to ground truth wavelet coefficients
diff = gt_theta_vec - theta_vec;


% image estimate
img_out = waverec2(theta_vec,wv_idx,WaveletName);

figure;
imagesc(img_out);
title('Image Estimate')
xticks((1:img_dims(1))-.5);
yticks((1:img_dims(2))-.5);
xticklabels({})
yticklabels({})
grid on


figure()
imagesc(theta_evo)
title('Parameter Convergence')
xlabel('iteration')
ylabel('$\theta_i$','interpreter','latex')
xticks(5:5:max_iter)
yticks(1:n_thetas)




