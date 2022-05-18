% INITIALIZATION

%load('db1_4sparse_4x4_img.mat');       % load in image
load('db1_2sparse_2x2_img.mat');       

% Image variables
img_dims = size(img);                  % image dimension vector [y pixels, x pixels]
img = img/sum(img,'all');             % normalized scene intensity distribution 

% Imaging system (Gaussian PSF)
sigma_x = 1;    % PSF width along x
sigma_y = 1;    % PSF width along y

% HG mode state space representation
n_HG_modes = 8;                        % max number of 1D Hermite-Gauss modes to consider
N_HG_modes = n_HG_modes*(n_HG_modes+1)/2;% total number of Hermite-Gauss modes considered

% Wavelet decomposition
WaveletName = 'db1';                   % wavelet type
WaveletLevel = 1;                      % wavelet decomposition level (complete decomp)
[gt_theta_vec, wv_idx] = wavedec2(img,WaveletLevel,WaveletName);                 % ground truth wavelet coefficients
gt_theta_vec = gt_theta_vec';
n_thetas = numel(gt_theta_vec);        % number of wavelet coefficients
n_as = n_thetas-1;                     % number of unconstrained parameters 

% Wavelet integrals
[f_vec,~] = wavedec2(ones(img_dims),WaveletLevel,WaveletName);              % wavelet integrals for normalization constraint
f_vec = f_vec';
ff_vec = f_vec/(f_vec'*f_vec);                                              % wavelet integrals rescaled for W matrix

% W matrix for transforming a_vec into theta_vec
W = W_matrix(n_thetas, ff_vec);

% ground truth unconstrained parameter vector
gt_aa_vec = W\gt_theta_vec;
gt_a_vec = gt_aa_vec(1:end-1);

% wavelet operators (pairs with theta_vec)
A_i = A_stack_HG(img_dims,n_HG_modes,n_thetas,WaveletName,WaveletLevel);

% normalize wavelet operators so that [tr(A1), tr(A2),...,tr(AN)] = f_vec
for i = 1:n_thetas
    if f_vec(i) ~= 0
       A_i(:,:,i) = A_i(:,:,i)/trace(A_i(:,:,i)) * f_vec(i);
    end
end



% Transform the wavelet operator stack A with W to define a new linear 
% combination of operators. We treat the operator stack as a 'vector' of 
% operators such that C = [C_1,C_2,...,C_N]^T = W^T [A_1, A_2, ... , A_N]^T

% transformed wavelet operators (pairs with aa_vec)
C_i = MatMulVecOp(W',A_i);

% photon collection variables
N_iter = 100000;                     % number of photons collected per bayesian update iteration
max_iter = 500;                      % number of Bayesian updates to perform

% Metropolis-Hastings parameters
n_MCMC = 10000;                   % number of MCMC samples of the posterior distribution
n_burn = 3000;                     % number of MCMC burnin samples (number of samples discarded while Markov Chain converges to stationary distribution)

% GBM prior parameters
q = 1/4;                                       % fractional sparsity  K/N = (# non-zero params/ # params)
z_min = 1;                                      % min variance
z_max = 10;                                     % max variance
mu = zeros([n_thetas-1,2]);                     % means for gaussian mixture random variables
z = [z_min*ones([n_as,1]),z_max*ones([n_as,1])];    % variances for gaussian mixture randomv variables

% Generate samples of GBM prior for creating probability density estimate
x0 = mvnrnd(mu(:,1),diag(z(:,1)),n_MCMC)';
x1 = mvnrnd(mu(:,2),diag(z(:,2)),n_MCMC)';
coinflips = binornd(1,q,[n_as,n_MCMC]);
GBM_samples = ~coinflips.*x0 + coinflips.*x1;

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

% initialize the wavelet vector estimator
theta_vec = W*aa_vec;                           % wavelet vector estimator
                     
% initialize measurement matrix as the identity - measure directly the HG
% modes
B_gamma = eye(N_HG_modes);
assert(ishermitian(B_gamma));

%% Display Initialization Figures

% display input image
figure(111)
imagesc(img)
title('Input Image')
xticks((1:img_dims(1))-.5)
yticks((1:img_dims(2))-.5)
xticklabels({})
yticklabels({})
grid on
axis('square')

% display the W matrix
figure(112)
imagesc(W)
title('Parameter Transform Matrix')
caxis([min(W(:)),max(W(:))])
yticks([])
xticks(1:5)
xticklabels({'w_1','w_2','w_3','f'})
axis 'square'
title('W matrix')
axis('square')
axis('off')

% display ground truth wavelet coefficients
figure(113)
stem(gt_theta_vec,'filled')
xlim([-1,n_thetas+1])
xlabel('Index $i$','interpreter','latex')
ylabel('Wavelet Coeff $\theta_i$','interpreter','latex')
title('Ground-Truth $\vec{\theta}$','interpreter','latex')

% display ground truth unconstrained coefficients
figure(114)
stem(gt_a_vec,'filled')
xlim([-1,n_as+1])
xlabel('Index $i$','interpreter','latex')
ylabel('Unconstrained Coeff $a_i$','interpreter','latex')
title('Ground-Truth $\vec{a}$','interpreter','latex')


%{
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
%}

%% Make video objects
make_videos = 0;

if make_videos

    vid_a = VideoWriter('UnconstrainedParams.avi');
    vid_a.FrameRate = 1;
    open(vid_a)
    fig_a = figure(101);

    vid_post_a1 = VideoWriter('a1_posterior.avi');
    vid_post_a1.FrameRate = 1;
    open(vid_post_a1)
    fig_post_a1 = figure(102);

    vid_post_a2 = VideoWriter('a2_posterior.avi');
    vid_post_a2.FrameRate = 1;
    open(vid_post_a2)
    fig_post_a2 = figure(103);

    vid_post_a3 = VideoWriter('a3_posterior.avi');
    vid_post_a3.FrameRate = 1;
    open(vid_post_a3)
    fig_post_a3 = figure(104);
end


%% Run Adaptive Bayesian Inference Algorithm 
% make coefficient convergence stack
a_evo = zeros(n_as, max_iter);

% Bayesian Update
priors = GBM_priors;
prior_doms = GBM_prior_doms;

N_collected = 0;     % number of photons collected
iter = 1;
while iter <= max_iter
     
    % simulate a measurement
    l_vec = SimulateMeasurement(B_gamma, N_iter, A_i, gt_theta_vec);
    
    % estimate the wavelet coefficients
    MCMC_start = mvnrnd(zeros(size(a_vec)), eye(n_as));
    [a_vec, posteriors, posterior_doms] = BayesianUpdate(l_vec,B_gamma,C_i,...
                                          priors, prior_doms, n_MCMC, n_burn, MCMC_start);
    aa_vec = [a_vec ; 1];

    % compute the variances and the means from the posteriors
    a_mu = sum(posteriors.*posterior_doms,2);
    a_var = sum(posteriors.*((posterior_doms-a_mu).^2),2);
    aa_mu = [a_mu; 1];
    aa_var = [a_var; 0];
    
    % set the posteriors of current iteration as the priors for the next
    % iteration
    priors = posteriors;
    prior_doms = posterior_doms;
    
    % compute Gamma_0 and Gamma_i1
    Gamma_0 = Gamma_0_HG(C_i,aa_mu);
    Gamma_i1 = Gamma_i1_HG(C_i,aa_mu,aa_var);
    
    % compute the optimal parameter estimators {B_i} with the implicit SLD equation.
    B_i = SLD_eval(Gamma_i1,Gamma_0);
    
    % compute the SLD eigenprojection vector
    h = h_proj(Gamma_0, B_i, aa_mu, aa_var);
    
    % calculate Gamma_1
    Gamma_1 = Gamma_1_HG(C_i,h,aa_mu,aa_var);
    
    % update the joint parameter estimator (measurement matrix) 
    B_gamma = SLD_eval(Gamma_1,Gamma_0);
        
    % update the number of photons
    N_collected = N_collected + N_iter;
    
    % update theta evolution stack (each time step is a new column)
    a_evo(:,iter) = a_vec;
    
    % update iteration index
    iter = iter + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Write figures to video objects
    if make_videos
        figure(fig_a)
        stem(a_vec,'filled')
        xlim([-1,n_as+1])
        xlabel('Index $i$','interpreter','latex')
        ylabel('Unconstrained Coeff $a_i$','interpreter','latex')
        title('Estimate $\hat{a}$','interpreter','latex')
        frame = getframe(fig_a);
        writeVideo(vid_a,frame)

        figure(fig_post_a1)
        plot(posterior_doms(1,:),posteriors(1,:))
        xlabel('$a_1$','interpreter','latex')
        ylabel('$P(a_1|\vec{l})$','interpreter','latex')
        title('Posterior $a_1$','interpreter','latex')
        frame = getframe(fig_post_a1);
        writeVideo(vid_post_a1,frame)

        figure(fig_post_a2)
        plot(posterior_doms(2,:),posteriors(2,:))
        xlabel('$a_2$','interpreter','latex')
        ylabel('$P(a_2|\vec{l})$','interpreter','latex')
        title('Posterior $a_2$','interpreter','latex')
        frame = getframe(fig_post_a2);
        writeVideo(vid_post_a2,frame)

        figure(fig_post_a3)
        plot(posterior_doms(3,:),posteriors(3,:))
        xlabel('$a_3$','interpreter','latex')
        ylabel('$P(a_3|\vec{l})$','interpreter','latex')
        title('Posterior $a_3$','interpreter','latex')
        frame = getframe(fig_post_a3);
        writeVideo(vid_post_a3,frame)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% close video objects
if make_videos
    close(vid_a)
    close(vid_post_a1)
    close(vid_post_a2)
    close(vid_post_a3)
end

% Compare output to ground truth wavelet coefficients
diff = gt_theta_vec - theta_vec;

% image estimate
img_out = waverec2(theta_vec,wv_idx,WaveletName);

figure(115);
imagesc(img_out);
title('Image Estimate')
xticks((1:img_dims(1))-.5);
yticks((1:img_dims(2))-.5);
xticklabels({})
yticklabels({})
grid on
axis 'square'

figure(116)
imagesc(a_evo)
title('Parameter Convergence')
xlabel('Iteration')
ylabel('$a_i$','interpreter','latex')
xticks(5:5:max_iter)
yticks(1:n_thetas)
colorbar




