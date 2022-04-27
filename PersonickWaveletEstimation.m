% INITIALIZATION

img = load('db1_sparse_4x4_img')

% HG mode state space representation
n_HG_modes = 4;                        % max number of 1D Hermite-Gauss modes to consider
N_HG_modes = n_HG_modes*(n_HG_modes+1)/2;% total number of Hermite-Gauss modes considered

% Image variables
img_y = 4;                            % vertical image dimension (pixels)
img_x = 4;                            % horizonal image dimension (pixels)
img_dims = [img_y,img_x];             % image dimension vector
img = abs(magic(img_dims));           % normalized scene intensity distribution
img = img/sum(img,'all');

% Wavelet variables
WaveletName = 'db1';                   % wavelet type
WaveletLevel = 2;                      % wavelet decomposition level (complete decomp)
[gt_theta_vec, wv_idx] = wavedec2(img,WaveletLevel,WaveletName);                 % ground truth wavelet coefficients
n_thetas = numel(gt_theta_vec);                                            % number of wavelet coefficients

% Wavelet integrals
[f_vec,~] = wavedec2(ones(img_dims),WaveletLevel,WaveletName);              % wavelet integrals for normalization constraint
f_vec = f_vec';
ff_vec = f_vec/(f_vec'*f_vec);                                              % wavelet integrals rescaled for W matrix

%{
% Sparsify wavelet decomposition
K = 4;                                   % k-sparsity constant (number of non-zero coeffs)
ws = gt_theta_vec;                       % sparsified wavelet coefficient vector
[w_sorted,sort_idx] = sort(abs(ws),'descend');
ws(sort_idx(K+1:end)) = 0;
img_sparse = waverec2(ws,wv_idx,WaveletName);

gt_theta_vec = ws;

%}

% Display the wavelet-sparse image
figure;
imagesc(img)
title('Wavelet-Sparse Input Image')

% Plot the Wavelet Decomposition Tree
figure;
plotwavelet2(gt_theta_vec, wv_idx, WaveletLevel,WaveletName,128,'square')
title('Db1 Wavelet Decomposition Tree')

%%
% normalize ground truth parameters to make density operator trace 1
gt_theta_vec = gt_theta_vec';
gt_theta_vec = gt_theta_vec/(gt_theta_vec'*f_vec);

% W matrix for transforming a_vec into theta_vec
W = W_matrix(n_thetas, ff_vec);

% ground truth constrained parameter vector
gt_aa_vec = W\gt_theta_vec;

% measurement variables
N_iter = 10000;                     % number of photons collected per bayesian update iteration
max_iter = 50;                      % number of Bayesian updates to perform
% Metropolis-Hastings parameters
n_MCMC = 1000;                   % number of MCMC samples of the posterior distribution
n_burn = 500;                     % number of MCMC burnin samples (number of samples discarded while Markov Chain converges to stationary distribution)

% wavelet operators
A_i = A_stack_HG(img_dims,n_HG_modes,n_thetas,WaveletName,WaveletLevel);


% initialize GBM prior parameters
q = 1-.125;                               % fractional sparsity  (# zero-valued params/# params)
z_min = 0;                                      % min variance
z_max = 1;                                      % max variance
mu = zeros([n_thetas-1,2]);                     % means for gaussian mixture random variables
z = [z_min*ones([n_thetas-1,1]),z_max*ones([n_thetas-1,1])];    % variances for gaussian mixture randomv variables


%  sample the unconstrained parameter vector from the GBM prior
x1 = normrnd(mu(:,1),z(:,1));
x2 = normrnd(mu(:,2),z(:,2));
coinflips = binornd(1,q,[numel(x1),1]);
a_vec = coinflips.*x1 + ~coinflips.*x2;        % unconstrained parameter vector
aa_vec = [a_vec; 1];                           % augmented unconstrained parameter vector
    
% initialize the wavelet vector
theta_vec = W*aa_vec;                           % wavelet vector
                     
% initialize measurement matrix
B_gamma = randn([N_HG_modes,N_HG_modes]);       % measurement matrix
B_gamma = B_gamma'*B_gamma;                     % matrix must be hermitian

% make video
v = VideoWriter('UnconstrainedCoeffs.avi');
v.FrameRate = 1;
open(v)
figure

% make coefficient convergence stack
theta_evo = zeros(n_thetas, max_iter);

% Bayesian Update
N_collected = 0;                                % number of photons collected
iter = 1;
while iter <= max_iter
    % simulate a measurement
    l_vec = SimulateMeasurement(B_gamma, N_iter, A_i, gt_theta_vec);
    
    % estimate the wavelet coefficients
    [a_vec,mu,z] = BayesianUpdate(l_vec,B_gamma, W, A_i, mu, z, q, n_MCMC, n_burn);
    aa_vec = [a_vec ; 1];
    theta_vec = W*aa_vec;
    
    % get mean and variance of the augmented parameter vector
    a_mu = mu*[q;(1-q)];                          % expected value of a_vec under GBM prior
    aa_mu = [a_mu; 1];
    a_var = (z + mu.^2)*[q;(1-q)] - a_mu.^2;      % variance of a_vec under GBM prior  
    aa_var = [a_var; 0];
    
    % compute Gamma_0 and Gamma_i1
    Gamma_0 = Gamma_0_HG(A_i,W,aa_mu);
    Gamma_i1 = Gamma_i1_HG(A_i,W,aa_mu,aa_var);
    
    % compute the optimal parameter estimators {B_i} with the implicit SLD equation.
    B_i = SLD_eval(Gamma_i1,Gamma_0);
    
    % compute the SLD eigenprojection vector
    h = h_proj(Gamma_0, B_i);
    
    % calculate Gamma_1
    Gamma_1 = Gamma_1_HG(A_i,W,h,aa_mu,aa_var);
    
    % update the joint parameter estimator (measurement matrix) 
    B_gamma = SLD_eval(Gamma_1,Gamma_0);
    
    [U,D] = eig(B_gamma);
    
    % update the number of photons
    N_collected = N_collected + N_iter;
    
    %{
    % display measurement matrix
    imagesc(B_gamma)
    xlabel('HG_i')
    ylabel('HG_j')
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
    ylabel('Wavelet Coefficient Magnitude')
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






