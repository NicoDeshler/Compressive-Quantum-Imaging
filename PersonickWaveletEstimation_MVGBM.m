function PersonickWaveletEstimation_MVGBM(array_id)

% INITIALIZATION

disp(['PBS_ARRAYID:',array_id]);
rng(str2double(array_id));

% Simulated image Preliminaries
img_dims = [8,8];
q = .1;                                % fractional sparsity  K/N = (# non-zero params/ # params)
WaveletName = 'db1';                   % wavelet type
WaveletLevel = log2(max(img_dims));    % wavelet decomposition level (full-depth decomposition)
% generate sparse image
[img,ws] = gen_wavelet_sparse_img(WaveletName,WaveletLevel,img_dims,q);


% Image variables
img_dims = size(img);                  % image dimension vector [y pixels, x pixels]
img = img/sum(img(:));              % normalized scene intensity distribution 

% Imaging system (Gaussian PSF)
rl = max(img_dims);                     % Rayleigh length (in pixels)
psf_width = rl;                         % Gaussian PSF width (in pixels)
img_direct = imgaussfilt(img,psf_width);     % what the image would look like after direct imaging

% Truncated Hilbert space dimensionality (Hermite-Gauss Representation) 
n_HG_modes = 12;                             % number of 1D Hermite-Gauss modes
N_HG_modes = n_HG_modes*(n_HG_modes+1)/2;   % total number of 2D Hermite-Gauss modes

% Wavelet decomposition
[gt_theta_vec, wv_idx] = wavedec2(img,WaveletLevel,WaveletName);                 % ground truth wavelet coefficients
gt_theta_vec = gt_theta_vec';
n_thetas = numel(gt_theta_vec);        % number of wavelet coefficients
n_as = n_thetas-1;                     % number of transformed parameters 

% Wavelet integrals
[f_vec,~] = wavedec2(ones(img_dims),WaveletLevel,WaveletName);              % wavelet integrals for normalization constraint
f_vec = f_vec';
ff_vec = f_vec/(f_vec'*f_vec);                                              % wavelet integrals rescaled for W matrix

% W matrix for transforming a_vec into theta_vec
W = W_matrix(n_thetas, ff_vec);

% ground truth transformed parameter vector
gt_aa_vec = W\gt_theta_vec;
gt_a_vec = gt_aa_vec(1:end-1);

% wavelet operators (pairs with theta_vec)
A_vec = A_stack_HG(img_dims,n_HG_modes,n_thetas,WaveletName,WaveletLevel);

%{
% normalize wavelet operators so that [tr(A1), tr(A2),...,tr(AN)] = f_vec
for i = 1:n_thetas
    if trace(A_vec(:,:,i)) ~= 0  && f_vec(i) ~= 0
       A_vec(:,:,i) = A_vec(:,:,i) ./ trace(A_vec(:,:,i)) .* f_vec(i);
    end
end
%}


% Transform the wavelet operator stack A with W to define a new linear 
% combination of operators. We treat the operator stack as a 'vector' of 
% operators such that C = [C_1,C_2,...,C_N]^T = W^T [A_1, A_2, ... , A_N]^T

% transformed wavelet operators (pairs with aa_vec)
C_vec = MatMulVecOp(W',A_vec);

% photon collection variables
N_pho_iter = 1e5;                  % number of photons collected per Bayesian update iteration

% sampling parameters
N_samples = 1e5;           % number of samples taken to approximate the posterior distribution



%% GBM PRIOR SETUP
% GBM prior parameters
z_min = 1e-3;                                     % min variance
z_max = 1e-1;                                     % max variance
a_mu0 = zeros(n_as,1);
a_mu1 = zeros(n_as,1);
a_cov0 = z_min*eye(n_as);
a_cov1 = z_max*eye(n_as);

% compute mean and covariance matrix of GBM prior
a_mu_GBM = (1-q)*a_mu0 + q*a_mu1;
a_cov_GBM = (1-q)*(a_cov0 + a_mu0*a_mu0') + (q)*(a_cov1 + a_mu1*a_mu1')...
            - a_mu_GBM*a_mu_GBM';



% Generate samples of GBM prior
x0 = mvnrnd(a_mu0,a_cov0,N_samples)';
x1 = mvnrnd(a_mu1,a_cov1,N_samples)';
coinflips = binornd(1,q,[n_as,N_samples]);
GBM_samples = ~coinflips.*x0 + coinflips.*x1;

% initialize the starting parameter estimate to one of the GBM samples
a_vec = GBM_samples(:,randi(N_samples));


%% Initialize estimation parameters
% initialize an estimate of the transformed transformed parameters
% sample the transformed parameter vector from the initial prior
aa_vec = [a_vec; 1];

% initialize an estimate of the wavelet coefficients
theta_vec = W*aa_vec;
                     
% initialize the measurement matrix as the identity 
% (equivalent to a POVM comprised of projectors onto the HG modes)
B_gamma = eye(N_HG_modes);
assert(ishermitian(B_gamma) && trace(B_gamma)>0);

%% Make directory for saving results

%{
save_dir = fullfile('Testing',posterior_method,sampling_method,...
[num2str(N_pho_iter/1000),'Kpho'],...
[num2str(N_samples/1000),'Ksamps']);
if exist(save_dir,'dir')
    i = 1;
    save_dir_x = fullfile(save_dir,'ex1');
    while exist(save_dir_x, 'dir')
        exi = ['ex',num2str(i)];
        save_dir_x = fullfile(save_dir,exi);
        i = i+1;
    end
    save_dir = save_dir_x;
end
mkdir(save_dir)
%}
% save directory for cluster
save_dir = fullfile('Testing','sparse8x8_recon_mvgbm');
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
    

%% Figures
% set default interpreters to LaTex
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


%% Make video objects
make_figures = 0;

if make_figures
    
    vid_a = VideoWriter(fullfile(save_dir,'TransformedParams.avi'));
    vid_a.FrameRate = 3;
    open(vid_a)
    fig_a = figure;
    fig_a.WindowState = 'maximized';

    %{
    vid_posteriors = VideoWriter(fullfile(save_dir,'Posteriors.avi'));
    vid_posteriors.FrameRate = 3;
    open(vid_posteriors)
    fig_posteriors = figure;
    fig_posteriors.WindowState = 'maximized';
    %}
    
    vid_recon = VideoWriter(fullfile(save_dir,'ImageRecon.avi'));
    vid_recon.FrameRate =3;
    open(vid_recon)
    fig_recon = figure;
    fig_recon.WindowState = 'maximized';
    
end

%% Run Adaptive Bayesian Inference Algorithm 

max_iter = 1;      % number of Bayesian updates to perform

% array for plotting coefficient convergence
a_evo = zeros([n_as, max_iter]);
% array for plotting posterior convergence (via variance reduction)
a_var_evo = zeros([n_as, max_iter]); 
% array for plotting Euclidian distance between estimated wavelets and
% ground truth
theta_dist = zeros([1,max_iter]);


N_collected = 0;     % number of photons collected
iter = 1;
while iter <= max_iter
     
    % simulate a measurement
    l_vec = SimulateMeasurement(B_gamma, N_pho_iter, A_vec, gt_theta_vec);
    
    % sample the posteriors and estimate the wavelet coefficients
    [a_vec, a_mu_post, a_cov_post] = BayesianUpdate_MVGBMposterior(...
        l_vec, B_gamma, C_vec,...
        a_mu0, a_cov0, a_mu1, a_cov1, q,...
        N_samples,...
        W,wv_idx,WaveletName);   
    
    % update GBM parameters
    a_mu1 = a_mu_post;
    a_cov1 = a_cov_post;
    q = 2*q - q^2;
    
    
    
    % augment updated parameter variables
    aa_vec = [a_vec ; 1];
    aa_mu = [a_mu_post ; 1];
    aa_cov = padarray(a_cov_post,[1,1],'post');
        
    % update wavelet parameters
    theta_vec = W*aa_vec;
    
    % updated image estimate
    img_est = waverec2(theta_vec, wv_idx, WaveletName);
    
    % compute Gamma_0 and {Gamma_i1}'s
    Gamma_0 = Gamma_0_HG(C_vec,aa_mu);
    Gamma_i1 = Gamma_i1_HG(C_vec,aa_mu,aa_cov);
    
    % compute the optimal parameter estimators {B_i} with the implicit SLD equation.
    B_vec = SLD_eval(Gamma_i1,Gamma_0);
    
    % compute the joint-parameter projection
    E_Q = Sigma_Q(Gamma_0, B_vec,aa_mu,aa_cov);
    E_Q = E_Q(1:end-1,1:end-1); % crop out augmented parameter
    [V_Q,lam] = eig(E_Q,'vector');

    % choose min eigenvector
    [~, min_eigval_idx] = min(lam);
    h = [V_Q(:,min_eigval_idx(1));0]; % joint parameter vector
    
    % calculate Gamma_1
    Gamma_1 = Gamma_1_HG(h,Gamma_i1);
    
    % update the joint parameter estimator (measurement matrix) 
    B_gamma = SLD_eval(Gamma_1,Gamma_0);
    [V_gamma,~] = eig(B_gamma);
        
    % update the number of photons
    N_collected = N_collected + N_pho_iter;
    
    % update convergence containers
    a_evo(:,iter) = a_vec;
    a_var_evo(:,iter) = diag(a_cov_post);
    theta_dist(iter) = norm(gt_theta_vec - theta_vec);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Write figures to video objects
    if make_figures
        
        % TRANSFORMED PARAMETERS
        figure(fig_a)
        % plot ground truth parameters
        stem(1:n_as, gt_a_vec,'filled','black','MarkerSize',5)
        hold on
        stem(1:n_as,a_vec,'red','MarkerSize',4)
        hold off
        xticks(1:n_as)
        xlim([0,n_as+1])
        xlabel('$i$')
        ylabel('$a_i$')
        ylim([-gt_theta_vec(1),gt_theta_vec(1)])
        title('Transformed Parameters')
        legend({'Ground Truth $\vec{a}$',['Estimate $\hat{\vec{a}}^{[',num2str(iter),']}$']})
        frame = getframe(fig_a);
        writeVideo(vid_a,frame)
        
        %{
        if plot_posteriors
            % POSTERIORS
            figure(fig_posteriors)
            % subplot figure dimensions
            fd1 = ceil(sqrt(n_as));
            fd2 = fd1;
            for i = 1:n_as
                % P(a1|l)
                subplot(fd1,fd2,i)
                plot(posterior_doms(i,:),posteriors(i,:))
                xlabel(['$a_{',num2str(i),'}$'])
                ylabel(['$P(a_{',num2str(i),'}^{[',num2str(iter),']}|\vec{l})$'])
                ylim([0,1])
            end

            frame = getframe(fig_posteriors);
            writeVideo(vid_posteriors,frame)
        end
        %}
        
        % RECONSTRUCTED IMAGE
        figure(fig_recon);
        % plot the ground truth
        subplot(1,3,1)
        imagesc(img)
        title('Target')
        xticks((0:img_dims(1))+0.5)
        yticks((0:img_dims(2))+0.5)
        xticklabels((-img_dims(1)/2:img_dims(1)/2)/rl)
        yticklabels(flip(-img_dims(2)/2:img_dims(2)/2)/rl)
        xlabel('$X [\sigma]$')
        ylabel('$Y [\sigma]$')
        grid on
        axis('square')
        colorbar
        caxis([min(img(:)),max(img(:))])
        % plot the direct imaging scene
        subplot(1,3,2)
        imagesc(img_direct)
        title('Direct Imaging')
        xticks((0:img_dims(1))+0.5)
        yticks((0:img_dims(2))+0.5)
        xticklabels((-img_dims(1)/2:img_dims(1)/2)/rl)
        yticklabels(flip(-img_dims(2)/2:img_dims(2)/2)/rl)
        xlabel('$X [\sigma]$')
        ylabel('$Y [\sigma]$')
        grid on
        axis('square')
        colorbar
        caxis([min(img(:)),max(img(:))])
        % plot the reconstruction
        subplot(1,3,3)
        imagesc(img_est)
        title(['Estimate [iter: ',num2str(iter),']'])
        xticks((0:img_dims(1))+0.5)
        yticks((0:img_dims(2))+0.5)
        xticklabels((-img_dims(1)/2:img_dims(1)/2)/rl)
        yticklabels(flip(-img_dims(2)/2:img_dims(2)/2)/rl)
        xlabel('$X [\sigma]$')
        ylabel('$Y [\sigma]$')
        grid on
        axis('square')
        colorbar
        caxis([min(img(:)),max(img(:))])
        
        frame = getframe(fig_recon);
        writeVideo(vid_recon,frame)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % update iteration index
    iter = iter + 1;
end


save(fullfile(save_dir,['Config_Recon_Data_t',array_id,'.mat']),'a_evo','a_var_evo','theta_dist',...
    'WaveletName','WaveletLevel','wv_idx',...
    'img','W','gt_a_vec','img_est',...
    'n_HG_modes','N_pho_iter','max_iter',...
    'N_samples','sampling_method')


% close video objects
if make_figures
    close(vid_a)
    close(vid_posteriors)
    close(vid_recon)
end


if make_figures
    % Convergence Plots
    fig_convergence = figure;
    fig_convergence.WindowState = 'maximized';
    subplot(1,3,1)
    imagesc(a_evo)
    title('Parameter Convergence')
    xlabel('Iteration')
    ylabel('$a_i$')
    xticks(5:5:max_iter)
    yticks(1:n_as)
    axis('square')
    colorbar

    subplot(1,3,2)
    imagesc(a_var_evo)
    title('Variance Convergence')
    xlabel('Iteration')
    ylabel('$Var(a_i)$')
    xticks(5:5:max_iter)
    yticks(1:n_as)
    axis('square')
    colorbar

    subplot(1,3,3)
    plot(1:max_iter,theta_dist);
    title('Wavelet Error Convergence')
    xlabel('Iteration')
    ylabel('$||\mathbf{\theta}-\hat{\mathbf{\theta}}||_2$')
    xticks(5:5:max_iter)
    axis('square')

    % save figure
    saveas(fig_convergence,fullfile(save_dir,'ConvergencePlots.png'))
end
end
