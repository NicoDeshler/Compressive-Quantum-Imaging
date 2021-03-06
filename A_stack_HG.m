function  A_stack = A_stack_HG(img_dims,n_modes,n_thetas,WaveletName,WaveletLevel)
% Computes the wavelet operators in the HG representation for a given
% wavelet transform. These operators are related to a wavelet transform of
% the system PSF, which is assumed to be a non-skew 2D gaussian with
% width sigma_x = sigma_y.
% ----------------------------------------------------------------
% INPUTS:
% ----------------------------------------------------------------
% image_dims    - Dimensions of the scene we wish to reconstruct (in pixels)
% n_thetas      - Number of parameters of the scene we wish to estimate
% n_modes       - Maximum order of 1D Hermite-Gauss modes used for the matrix
%                 representation. The total number of 2D modes used is thus n(n+1)/2
% WaveletName   - The wavelet type (e.g. 'sym8','db1', etc)
% WaveletLevel  - The maximum wavelet transform depth
% ----------------------------------------------------------------
% OUTPUTS:
% ----------------------------------------------------------------
% A_stack - a data cube of dimensions [N_mode,N_mode,n_thetas]

% Rayleigh length
rl = 1;

% Gaussian OTF widths
sigma = rl;
sigma_x = sigma;    
sigma_y = sigma;

% Discretize object plane coordinates system
x=linspace(-rl/2,rl/2,img_dims(1)); y=linspace(-rl/2,rl/2,img_dims(2));
[X,Y] = meshgrid(x,y);

N_modes = n_modes*(n_modes+1)/2;
A_stack = zeros(N_modes,N_modes,n_thetas); 
count = 1;

% build up index
for i = 1:n_modes
    for j = 1:i

        HG_proj(count).ind_x = i-j;
        HG_proj(count).ind_y = i-HG_proj(count).ind_x-1;  
        
        count = count + 1;

    end
end



for i = 1:N_modes
    for j = 1:N_modes
        p = HG_proj(i).ind_x;
        q = HG_proj(i).ind_y;
        m = HG_proj(j).ind_x;
        n = HG_proj(j).ind_y;
        
        % g(X,Y) = <HG_pq|Psi(X,Y)><Psi(X,Y)|HG_mn>
        XX = (X/2/sigma_x);
        YY = (Y/2/sigma_y);
        
        g = XX.^(p+m) .* YY.^(q+n) .* ...
            exp(- 1/2 * (XX.^2 + YY.^2)) * ...
            1/sqrt(factorial(p)*factorial(q) ...
            *factorial(m)*factorial(n));
        
       % wavelet transform g(X,Y)       
       [w,~] = wavedec2(g, WaveletLevel, WaveletName);
       
       % assign wavelet coefficients from each wavelet to matrix element in
       % stack
       A_stack(i,j,:) = w;        
    end
end

end