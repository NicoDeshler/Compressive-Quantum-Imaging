function  A_stack = A_stack_HG(img_dims,n_modes,n_thetas,WaveletName,WaveletLevel)
% Computes the 'A' operators in the HG representation for a given
% wavelet transform. These operators are the OTF projectors after undergoing 
% a wavelet transform. The OTF of the system is assumed to 
% be a gaussian PSF of width sigma_x = sigma_y.
%
% --------
% Inputs:
% --------
% image_dims    - Dimensions of the scene we wish to reconstruct (in pixels)
% n_thetas      - Number of parameters of the scene we wish to estimate
% n_modes       - Maximum order of 1D Hermite-Gauss modes used for the matrix
%                 representation. The total number of 2D modes used is thus n(n+1)/2
% WaveletName   - The wavelet type (e.g. 'sym8','db1', etc)
% WaveletLevel  - The maximum wavelet transform depth

% --------
% Outputs:
% --------
% A_stack - a data cube of dimensions (N_mode,N_mode,n_thetas)

% Rayleigh length
rl = 1;

% Gaussian OTF widths
sigma_x = rl;    
sigma_y = rl;

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
        
        % f = <HG_pq|Psi(x,y)><Psi(x,y)|HG_mn>
        g = (X/2/sigma_x).^(p+m)*(Y/2/sigma_y).^(q+n)*exp(-((X/2/sigma_x).^2+(Y/2/sigma_y).^2)) * ...
            1/sqrt(factorial(p)*factorial(q) ...
            *factorial(m)*factorial(n));
        
       % wavelet transform f                        
       [C1,~] = wavedec2(g, WaveletLevel, WaveletName);
       A_stack(i,j,:) = C1;        
    end
end

%A_i(:,:,1) = A_i(:,:,1)/trace(A_i(:,:,1)) * f_vec(1)/(8*2*pi*sigma_x*sigma_y);

end