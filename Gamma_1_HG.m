function Gamma_1 = Gamma_1_HG(A,W,h,aa_mu,aa_var)
% Computes the 1th-index Personick Operator represented in the
% Hermite-Gauss basis for a scene described via a wavelet transform.
% This operator is used for solving the equation:
%           2 x Gamma_1 = B Gamma_0 + Gamma_0 B 
% where B is the Quantum Minimum Mean-Squared Estimator (QMMSE).
%
%
% --------
% Inputs:
% --------
% A - A stack of matrices containing the OTF projectors after a wavelet
%     transform represented in the HG basis.
% W - the transformation matrix that takes a_vec to theta_vec where a_vec
%     contains a list of unconstrained (yet lower dimensional) parameters
%     and theta_vec contains (constrained) wavelet coefficients
% h - a unit vector for computing the joint parameter dot(h,a_vec)
% aa_mu - a vector containing the expected values of the 'a' parameters
% aa_var - a vector containing the variances of the 'a' parameters
%
% --------
% Outputs:
% --------
% Gamma_1 - A matrix with dimensions equal to size(A(:,:,1))

% get the matrix E[a_i a_j] of expected values for pair-wise parameter 
% products
M = diag(aa_var) + aa_mu*aa_mu';

% Transform the wavelet operator stack A with W to define a new linear 
% combination of operators. We treat the operator stack as a 'vector' of 
% operators such that C = [C_1,C_2,...,C_N]^T = W [A_1, A_2, ... , A_N]^T;
C = squeeze(sum(reshape(A,[size(A),1]).*reshape(W,[1,1,size(W)]),3));

% compute Gamma_1
Gamma_1 = zeros(size(A(:,:,1)));
for i = 1:size(M,1)
    for j = 1:size(M,2)
        Gamma_1 = Gamma_1 + C(:,:,i)*h(j)*M(i,j);
    end
end

%{
% fast method (scales poorly)
M = diag(aa_var) + aa_mu*aa_mu';   % E[ai aj] matrix
M = reshape(M(:),[1,1,numel(M)]);  

H = repmat(h',[numel(h),1]);
H = reshape(H(:),[1,1,numel(H)]);

C = squeeze(sum(reshape(A,[size(A),1]).*reshape(W,[1,1,shape(W)]),3));
C = repmat(C,[1,1,numel(h)]);
Gamma_1 = sum(C.*H.*M,3);
%}
end
