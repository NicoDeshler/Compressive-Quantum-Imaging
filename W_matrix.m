function W = W_matrix(n_thetas, ff_vec)
% Computes the matrix for transforming the unconstrained parameter vector 
% a_vec into the wavelet coefficient vector theta_vec. The matrix is
% defined such that it enforces the hyperplane constraint imposed by the 

% Implementation of sparsity-preserving W
N = n_thetas;
W = eye(N);

% Find where the non-zero wavelet integrals are
% These are the wavelet functions that do not have high-pass-filter block
k = find(ff_vec);

% Full-depth wavelet decomposition case
if numel(k) == 1
    W(:,end) = circshift(ff_vec,-k);
    W = circshift(W,k,1);

% Partial-depth wavelet decomposition case
else
    W(k(end),k(end)) = 0;
    W(sub2ind(size(W),k(2:end),k(1:end-1))) = -ff_vec(k(1:end-1)).*diag(W(k(1:end-1),k(1:end-1)))./ff_vec(k(2:end));
    W(end,k(end)) = 1;
    W(:,end)= ff_vec;
end

% Check W matrix conditions
zero_tol = 1e-15;       % tolerance for numerical stability
assert(isequal(W(:,1:end-1)'*ff_vec < zero_tol,ones(N-1,1))) % cond 1: columns 1:N-1 of W are orthogonal to ff_vec
assert(det(W) ~= 0)                                          % cond 2: W is invertible
end

