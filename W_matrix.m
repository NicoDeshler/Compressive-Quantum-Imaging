function W = W_matrix(n_thetas, ff) 
% Computes the matrix for transforming the unconstrained parameter vector 
% a_vec into the wavelet coefficient vector theta_vec. The matrix is
% defined such that it enforces the hyperplane constraint imposed by the 
% trace-1 property of the density oeprator. 

N = n_thetas;       % Number of parameters
idx = 1:N;          % Index list
S0 = idx(ff==0);    % Index list of 0-valued f' elements
S1 = idx(ff~=0);    % Index list of non-0-valued f' elements
W = zeros([N,N]);       % Instantiate the W matrix
W(:,N) = ff;        % Set the last column to f'

% Case 1: Only one non-0-valued element in f'
if numel(S0) == 1
    % index of non-zero element
    j = S0(1);                
    % circular shift non-zero element to W(N,N)
    W = circshift(W,-j,1);   
    % make remaining columns identity
    W(1:N-1,1:N-1) = eye(N-1);
    % undo circular shift to preserve ordering
    W = circshift(W,j,1);      

% Case 2: More than one non-0-valued element in f'
else

    % W matrix columns for non-0-valued f' elements
    for k = 1:numel(S1)-1
        W(S1(k),k) = ff(S1(k+1));
        W(S1(k+1),k) = -ff(S1(k));
    end

    % W matrix columns for 0-valued f' elements
    for i = 1:numel(S0)
        j = numel(S1) + i - 1;
        W(S0(i),j) = 1;
    end
end

% Check W matrix conditions
zero_tol = 1e-15;       % tolerance for numerical stability
assert(isequal(W(:,1:end-1)'*ff < zero_tol,ones(N-1,1))) % cond 1: columns 1:N-1 of W are orthogonal to ff
assert(det(W) ~= 0)                                      % cond 2: W is invertible

end