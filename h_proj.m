function h = h_proj(Gamma_0,B_stack)
% Compute joint parameter projection vector h

% Inputs:
% B_stack : A matrix stack of the optimal parameter estimators in HG
% representation

% Outputs:
% h : The joint parameter projection vector

g = size(B_stack,3);
G = zeros([g,g]);

for i = 1:g
    for j = 1:g
    G(i,j) = 1/2 * trace(Gamma_0*(B_stack(:,:,i)*B_stack(:,:,j)+B_stack(:,:,j)*B_stack(:,:,i))/2);
    end
end

% Get eigenvector corresponding to minimum eigenvalue of the G matrix
[V,lam] = eig(G);
%[~, min_eigval_idx] = min(lam);
[~, max_eigval_idx] = max(lam);
%h = V(:,min_eigval_idx); % joint parameter vector
h = V(:,max_eigval_idx); % joint parameter vector
end
