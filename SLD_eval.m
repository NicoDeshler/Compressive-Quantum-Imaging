function S_stack = SLD_eval(F_stack, R)
% A function for evaluating S in the implicit matrix equation for the
% Symmetric Logarithmic Derivative
%                       2F = RS + SR
% This method requires that F and R be Hermitian.

% --------
% Inputs:
% --------
% F_stack   - A stack of matrices corresponding to the operator F in the
% implicit SLD equation
% R         - A matrix corresponding to the operator R in the implicit SLD
% equation
% --------
% Outputs:
% --------
% S_stack   - solutions to the implicit SLD equations

% Add dimensionality to F if its not a matrix stack (i.e. just 1 matrix)
is_stack = 1;
if length(size(F_stack)) < 3
    F_stack = reshape(F_stack,[size(F_stack),1]); 
    is_stack = 0;
end


% Check that the inputs are Hermitian
assert(ishermitian(R),'R matrix in implicit SLD equation is not Hermitian');
for j =1:size(F_stack,3)
    assert(ishermitian(F_stack(:,:,j)),['Matrix ',num2str(j),' in F_stack is not Hermitian']);
end

% Get the eigenvectors/values of R
[V, d] = eig(R,'vector');

% Loop method (slow but works)
S_stack = zeros(size(F_stack));
for i = 1:size(F_stack,1)
    for j = 1:size(F_stack,2)
        for k = 1:size(F_stack,3)
            if d(i)+d(j) ~= 0
                S_stack(:,:,k) = S_stack(:,:,k) + 1/2 * (V(:,i)'*F_stack(:,:,k)*V(:,j))/(d(i)+d(j)) * V(:,i)*V(:,j)';
            end
        end
    end
end



%{
% The columns of V corresponding to non-zero eigenvalues span the
% perpendicular space. To span the entire Hilbert space we need to build 
% the tangent space vectors using Gram Schmidt.

% Isolate the perp-space vectors
eps = 1e-12; % threshold value for numerical stability
V(:, abs(d)<eps) = [];

% Get the tangent space dimensionality
tan_space_dim = length(d(abs(d)<eps));

% Run Gram Schmidt procedure to build V into an orthogonal matrix that spans
% the Hilbert space.
for i = 1:tan_space_dim
    v = rand(size(V,1),1);
    proj = V'*v;
    v = v - sum(V*diag(proj),2);
    v = v/norm(v);
    V = [V,v];
end

%} 


%{
% Fast matrix method (might be wrong - compare output to loop method!)
for k = 1:size(F_stack,3)
    F = F_stack(:,:,k);
    D1 = repmat(d,[1,numel(d)]);
    D2 = D1';
    eigval_pairs = D1+D2;
    eigval_pairs(eigval_pairs~=0) = 1./eigval_pairs(eigval_pairs ~= 0); 
    F_proj = V'*F*V;    
    dyads = mat2cell(V(:)*V(:)',size(V),size(V));
    
    dyad_stack = cellfun(@(x)reshape(x,[1,1,[]]),dyads,'un',0);
    dyad_stack = cell2mat(dyad_stack);
    dyads_stack = squeeze(permute(dyads_stack,[3,4,2,1]));
    C = reshape(eigval_pairs(:)*F_proj(:),[1,1,numel(V)]);
    S_stack(:,:,k) = V'*sum(C.*dyad_stack,3)*V;
end
%}

% Remove added dimension if F was not a stack
if ~is_stack
    S_stack = squeeze(S_stack);
end

end