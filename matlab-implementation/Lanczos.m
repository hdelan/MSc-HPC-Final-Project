function [alpha, beta,Q] = Lanczos(A, v, k)

Q(:, 1) = v/norm(v);
for j=1:k
    v = A*Q(:,j);
    alpha(j) = Q(:,j)'*v;
    v = v-alpha(j)*Q(:,j);
    if j > 1
        v = v-beta(j-1)*Q(:,j-1);
    end
    if j < k 
        beta(j) = norm(v);
        Q(:,j+1) = v/beta(j);
    end
end