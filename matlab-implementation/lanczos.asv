clear; clc;
n = 10000; E = 150; %idivide(int32(n), 5);
krylov_dim = 40;

% starting vector
v0 = ones(n,1);

tol = 1e-5;

adj = make_graph(n, E);
% plot(graph(adj));

v = zeros(n, krylov_dim);
w = zeros(n, krylov_dim);

v(:,1) = v0/norm(v0);

alpha = zeros(krylov_dim,1);
beta = zeros(krylov_dim,1);

for j = 1:krylov_dim
    if j > 1
        beta(j) = norm(w(:,j-1));
        if beta(j) > 0
            v(:,j) = w(:,j-1)/beta(j);
        end
    end
    w_tmp = adj*v(:,j);
    alpha(j) = w_tmp'*v(:,j);
    if j > 1
        w(:,j) = w_tmp - alpha(j)*v(:,j) - beta(j)*v(:,j-1);
    else
        w(:,j) = w_tmp - alpha(j)*v(:,j);
    end
end


h = zeros(krylov_dim, krylov_dim);
for j=1:krylov_dim
    h(j,j) = alpha(j);
end
for j=2:krylov_dim
    h(j-1,j) = beta(j);
    h(j,j-1) = beta(j);
end

tmp = expm(h);

y = norm(v0)*w*tmp*w';
y = y(:,1);

check = expm(adj)*v0;

diff = check - y;

