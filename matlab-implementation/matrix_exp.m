%% Setting up
clear; clc;
%n = 50; E = 70; %idivide(int32(n), 5);
%krylov_dim = 7;

adj = [0 0 1 1 1 ; 0 0 1 0 0 ; 1 1 0 0 0 ; 1 0 0 0 1 ; 1 0 0 1 0];
load NotreDame_yeast.mtx;
adj = read_mat(NotreDame_yeast, 2114);n=2114;E = 2240;
krylov_dim=2;

[vecs, vals] = eigs(adj);

% starting vector
x = vecs(:,1); % x = rand(n,1);


%% Make graph and run Lanczos iteration

%adj = make_graph(n, E);
% plot(graph(adj));

[w, alpha, beta, Q] = Lanczos(adj, x, krylov_dim);

%% Turn alpha, beta into T
T = zeros(krylov_dim, krylov_dim);
for j=1:krylov_dim
    T(j,j) = alpha(j);
end
for j=2:krylov_dim
    T(j-1,j) = beta(j-1);
    T(j,j-1) = beta(j-1);
end

%% check that f(A)*x = norm(x)*Q*f(T)*e1
%  Sanity check: check for f(A) = A

LHS = adj*x;
RHS = norm(x)*Q*T;
RHS = RHS(:,1);

diff = LHS - RHS;
display("Norm of error f(A) = A:      "+norm(diff));

%% Now to check for f(A) = expm(A)

LHS = expm(adj)*x;
RHS = norm(x)*Q*expm(T);
RHS = RHS(:,1);

diff = LHS - RHS;
display("Norm of error f(A) = exp(A): "+norm(diff));

%% Now to use the decomposition f(T) = V*f(lambda)*V'

[V, lambda] = eig(T);

LHS = LHS; % same as before
RHS = norm(x)*Q*V*expm(lambda)*V(1,:)';

diff = LHS - RHS;
display("Norm of error with eig decomp f(A) = exp(A): "+norm(diff));
