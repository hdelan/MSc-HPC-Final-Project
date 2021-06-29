clear; clc;
n = 100; E = idivide(int32(n), 5);

v0 = ones(n,1)./sqrt(n);

tol = 1e-5;

adj = spalloc(n, n, E);
idx = randperm(n * n, E);
adj(idx) = 1;
adj = min( adj + adj.', 1);

for i = 1:n
    adj(i,i) = 0;
end
full(adj)

% adj = rand(n, n);
% adj = adj + adj';

q = zeros(n, n);

q(:,1) = rand(n,1);
q(:,1) = q(:,1)/norm(q(:,1));
h = zeros(n,n);

for j = 1:n
    w = adj*q(:,j);
    for i = 1:j
        h(i,j) = w'*q(:,i);
        w = w - h(i,j)*q(:,i);
    end
    h(j+1,j) = norm(w);
    q(:,j+1) = w/h(j+1,j);
end

<<<<<<< HEAD
% h is now in tridiagonal form

[h_eigvec, h_eigvals] = eigs(h(1:100, 1:100)); 

=======
Q = q(:,1:n);
Q
>>>>>>> 82704f79b2b2287194cad470594693e4d9e40f03
%plot(graph(adj));

