n = 6; E = 10;

v0 = ones(n,1)./sqrt(n);

tol = 1e-5;

adj = spalloc(n, n, E);
idx = randperm(n * n, E);
adj(idx) = 1;
adj = min( adj + adj.', 1);

for i = 1:n
    adj(i,i) = 0;
end

space_rank = 3;

v = zeros(n, space_rank+1);
v(:,1) = rand(n, 1);
v(:,1) = v(:,1)/norm(v(:,1));

for i = 2:space_rank+1
    v(:,i) = adj*v(:,i-1);
    v(:,i) = v(:,i)/norm(v(:,i));






