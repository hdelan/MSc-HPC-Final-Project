n = 6; E = 15;

v0 = ones(n,1)./sqrt(n);

tol = 1e-5;

adj = spalloc(n, n, E);
idx = randperm(n * n, E);
adj(idx) = 1;
adj = min( adj + adj.', 1);

for i = 1:n
    adj(i,i) = 0;
end



plot(graph(adj));

