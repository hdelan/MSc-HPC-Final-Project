function adj = make_graph(n, E)

adj = spalloc(n, n, E);
idx = randperm(n * n, E);
adj(idx) = 1;
adj = min( adj + adj.', 1);

for i = 1:n
    adj(i,i) = 0;
end
return;
end
