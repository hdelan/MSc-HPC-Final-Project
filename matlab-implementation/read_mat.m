function A = read_mat(adj, n)
    [E,~] = size(adj);
    A = spalloc(n,n,2*E);
    for i=1:E
        A(adj(i,1), adj(i,2)) = 1;
        A(adj(i,2), adj(i,1)) = 1;
    end
end