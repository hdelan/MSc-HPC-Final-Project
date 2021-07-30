#ifndef EDGE_H_823482834
#define EDGE_H_823482834

struct Edge
{
        long unsigned n1, n2;
        Edge() = default;
        Edge(long unsigned _x, long unsigned _y) : n1{_x}, n2{_y} {};
};

inline bool operator<(const Edge &a, const Edge &b)
{
        return a.n1 < b.n1 || (a.n1 == b.n1 && a.n2 < b.n2);
}

inline bool operator==(const Edge &a, const Edge &b)
{
        return a.n1 == b.n1 && a.n2 == b.n2;
}

#endif