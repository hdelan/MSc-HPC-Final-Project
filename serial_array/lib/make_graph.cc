#include "adjMatrix.h"
#include "edge.h"
#include <set>

void adjMatrix::random_adj()
{
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<long unsigned> distrib(0, n - 1);

        unsigned n1, n2;

        std::set<Edge> edges;
        while (edges.size() < 2 * edge_count)
        {
                n1 = distrib(gen);
                n2 = distrib(gen);
                while (n1 == n2)
                        n2 = distrib(gen);
                edges.insert(Edge(n1, n2));
                edges.insert(Edge(n2, n1));
        }

        auto i{0u};
        for (auto it = edges.begin(); it != edges.end(); it++)
        {
                row_idx[i] = it->n1;
                col_idx[i] = it->n2;
                i++;
        }
}

void adjMatrix::stencil_adj()
{
        // 2d stencil graph
        auto j = 0u, offset{static_cast<unsigned>(sqrt(n))};
        for (auto i = 0u; i < n; i++)
        {
                for (auto k : {i - offset, i - 1, i + 1, i + offset})
                {
                        if (k < n)
                        {
                                row_idx[j] = i;
                                col_idx[j] = k;
                                j++;
                        }
                }
        }
}

void adjMatrix::barabasi(const unsigned m)
{
        unsigned min_degree = (m > n - 1) ? n-1 : m;
        
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<long unsigned> distrib(0, n - 1);
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        std::set<Edge> edges;

        std::vector<double> degrees(n);

        // Generating initial complete graph kernel
        for (auto i = 0u; i < min_degree + 1; i++)
        {
                for (auto j = i + 1; j < min_degree + 1; j++)
                {
                        edges.insert(Edge(i, j));
                        edges.insert(Edge(j, i));
                }
                degrees[i] = min_degree;
        }
        for (auto i = min_degree + 1; i < n; i++)
        {
                auto prev_size = edges.size();
                for (auto k = 0u; k < min_degree; k++)
                {
                        auto p = prob(gen);
                        for (auto j = 0u; j < edges.size(); j++)
                        {
                                p -= degrees[j] / prev_size;
                                if (p < 0 && edges.find(Edge(i, j)) == edges.end())
                                {
                                        edges.insert(Edge(i, j));
                                        edges.insert(Edge(j, i));
                                        degrees[i]++;
                                        degrees[j]++;
                                        break;
                                }
                        }
                }
        }
        edge_count = edges.size() / 2;

        row_idx = new long unsigned[2 * edge_count];
        col_idx = new long unsigned[2 * edge_count];

        auto i{0u};
        for (auto it = edges.begin(); it != edges.end(); it++)
        {
                row_idx[i] = it->n1;
                col_idx[i] = it->n2;
                i++;
        }
}

// TODO make a function to generate Erdos-Renyi graph
