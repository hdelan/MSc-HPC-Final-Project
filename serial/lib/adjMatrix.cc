/**
 * \file:        adjMatrix.cc
 * \brief:       An adjacency matrix helper constructor
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-15
 */
#include "adjMatrix.h"
#include "edge.h"

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to populate a sparse matrix from file
 *
 * \param:       f
 */
/* ----------------------------------------------------------------------------*/
void adjMatrix::populate_sparse_matrix(std::ifstream &f)
{
        std::set<Edge> edges;
        // This variable holds the number of nonzero values
        // above the diagonal in each column. This is needed in
        // order to
        long unsigned row, col;

        // Putting both upper, lower triangular parts into rows, cols
        for (auto i = 0u; i < edge_count; ++i)
        {
                f >> col >> row;
                edges.insert(Edge(--row, --col));
                edges.insert(Edge(col, row));
        }

        auto i{0u};
        auto prev_row {0u};
        for (auto it = edges.begin(); it != edges.end(); it++)
        {
                while (prev_row != it->n1) row_offset[++prev_row] = i;
                col_idx[i++] = it->n2;
        }
        row_offset[n] = edges.size();
        /*
        auto i {0u};
        for (auto it=edges.begin();it!=edges.end();it++) {
                row_idx[i] = it->n1;
                col_idx[i] = it->n2;
                i++;
        }
        */
        edge_count = edges.size()/2;

}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       A function to generate sparse matrices in a few different ways
 *
 * \param:       c
 */
/* ----------------------------------------------------------------------------*/
void adjMatrix::generate_sparse_matrix(const char c, const unsigned m)
{
        switch (c)
        {
        case 's':
        {
                row_offset = new long unsigned[n*n + 1];
                col_idx = new long unsigned[2 * (2 * n * n - n - 1)];
                edge_count = (2 * n * n - n - 1);
                n = n * n;
                //stencil_adj();
                break;
        }
        case 'b':
        {
                barabasi(m);
                break;
        }
        case 'r':
        {
                random_adj();
                break;
        }
        }
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Overloaded ostream operator
 *
 * \param:       os
 * \param:       A
 *
 * \returns      
 */
/* ----------------------------------------------------------------------------*/
std::ostream &operator<<(std::ostream &os, const adjMatrix &A)
{
        os << "JA\n";
        for (auto i = 0u; i < A.edge_count * 2; ++i)
        {
                os << A.col_idx[i] << " ";
        }
        os << "\nIA\n";
        for (auto i = 0u; i < A.n+1; ++i)
        {
                os << A.row_offset[i] << " ";
        }

        return os;
}
