#ifndef ADJ_MATRIX_H_9324
#define ADJ_MATRIX_H_9324

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <set>
#include <random>

class eigenDecomp;
class lanczosDecomp;

class adjMatrix
{
private:
        // The indexes of the non-zero values (in this case all nonzeros
        // are 1)
        long unsigned *row_idx = nullptr;
        long unsigned *col_idx = nullptr;

        long unsigned n;          // The number of nodes
        long unsigned edge_count; // The number of edges

        void populate_sparse_matrix(std::ifstream &);

        void generate_sparse_matrix(const char c, const unsigned m=3);

        void stencil_adj();
        void random_adj();
        void barabasi(const unsigned m);

public:
        //ctors
        adjMatrix() = default;
        adjMatrix(const long unsigned N, const long unsigned E, std::ifstream &f) : row_idx(new long unsigned[2 * E]),
                                                                                    col_idx(new long unsigned[2 * E]),
                                                                                    n{N},
                                                                                    edge_count{E}
        {
                populate_sparse_matrix(f);      // Reads graph from file
        };
        adjMatrix(const long unsigned N, const unsigned m, const char c) : n {N}
        {
                generate_sparse_matrix(c, m);   // Creates random Barabasi-Albert scale free graph
        }
        adjMatrix(const long unsigned N, const long unsigned E) : row_idx(new long unsigned[2 * E]),
                                                                  col_idx(new long unsigned[2 * E]),
                                                                  n{N},
                                                                  edge_count{E % (n * (n - 1) / 2 + 1)}
        {
                generate_sparse_matrix('r');    // Creates naive random graph
        }

        // Deleted copy
        adjMatrix(adjMatrix &) = delete;
        adjMatrix &operator=(adjMatrix &) = delete;

        // Move assignment
        adjMatrix &operator=(adjMatrix &&rhs)
        {
                row_idx = rhs.row_idx;
                col_idx = rhs.col_idx;
                n = rhs.n;
                edge_count = rhs.edge_count;

                rhs.row_idx = nullptr;
                rhs.col_idx = nullptr;

                return *this;
        };
        ~adjMatrix()
        {
                if (row_idx != nullptr)
                        delete[] row_idx;
                if (col_idx != nullptr)
                        delete[] col_idx;
        };

        long unsigned get_n() const { return n; };
        long unsigned get_edges() const { return edge_count; };

        void print_full() const;

        template <typename T>
        friend void sparse_adj_mat_vec_mult(const adjMatrix &, const T *const, T *const);
        friend std::ostream &operator<<(std::ostream &, const adjMatrix &);
        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend void get_raw_upper_matrix(double *, adjMatrix &);
        friend void get_raw_full_matrix(double *, adjMatrix &);

        friend class lanczosDecomp;
};
#endif
