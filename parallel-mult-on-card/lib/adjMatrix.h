#ifndef ADJ_MATRIX_H_9324
#define ADJ_MATRIX_H_9324

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <sstream>
#include <string>
#include <cmath>
#include <set>
#include <random>

template <typename T>
class eigenDecomp;
template <typename T>
class lanczosDecomp;

class adjMatrix
{
  private:
    // The indexes of the non-zero values (in this case all nonzeros
    // are 1)
    // The adjacency matrix is stored in CSR format. We do not need the third array AA
    // Since it is all ones
    unsigned *row_offset = nullptr; // JA
    unsigned *col_idx = nullptr; // IA

    unsigned n;          // The number of nodes
    unsigned edge_count; // The number of edges

    unsigned barabasi_degree; // The min degree of each node in barabasi graph

    char matrix_type; // 'b' for barabasi 'r' for random 's' for stencil 'f' for read-from-file

    void populate_sparse_matrix(std::ifstream &);

    void generate_sparse_matrix(const char c);

    void stencil_adj();
    void random_adj();
    void barabasi(const unsigned m);

  public:
    //ctors
    adjMatrix() = default;
    adjMatrix(const unsigned N, const unsigned E, std::ifstream &f) : row_offset(new unsigned[N+1]),
    col_idx(new unsigned[2 * E]),
    n{N},
    edge_count{E},
    matrix_type {'f'}
    {
      populate_sparse_matrix(f);      // Reads graph from file
    };
    adjMatrix(const unsigned N, const unsigned m, const char c) : n {N}, 
      matrix_type {c},
      barabasi_degree {m}
    {
      generate_sparse_matrix(c);   // Creates random Barabasi-Albert scale free graph

    }
    adjMatrix(const unsigned N, const unsigned E) : row_offset(new unsigned[N+1]),
    col_idx(new unsigned[2 * E]),
    n{N},
    edge_count{E % (n * (n - 1) / 2 + 1)},
    matrix_type {'r'}
    {
      generate_sparse_matrix('r');    // Creates naive random graph
    }

    // Only a shallow copy is needed
    adjMatrix(adjMatrix & A) : row_offset{A.row_offset},
      col_idx {A.col_idx},
      n {A.n},
      edge_count {A.edge_count},
      matrix_type {A.matrix_type}{};
    // Deleted copy
    adjMatrix &operator=(adjMatrix &) = delete;

    // Move assignment
    adjMatrix &operator=(adjMatrix &&rhs)
    {
      row_offset = rhs.row_offset;
      col_idx = rhs.col_idx;
      n = rhs.n;
      edge_count = rhs.edge_count;
      matrix_type = rhs.matrix_type;

      rhs.row_offset = nullptr;
      rhs.col_idx = nullptr;

      return *this;
    };
    ~adjMatrix()
    {
      if (row_offset != nullptr) { delete[] row_offset; row_offset=nullptr;}
      if (col_idx != nullptr) { delete[] col_idx; col_idx=nullptr;}
    };

    unsigned get_n() const { return n; };
    unsigned get_edges() const { return edge_count; };

    void write_matrix_to_file();

    void print_full() const;

    template <typename T>
      friend void spMV(const adjMatrix &, const T *const, T *const);
    friend std::ostream &operator<<(std::ostream &, const adjMatrix &);
    template <typename T>
      friend void multOut(lanczosDecomp<T> &, eigenDecomp<T> &, adjMatrix &);
    template <typename T>
      friend void cu_multOut(lanczosDecomp<T> &, eigenDecomp<T> &, adjMatrix &);
    template <typename T>
      friend void cu_linalg_test(const unsigned);
    template <typename T>
      friend void cu_SPMV_test(const unsigned, adjMatrix &);
    template <typename T>
      friend void get_blockrows(adjMatrix &, const unsigned, unsigned *, unsigned &);

    template <typename T>
      friend class lanczosDecomp;
};
#endif
