#ifndef ADJ_MATRIX_H_9324
#define ADJ_MATRIX_H_9324

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

class eigenDecomp;
class lanczosDecomp;

class adjMatrix
{
public:
        adjMatrix() = delete;
        adjMatrix(const unsigned N, const unsigned edges, std::ifstream &f) : row_idx(2 * edges),
                                                                              col_idx(2 * edges),
                                                                              n{N},
                                                                              edge_count{edges}
        {
                populate_sparse_matrix(f);
        };
        adjMatrix(adjMatrix &) = delete;
        adjMatrix &operator=(adjMatrix &) = delete;
        ~adjMatrix() = default;

        unsigned get_n() const { return n; };
        unsigned get_edges() const { return edge_count; };

        template <typename T>
        friend void sparse_adj_mat_vec_mult(const adjMatrix &, const std::vector<T>, std::vector<T> &);
        friend std::ostream &operator<<(std::ostream &, const adjMatrix &);
        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);

private:
        // The indexes of the non-zero values (in this case all nonzeros
        // are 1)
        std::vector<unsigned> row_idx;
        std::vector<unsigned> col_idx;

        unsigned n;          // The number of nodes
        unsigned edge_count; // The number of edges

        void populate_sparse_matrix(std::ifstream &f);
};
#endif
