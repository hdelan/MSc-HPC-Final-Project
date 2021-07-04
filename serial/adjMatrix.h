#ifndef MATRIX_EXP_H_9324
#define MATRIX_EXP_H_9324

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

class adjMatrix {
        public:
                adjMatrix() = delete;
                adjMatrix(const unsigned N, const unsigned edges, std::ifstream & f) :
                        idx(2, std::vector<unsigned>(2*edges)),
                        rows(N, std::vector<unsigned>()),
                        n {N},
                        edge_count {edges}
                        {
                                populate_sparse_matrix(f);
                        };
                adjMatrix(adjMatrix &) = delete;
                adjMatrix & operator=(adjMatrix &) = delete;
                ~adjMatrix() = default;

                unsigned get_n() const { return n;};
                unsigned get_edges() const { return edge_count;};
                friend std::ostream& operator<<(std::ostream & os, const adjMatrix & A) {
                        for (auto i = 0u; i < A.edge_count*2; ++i) {
                                os << "(" << A.idx[0][i] << ", " << A.idx[1][i] << ")\n";
                        }
                        return os;
                };
                void print_full();
                template <typename T>
                friend void sparse_adj_mat_vec_mult(const adjMatrix & A, const std::vector<T> in, std::vector<T> & out);

        private:
                // This will be stored in row-major order
                std::vector<std::vector<unsigned>> idx;
                std::vector<std::vector<unsigned>> rows;

                unsigned n;
                unsigned edge_count;

                void populate_sparse_matrix(std::ifstream & f);
};
#endif
