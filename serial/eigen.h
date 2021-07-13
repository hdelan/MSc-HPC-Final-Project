#ifndef EIGEN_H_732234189
#define EIGEN_H_732234189

#include "lanczos.h"
#include "adjMatrix.h"
#include <lapacke.h>
#include <vector>

class eigenDecomp {
        public: 
                eigenDecomp() = delete;
                eigenDecomp(lanczosDecomp & D) : 
                        eigenvalues(D.alpha.begin(), D.alpha.end()),
                        eigenvectors(D.krylov_dim*D.krylov_dim)
        {
                decompose(D);
        };
                eigenDecomp(eigenDecomp &)=delete;
                eigenDecomp & operator=(eigenDecomp &)=delete;
                ~eigenDecomp()=default;
                
                friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);

                friend std::ostream & operator<<(std::ostream & os, eigenDecomp & E) {
                        os << "Eigenvalues:\n";
                        for (auto it=E.eigenvalues.begin(); it!=E.eigenvalues.end();it++) {
                                os << *it << " \t";
                        }
                        os << "\n\n";
                        auto n {E.eigenvalues.size()};
                        os << "Eigenvectors:\n";
                        for (auto i=0u; i<n; ++i) {
                                for (auto it=E.eigenvectors.begin()+i*n; it!=E.eigenvectors.begin()+(i+1)*n;it++) {
                                        os << *it << " \t";
                                }
                                os << '\n';
                        }
                        return os;
                };

        private:
                std::vector<double> eigenvalues;
                std::vector<double> eigenvectors;

                void decompose(lanczosDecomp & D);
};
#endif
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
                        ans(N),
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
                friend void sparse_adj_mat_vec_mult(const adjMatrix &, const std::vector<T>, std::vector<T>&);
                friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);

        private:
                // This will be stored in row-major order
                std::vector<std::vector<unsigned>> idx;
                std::vector<std::vector<unsigned>> rows;
                
                std::vector<double> ans;

                unsigned n;
                unsigned edge_count;

                void populate_sparse_matrix(std::ifstream & f);
};
#endif
