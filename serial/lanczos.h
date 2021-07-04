#ifndef LANCZOS_H_1238249102
#define LANCZOS_H_1238249102

#include <vector>
#include <cmath>
#include <numeric>
#include "adjMatrix.h"

class lanczosDecomp {
        public:
                lanczosDecomp() = delete;
                lanczosDecomp(adjMatrix & A, unsigned krylov, std::vector<double> starting_vec) :
                        alpha(krylov),
                        beta(krylov-1),
                        Q(krylov, std::vector<double>(A.get_n())),
                        x {starting_vec},
                        krylov_dim {krylov} 
                {
                        decompose(A);
                };
                lanczosDecomp(lanczosDecomp &) = delete;
                lanczosDecomp& operator=(lanczosDecomp &) = delete;
                ~lanczosDecomp() = default;

                friend std::ostream& operator<<(std::ostream & os, const lanczosDecomp & D) {
                        os << "Alpha: \n";
                        auto n {D.Q[0].size()};
                        for (auto j=0u;j<D.krylov_dim;j++){
                                os << D.alpha.at(j) << " ";
                        }
                        os << "\nBeta: \n";
                        for (auto j=0u;j<D.krylov_dim-1;j++) {
                                os << D.beta.at(j) << " ";
                        }
                        os << "\nQ:\n";
                        for (auto j=0u;j<n;j++) {
                                for (auto k=0u;k<D.krylov_dim;k++) {
                                        os << D.Q.at(k).at(j) << " ";
                                }
                                os << '\n';
                        }

                        os << '\n';
                        return os;
                };


        private:
                std::vector<double> alpha;
                std::vector<double> beta;
                std::vector<std::vector<double>> Q;
                std::vector<double> x;

                unsigned krylov_dim;

                void decompose(const adjMatrix & A);

};
#endif
