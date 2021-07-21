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
