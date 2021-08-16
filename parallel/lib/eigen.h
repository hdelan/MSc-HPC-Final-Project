#ifndef EIGEN_H_732234189
#define EIGEN_H_732234189

#include "cu_lanczos.h"
#include "adjMatrix.h"
#include <lapacke.h>
#include <vector>

class eigenDecomp
{
public:
        eigenDecomp() = delete;
        eigenDecomp(lanczosDecomp &_L) : eigenvalues(new double[_L.krylov_dim]),
                                        eigenvectors(new double[_L.krylov_dim*_L.krylov_dim]),
                                        L {_L}
        {
                for (auto i=0u;i<L.get_krylov();i++) eigenvalues[i] = L.alpha[i];
                decompose();
        };
        eigenDecomp(eigenDecomp &) = delete;
        eigenDecomp &operator=(eigenDecomp &) = delete;
        ~eigenDecomp() {
                delete[] eigenvalues;
                delete[] eigenvectors;
        };

        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend void cu_multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend std::ostream &operator<<(std::ostream &, eigenDecomp &);

private:
        double * eigenvalues;
        double * eigenvectors;

        lanczosDecomp & L;

        void decompose();
};
#endif
