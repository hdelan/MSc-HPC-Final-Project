#ifndef EIGEN_H_732234189
#define EIGEN_H_732234189

#include "lanczos.h"
#include "adjMatrix.h"
#include <lapacke.h>
#include <vector>

class eigenDecomp
{
public:
        eigenDecomp() = delete;
        eigenDecomp(lanczosDecomp &D) : eigenvalues(new double(D.n)),
                                        eigenvectors(new double (D.krylov_dim * D.krylov_dim)),
                                        n {D.n},
                                        k {D.krylov_dim}
        {
                decompose(D);
        };
        eigenDecomp(eigenDecomp &) = delete;
        eigenDecomp &operator=(eigenDecomp &) = delete;
        ~eigenDecomp() {
                delete[] eigenvalues;
                delete[] eigenvectors;
        };

        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend std::ostream &operator<<(std::ostream &, eigenDecomp &);

private:
        double * eigenvalues;
        double * eigenvectors;

        const unsigned n;
        const unsigned k;

        void decompose(lanczosDecomp &);
};
#endif
