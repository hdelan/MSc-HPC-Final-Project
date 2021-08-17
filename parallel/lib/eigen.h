#ifndef EIGEN_H_732234189
#define EIGEN_H_732234189

#include "cu_lanczos.h"
#include "adjMatrix.h"
#include <lapacke.h>
#include <vector>

template <typename T>
class eigenDecomp
{
public:
        eigenDecomp() = delete;
        eigenDecomp(lanczosDecomp<T> &_L) : eigenvalues(new T[_L.krylov_dim]),
                                        eigenvectors(new T[_L.krylov_dim*_L.krylov_dim]),
                                        L {_L}
        {
                for (auto i=0u;i<L.get_krylov();i++) eigenvalues[i] = L.alpha[i];
                decompose();
        };
        eigenDecomp(eigenDecomp<T> &) = delete;
        eigenDecomp &operator=(eigenDecomp<T>&) = delete;
        ~eigenDecomp() {
                delete[] eigenvalues;
                delete[] eigenvectors;
        };

        friend void multOut(lanczosDecomp<T> &, eigenDecomp<T> &, adjMatrix &);
        friend void cu_multOut(lanczosDecomp<T> &, eigenDecomp<T> &, adjMatrix &);

private:
        T * eigenvalues;
        T * eigenvectors;

        lanczosDecomp<T> & L;

        void decompose();
};
#endif
