#include "multiplyOut.h"
//#include "lapacke.h"
#include "helpers.h"
#include <cblas.h>
#include <iomanip>

template <typename T>
void multOut(lanczosDecomp<T> & L, eigenDecomp<T> & E, adjMatrix & A) {

        auto n {L.get_n()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) my_exp_func(E.eigenvalues[j]);
        //printf("\nSome values from Q serial: %E %E %E\n", L.Q[0], L.Q[L.krylov_dim], L.Q[2*L.krylov_dim]);
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        auto k = L.get_krylov();
        //cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, L.Q, k, E.eigenvectors, k, 0, QV, k);
        std::vector<T> tmp(k);
        //printf("\nSome values from QV serial: %E %E %E\n", QV[0], QV[L.krylov_dim], QV[2*L.krylov_dim]);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, k, k, 1, E.eigenvectors, k, &E.eigenvalues[0], 1, 0, &tmp[0],1);
        
        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, k, 1, L.Q, k, &tmp[0], 1, 0, &L.ans[0],1);
        
}

template void multOut(lanczosDecomp<double>&, eigenDecomp<double>&, adjMatrix &);
template void multOut(lanczosDecomp<float>&, eigenDecomp<float>&, adjMatrix &);
